import torch
import torch.nn as nn
from models.encoder import CNNEncoder, ResNetEncoder
from models.decoder import SpatialBroadcastDecoder
from models.attention import SlotAttention
from models.predictor import SlotPredictor
from models.misc import GradientReversal, AffineCoupling


class SlotDynamicsModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pretrain = getattr(config, 'pretrain', False)
        self.jepa = getattr(config, 'jepa', False)
        self.jepa_alpha = getattr(config, 'jepa_alpha', 0.996)
        burnin = getattr(config, 'burnin_frames', 6)
        rollout = getattr(config, 'rollout_frames', 10)
        self.total_frames = burnin + rollout

        enc_hidden = getattr(config, 'encoder_hidden', 32)
        if isinstance(enc_hidden, (list, tuple)):
            enc_hidden = enc_hidden[0]

        img_sz = getattr(config, 'img_size', 64)
        if isinstance(img_sz, (list, tuple)):
            img_sz = img_sz[0]

        in_channels = getattr(config, 'in_channels', 3)
        static_dim = getattr(config, 'static_dim', 128)
        dynamic_dim = getattr(config, 'dynamic_dim', 128)
        slot_dim = static_dim + dynamic_dim
        num_slots = getattr(config, 'num_slots', 7)

        enc_type = getattr(config, 'encoder_type', 'cnn')
        if enc_type == 'cnn':
            self.encoder = CNNEncoder(
                in_channels=in_channels, hidden_channels=enc_hidden,
                out_dim=slot_dim, img_size=img_sz,
                pos_embedding=None, reduction='flatten',
            )
        else:
            self.encoder = ResNetEncoder(
                resnet_version=getattr(config, 'resnet_version', 18),
                pretrained=getattr(config, 'resnet_pretrained', False),
                out_dim=slot_dim, pos_embedding=None, reduction='none',
            )

        self.slot_attention = SlotAttention(
            num_slots=num_slots, slot_dim=slot_dim,
            hidden_dim=getattr(config, 'slot_hidden', 128),
            iters=getattr(config, 'slot_iters', 3),
        )

        dec_hidden = getattr(config, 'decoder_hidden', 64)
        if isinstance(dec_hidden, (list, tuple)):
            dec_hidden = dec_hidden[0]
        bs = getattr(config, 'broadcast_size', 8)
        if isinstance(bs, (list, tuple)):
            bs = bs[0]
        self.decoder = SpatialBroadcastDecoder(
            slot_dim=slot_dim, output_channels=in_channels,
            hidden_channels=dec_hidden, broadcast_size=bs, img_size=img_sz,
            num_slots=num_slots, predict_mask=True, use_alpha=True,
        )

        # f_θ_Z：可逆变换 S ↔ Z（按 static_dim / dynamic_dim 拆分）
        self.f_z = AffineCoupling(
            static_dim=static_dim,
            dynamic_dim=dynamic_dim,
            hidden_dim=getattr(config, 'hidden_dim', 256),
        )

        self.predictor = SlotPredictor(config)

        # GRL + 反向预测器（idea.md §4）
        self.grl = GradientReversal()
        self.mlp_rev = nn.Linear(dynamic_dim, static_dim)

        # JEPA 组件（E2E 自监督）
        if self.jepa:
            if enc_type == 'cnn':
                self.target_encoder = CNNEncoder(
                    in_channels=in_channels, hidden_channels=enc_hidden,
                    out_dim=slot_dim, img_size=img_sz,
                    pos_embedding=None, reduction='flatten',
                )
            else:
                self.target_encoder = ResNetEncoder(
                    resnet_version=getattr(config, 'resnet_version', 18),
                    pretrained=False,
                    out_dim=slot_dim, pos_embedding=None, reduction='none',
                )
            self.target_encoder.load_state_dict(self.encoder.state_dict())
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)

            self.target_slot_attention = SlotAttention(
                num_slots=num_slots, slot_dim=slot_dim,
                hidden_dim=getattr(config, 'slot_hidden', 128),
                iters=getattr(config, 'slot_iters', 3),
            )
            self.target_slot_attention.load_state_dict(self.slot_attention.state_dict())
            for p in self.target_slot_attention.parameters():
                p.requires_grad_(False)

        if hasattr(torch, 'compile'):
            self.decoder = torch.compile(self.decoder)
            self.encoder = torch.compile(self.encoder)
            if self.jepa:
                self.target_encoder = torch.compile(self.target_encoder)

    POS_EMBED_DIM = 32  # S 空间中位置编码占用的维度数，固定不变，与 Z 空间的划分解耦

    def _add_sd_pos_encoding(self, slots, attn, grid_sz):
        B, N, _ = slots.shape
        D_pos = self.POS_EMBED_DIM
        pos_start = slots.shape[-1] - D_pos
        attn_2d = attn.view(B, N, grid_sz, grid_sz)
        gy = torch.linspace(-1, 1, grid_sz, device=slots.device)
        gx = torch.linspace(-1, 1, grid_sz, device=slots.device)
        gy, gx = torch.meshgrid(gy, gx, indexing='ij')
        pos_y = (attn_2d * gy[None, None]).sum(dim=[2, 3])
        pos_x = (attn_2d * gx[None, None]).sum(dim=[2, 3])
        half = D_pos // 4
        freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=slots.device).float() / half))
        pe = torch.stack([
            torch.sin(pos_y.unsqueeze(-1) * freq), torch.cos(pos_y.unsqueeze(-1) * freq),
            torch.sin(pos_x.unsqueeze(-1) * freq), torch.cos(pos_x.unsqueeze(-1) * freq),
        ], dim=-1).reshape(B, N, D_pos)
        out = slots.clone()
        out[:, :, pos_start:] = out[:, :, pos_start:] + pe
        return out

    def _sa(self, feat_t, slots, t, attn_module=None, iters_first=3, iters_rest=1):
        '''封装 slot_attention 调用，第一帧用 iters_first，后续用 iters_rest。'''
        mod = attn_module or self.slot_attention
        mod.iters = iters_first if t == 0 else iters_rest
        return mod(feat_t, slots)

    def _forward_pretrain(self, frames):
        '''预训练：仅 STATM-SAVi，无 rollout，仅 burnin 重建。'''
        B, T, C, H, W = frames.shape
        burnin = self.config.burnin_frames
        static_dim = getattr(self.config, 'static_dim', 128)

        enc_features = self.encoder(frames)
        B, T, N, D = enc_features.shape
        grid_sz = int(N ** 0.5)

        burnin_slots = []
        slots = None
        for t in range(burnin):
            feat_t = enc_features[:, t]
            slots, attn = self._sa(feat_t, slots, t)
            slots = self._add_sd_pos_encoding(slots, attn, grid_sz)
            burnin_slots.append(slots)
        burnin_slots = torch.stack(burnin_slots, dim=1)

        dec_burnin = torch.stack([self.decoder(burnin_slots[:, t]) for t in range(burnin)], dim=1)

        all_slots = burnin_slots
        slots_dyn = all_slots[:, :, :, static_dim:]
        rev_pred = self.mlp_rev(self.grl(slots_dyn))
        S_c = all_slots[:, :, :, :static_dim]

        return {
            "outputs": {
                "video_burnin": dec_burnin,
                "video_pred": None,
                "video_target": None,
            },
            "slots": {
                "corrected": burnin_slots,
                "predicted": None,
                "target": None,
            },
            "rev_pred": rev_pred,
            "S_c": S_c,
            "energy_pairs": None,
        }

    def _forward_finetune(self, frames):
        B, T, C, H, W = frames.shape
        burnin = self.config.burnin_frames
        rollout = self.config.rollout_frames
        static_dim = getattr(self.config, 'static_dim', 128)
        dynamic_dim = getattr(self.config, 'dynamic_dim', 128)
        slot_dim = static_dim + dynamic_dim

        with torch.no_grad():
            enc_features = self.encoder(frames)
        B_e, T_e, N_feat, D_e = enc_features.shape
        grid_sz = int(N_feat ** 0.5)

        # Phase 2: Burnin → S（frozen）→ Z
        buf_sz = getattr(self.config, 'buffer_len', burnin + rollout)
        Z_buffer = torch.zeros(B, buf_sz, self.config.num_slots, slot_dim, device=frames.device)
        burnin_S = []
        burnin_Z = []
        slots = None
        for t in range(burnin):
            feat_t = enc_features[:, t]
            with torch.no_grad():
                slots, attn = self._sa(feat_t, slots, t)
                slots = self._add_sd_pos_encoding(slots, attn, grid_sz)
            Z_t = self.f_z(slots)
            burnin_S.append(slots)
            burnin_Z.append(Z_t)
            Z_buffer[:, t] = Z_t
        burnin_S = torch.stack(burnin_S, dim=1)
        burnin_Z = torch.stack(burnin_Z, dim=1)

        # Phase 3: Rollout 在 Z 空间
        freeze_C = getattr(self.config, 'freeze_C', False)
        global_C = self.predictor.compute_C(burnin_Z) if freeze_C else None
        pred_Z_list = []
        energy_pairs = []
        cur_Z = Z_t
        for t in range(rollout):
            C_use = global_C if freeze_C else cur_Z[:, :, :static_dim]
            out = self.predictor(cur_Z, Z_buffer[:, :burnin + t], C=C_use, return_energy=True)
            next_Z, ep = out if isinstance(out, tuple) else (out, None)
            pred_Z_list.append(next_Z)
            if ep is not None:
                energy_pairs.append(ep)
            if burnin + t < buf_sz:
                Z_buffer[:, burnin + t] = next_Z
            cur_Z = next_Z
        pred_Z = torch.stack(pred_Z_list, dim=1)

        # Phase 4: Target S（frozen → S，no_grad）— 用于 slot loss 和 decoder target
        with torch.no_grad():
            target_S_list = []
            s = slots
            for t in range(burnin, burnin + rollout):
                feat_t = enc_features[:, t]
                s, attn = self._sa(feat_t, s, t)
                s = self._add_sd_pos_encoding(s, attn, grid_sz)
                target_S_list.append(s)
            target_S = torch.stack(target_S_list, dim=1)

        # Phase 5: 解码 + slot loss（S 空间）
        dec_burnin = torch.stack([self.decoder(burnin_S[:, t]) for t in range(burnin)], dim=1)
        pred_S_rollout = [self.f_z.inverse(pred_Z[:, t]) for t in range(rollout)]
        pred_S = torch.stack(pred_S_rollout, dim=1)
        dec_pred = torch.stack([self.decoder(s) for s in pred_S_rollout], dim=1)
        dec_target = torch.stack([self.decoder(target_S[:, t]) for t in range(rollout)], dim=1)

        # GRL 在 Z 空间
        all_Z = torch.cat([burnin_Z, pred_Z], dim=1)
        Z_dyn = all_Z[:, :, :, static_dim:]
        rev_pred = self.mlp_rev(self.grl(Z_dyn))
        Z_c = all_Z[:, :, :, :static_dim]

        return {
            "outputs": {
                "video_burnin": dec_burnin,
                "video_pred": dec_pred,
                "video_target": dec_target,
            },
            "slots": {
                "corrected": burnin_S,
                "predicted": pred_S,
                "target": target_S,
            },
            "rev_pred": rev_pred,
            "S_c": Z_c,
            "energy_pairs": energy_pairs if energy_pairs else None,
        }

    def _forward_jepa(self, frames):
        '''E2E 训练 + JEMA 自监督（使用 EMA target 作为自监督目标）。
        复用 rollout 作为 predictor，slot_pred_loss = MSE(pred_S, target_S) 即 JEPA 损失。
        输出格式与 _forward_finetune 完全一致，trainer 无需区分。'''
        B, T, C, H, W = frames.shape
        burnin = self.config.burnin_frames
        rollout = self.config.rollout_frames
        static_dim = getattr(self.config, 'static_dim', 128)
        dynamic_dim = getattr(self.config, 'dynamic_dim', 128)
        slot_dim = static_dim + dynamic_dim

        # === Online path（可训练，S 空间）===
        enc_features = self.encoder(frames)
        B_e, T_e, N_feat, D_e = enc_features.shape
        grid_sz = int(N_feat ** 0.5)

        # buffer = cat(burnin, ...) like _forward_finetune
        burnin_S = []
        slots = None
        for t in range(burnin):
            feat_t = enc_features[:, t]
            slots, attn = self._sa(feat_t, slots, t)
            slots = self._add_sd_pos_encoding(slots, attn, grid_sz)
            burnin_S.append(slots)
        burnin_S = torch.stack(burnin_S, dim=1)

        # === Rollout（S 空间，predictor 作为 predictor，含物理+时空推理）===
        buf_sz = getattr(self.config, 'buffer_len', burnin + rollout)
        slot_buffer = torch.zeros(B, buf_sz, self.config.num_slots, slot_dim, device=frames.device)
        for t in range(burnin):
            slot_buffer[:, t] = burnin_S[:, t]
        freeze_C = getattr(self.config, 'freeze_C', False)
        global_C = self.predictor.compute_C(burnin_S) if freeze_C else None
        pred_S_list = []
        energy_pairs = []
        cur_S = slots
        for t in range(rollout):
            C_use = global_C if freeze_C else cur_S[:, :, :static_dim]
            out = self.predictor(cur_S, slot_buffer[:, :burnin + t], C=C_use, return_energy=True)
            next_S, ep = out if isinstance(out, tuple) else (out, None)
            pred_S_list.append(next_S)
            if ep is not None:
                energy_pairs.append(ep)
            if burnin + t < buf_sz:
                slot_buffer[:, burnin + t] = next_S
            cur_S = next_S
        pred_S = torch.stack(pred_S_list, dim=1)

        # === Target（EMA，no_grad，同时作为 rollout target 和 JEPA target）===
        with torch.no_grad():
            target_enc = self.target_encoder(frames)
            target_slots = []
            s_target = None
            for t in range(burnin + rollout):
                feat_t = target_enc[:, t]
                s_target, attn_t = self._sa(feat_t, s_target, t, attn_module=self.target_slot_attention)
                s_target = self._add_sd_pos_encoding(s_target, attn_t, grid_sz)
                target_slots.append(s_target)
            target_all = torch.stack(target_slots, dim=1)

        # === 解码 ===
        dec_burnin = torch.stack([self.decoder(burnin_S[:, t]) for t in range(burnin)], dim=1)
        dec_pred = torch.stack([self.decoder(pred_S[:, t]) for t in range(rollout)], dim=1)
        dec_target = torch.stack([self.decoder(target_all[:, burnin + t]) for t in range(rollout)], dim=1)

        # GRL（S 空间）
        all_S = torch.cat([burnin_S, pred_S], dim=1)
        S_dyn = all_S[:, :, :, static_dim:]
        rev_pred = self.mlp_rev(self.grl(S_dyn))
        S_c = all_S[:, :, :, :static_dim]

        return {
            "outputs": {
                "video_burnin": dec_burnin,
                "video_pred": dec_pred,
                "video_target": dec_target,
            },
            "slots": {
                "corrected": burnin_S,
                "predicted": pred_S,
                "target": target_all[:, burnin:],
            },
            "rev_pred": rev_pred,
            "S_c": S_c,
            "energy_pairs": energy_pairs if energy_pairs else None,
        }

    def forward(self, frames):
        if self.pretrain:
            return self._forward_pretrain(frames)
        if self.jepa:
            return self._forward_jepa(frames)
        return self._forward_finetune(frames)
