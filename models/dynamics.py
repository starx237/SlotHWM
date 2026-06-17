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

        # Finetune 模式：冻结 STATM-SAVi 模块
        if getattr(config, 'freeze_slot', False):
            for name in ['encoder', 'slot_attention', 'decoder']:
                mod = getattr(self, name, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(False)
            print("Frozen: encoder + slot_attention + decoder (STATM-SAVi)")

    POS_EMBED_DIM = 32  # S 空间中位置编码占用 32 维（8 频率 × 2 轴 × sin/cos）

    @staticmethod
    def _compute_slot_centroid(attn, grid_sz):
        '''从注意力图提取每个 slot 的空间质心 (pos_y, pos_x)。
        attn: (B, num_heads, K, N) or (B, K, N)'''
        if attn.dim() == 4:
            attn = attn.mean(dim=1)  # average over heads → (B, K, N)
        B, K, N = attn.shape
        attn_2d = attn.view(B, K, grid_sz, grid_sz)
        gy = torch.linspace(-1, 1, grid_sz, device=attn.device)
        gx = torch.linspace(-1, 1, grid_sz, device=attn.device)
        gy, gx = torch.meshgrid(gy, gx, indexing='ij')
        pos_y = (attn_2d * gy[None, None]).sum(dim=[2, 3])
        pos_x = (attn_2d * gx[None, None]).sum(dim=[2, 3])
        return torch.stack([pos_y, pos_x], dim=-1)  # (B, K, 2)

    @staticmethod
    def _reconstruct_pe(pos_yx):
        '''从 (pos_y, pos_x) 重建完整 32 维位置编码。'''
        B, N = pos_yx.shape[:2]
        pos_y, pos_x = pos_yx[:, :, 0], pos_yx[:, :, 1]
        D_pos = 32
        half = D_pos // 4
        device = pos_yx.device
        freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
        pe = torch.stack([
            torch.sin(pos_y.unsqueeze(-1) * freq), torch.cos(pos_y.unsqueeze(-1) * freq),
            torch.sin(pos_x.unsqueeze(-1) * freq), torch.cos(pos_x.unsqueeze(-1) * freq),
        ], dim=-1).reshape(B, N, D_pos)
        return pe

    @staticmethod
    def _encode_pos_to_zd(pos_yx, pos_enc_dim):
        '''从 (pos_y, pos_x) 编码位置向量 p 用于 Z^d。
        前 4 维始终是基频率 (ω₀=1) 的 sin_y, cos_y, sin_x, cos_x，
        用于无损提取 (pos_y, pos_x)。剩余维度填充更高频率。

        Returns: (B, N, pos_enc_dim)
        '''
        B, N = pos_yx.shape[:2]
        pos_y, pos_x = pos_yx[:, :, 0], pos_yx[:, :, 1]
        p = torch.zeros(B, N, pos_enc_dim, device=pos_yx.device)

        # 基频率 (ω₀=1)：atan2(p[0], p[1]) = pos_y, atan2(p[2], p[3]) = pos_x
        p[:, :, 0] = torch.sin(pos_y)
        p[:, :, 1] = torch.cos(pos_y)
        p[:, :, 2] = torch.sin(pos_x)
        p[:, :, 3] = torch.cos(pos_x)

        # 更高频率（如空间允许）
        n_freq = (pos_enc_dim - 4) // 4
        if n_freq > 0:
            freq = 1.0 / (10000.0 ** (torch.arange(1, n_freq + 1, device=pos_yx.device).float() / 8))
            for i in range(n_freq):
                o = 4 + i * 4
                p[:, :, o + 0] = torch.sin(pos_y * freq[i])
                p[:, :, o + 1] = torch.cos(pos_y * freq[i])
                p[:, :, o + 2] = torch.sin(pos_x * freq[i])
                p[:, :, o + 3] = torch.cos(pos_x * freq[i])
        return p

    @staticmethod
    def _decode_pe_from_zd(p_zd):
        '''从 Z^d 中的 p 向量重建完整 32 维解码器 PE。
        p 每 4 维为一组 [sin_y, cos_y, sin_x, cos_x] 对应一个频率。
        使用 safe_atan2 避免 p_pred=0 时 atan2(0,0) 梯度 NaN。
        '''
        B, N, pos_enc_dim = p_zd.shape
        D_pos = 32
        half = D_pos // 4
        device = p_zd.device
        n_groups = pos_enc_dim // 4
        eps = 1e-6

        near_zero_y = (p_zd[:, :, 0].abs() < eps) & (p_zd[:, :, 1].abs() < eps)
        near_zero_x = (p_zd[:, :, 2].abs() < eps) & (p_zd[:, :, 3].abs() < eps)
        safe_siny = torch.where(near_zero_y, torch.zeros_like(p_zd[:, :, 0]).detach(), p_zd[:, :, 0])
        safe_cosy = torch.where(near_zero_y, torch.ones_like(p_zd[:, :, 1]).detach(), p_zd[:, :, 1])
        safe_sinx = torch.where(near_zero_x, torch.zeros_like(p_zd[:, :, 2]).detach(), p_zd[:, :, 2])
        safe_cosx = torch.where(near_zero_x, torch.ones_like(p_zd[:, :, 3]).detach(), p_zd[:, :, 3])
        pos_y = torch.atan2(safe_siny, safe_cosy)
        pos_x = torch.atan2(safe_sinx, safe_cosx)

        freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
        pe_parts = []
        for fi in range(half):
            if fi < n_groups:
                o = fi * 4
                pe_parts.extend([p_zd[:, :, o], p_zd[:, :, o + 1],
                                 p_zd[:, :, o + 2], p_zd[:, :, o + 3]])
            else:
                pe_parts.extend([
                    torch.sin(pos_y * freq[fi]), torch.cos(pos_y * freq[fi]),
                    torch.sin(pos_x * freq[fi]), torch.cos(pos_x * freq[fi]),
                ])
        return torch.stack(pe_parts, dim=-1).reshape(B, N, D_pos)

    def _add_sd_pos_encoding(self, slots, attn, grid_sz):
        centroid = self._compute_slot_centroid(attn, grid_sz)
        pe = self._reconstruct_pe(centroid)
        D_pos = pe.shape[-1]
        out = slots.clone()
        out[:, :, -D_pos:] = out[:, :, -D_pos:] + pe
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
        dyn_core_dim = getattr(self.config, 'dynamic_dim', 128)
        pos_enc_dim = getattr(self.config, 'pos_enc_dim', 8)
        # Z 空间总维数: [Z^c | Z₀ | p], p = 位置编码 (pos_enc_dim 维)
        slot_dim = static_dim + dyn_core_dim + pos_enc_dim

        with torch.no_grad():
            enc_features = self.encoder(frames)
        B_e, T_e, N_feat, D_e = enc_features.shape
        grid_sz = int(N_feat ** 0.5)

        # Phase 2: Burnin
        #   每帧: S_raw → f_z → Z_core = [Z^c | Z₀]
        #   从 attn 提取 centroid = (pos_y, pos_x) → 编码为 p (pos_enc_dim 维)
        #   Z = [Z_core | p]
        #   S_pe = S_raw + PE_32 供解码器重建
        buf_sz = getattr(self.config, 'buffer_len', burnin + rollout)
        Z_buffer = torch.zeros(B, buf_sz, self.config.num_slots, slot_dim, device=frames.device)
        burnin_S = []
        burnin_Z = []
        slots = None
        for t in range(burnin):
            feat_t = enc_features[:, t]
            with torch.no_grad():
                slots, attn = self._sa(feat_t, slots, t)
                centroid = self._compute_slot_centroid(attn, grid_sz)
                pe_32 = self._reconstruct_pe(centroid)
                slots_pe = slots.clone()
                slots_pe[:, :, -self.POS_EMBED_DIM:] = slots_pe[:, :, -self.POS_EMBED_DIM:] + pe_32
            Z_core = self.f_z(slots)                    # [Z^c | Z₀]
            p = self._encode_pos_to_zd(centroid, pos_enc_dim)
            Z_t = torch.cat([Z_core, p], dim=-1)         # [Z^c | Z₀ | p]
            burnin_S.append(slots_pe)
            burnin_Z.append(Z_t)
            Z_buffer[:, t] = Z_t
        burnin_S = torch.stack(burnin_S, dim=1)
        burnin_Z = torch.stack(burnin_Z, dim=1)

        # Phase 3: Rollout 在 Z 空间 (Z^d = [Z₀ | p])
        freeze_C = getattr(self.config, 'freeze_C', False)
        global_C = self.predictor.compute_C(burnin_Z) if freeze_C else None
        pred_Z_list = []
        energy_pairs = []
        cur_Z = Z_t
        qp_metrics = {"q_next_list": [], "p_next_list": [],
                      "fresh_q_list": [], "fresh_p_list": []}
        for t in range(rollout):
            C_use = global_C if freeze_C else cur_Z[:, :, :static_dim]
            out = self.predictor(cur_Z, Z_buffer[:, :burnin + t], C=C_use,
                                 return_energy=True, return_qp=True)
            next_Z, ep, (fresh_q, fresh_p), (q_next, p_next) = out
            pred_Z_list.append(next_Z)
            if ep is not None:
                energy_pairs.append(ep)
            if burnin + t < buf_sz:
                Z_buffer[:, burnin + t] = next_Z
            cur_Z = next_Z
            qp_metrics["fresh_q_list"].append(fresh_q)
            qp_metrics["fresh_p_list"].append(fresh_p)
            qp_metrics["q_next_list"].append(q_next)
            qp_metrics["p_next_list"].append(p_next)
        pred_Z = torch.stack(pred_Z_list, dim=1)

        loss_q = torch.tensor(0.0, device=frames.device)
        loss_p = torch.tensor(0.0, device=frames.device)
        count = 0
        for t in range(1, rollout):
            loss_q = loss_q + (qp_metrics["fresh_q_list"][t].detach() -
                               qp_metrics["q_next_list"][t - 1].detach()).square().mean()
            loss_p = loss_p + (qp_metrics["fresh_p_list"][t].detach() -
                               qp_metrics["p_next_list"][t - 1].detach()).square().mean()
            count = count + 1
        if count > 0:
            loss_q = loss_q / count
            loss_p = loss_p / count

        # Phase 4: Target S（frozen → S，no_grad）
        with torch.no_grad():
            target_S_list = []
            s = slots
            for t in range(burnin, burnin + rollout):
                feat_t = enc_features[:, t]
                s, attn = self._sa(feat_t, s, t)
                s = self._add_sd_pos_encoding(s, attn, grid_sz)
                target_S_list.append(s)
            target_S = torch.stack(target_S_list, dim=1)

        # Phase 5: 解码 — 从 pred_Z 取出 Z_core = [C | Z₀] → f_z⁻¹ → S_raw
        #           取出 p → _decode_pe_from_zd → PE_32 → S = S_raw + PE_32
        dec_burnin = torch.stack([self.decoder(burnin_S[:, t]) for t in range(burnin)], dim=1)
        pred_S_list = []
        for t in range(rollout):
            Z_core = pred_Z[:, t, :, :static_dim + dyn_core_dim]
            p_pred = pred_Z[:, t, :, static_dim + dyn_core_dim:]
            S_raw = self.f_z.inverse(Z_core)
            P_recon = self._decode_pe_from_zd(p_pred)
            S = S_raw.clone()
            S[:, :, -self.POS_EMBED_DIM:] = S[:, :, -self.POS_EMBED_DIM:] + P_recon
            pred_S_list.append(S)
        pred_S = torch.stack(pred_S_list, dim=1)
        dec_pred = torch.stack([self.decoder(pred_S[:, t]) for t in range(rollout)], dim=1)
        dec_target = torch.stack([self.decoder(target_S[:, t]) for t in range(rollout)], dim=1)

        # GRL（仅对 Z₀ 做梯度反转，不含 p）
        all_Z = torch.cat([burnin_Z, pred_Z], dim=1)
        Z_dyn = all_Z[:, :, :, static_dim:static_dim + dyn_core_dim]
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
            "qp_metrics": {
                "loss_q": loss_q.item(),
                "loss_p": loss_p.item(),
            },
        }

    def _forward_jepa(self, frames):
        B, T, C, H, W = frames.shape
        burnin = self.config.burnin_frames
        rollout = self.config.rollout_frames
        static_dim = getattr(self.config, 'static_dim', 128)
        dyn_core_dim = getattr(self.config, 'dynamic_dim', 128)
        pos_enc_dim = getattr(self.config, 'pos_enc_dim', 8)
        slot_dim = static_dim + dyn_core_dim + pos_enc_dim
        half_idx = static_dim + dyn_core_dim  # S 部分的维数（不含 p）

        enc_features = self.encoder(frames)
        B_e, T_e, N_feat, D_e = enc_features.shape
        grid_sz = int(N_feat ** 0.5)

        burnin_S = []
        buf_sz = getattr(self.config, 'buffer_len', burnin + rollout)
        slot_buffer = torch.zeros(B, buf_sz, self.config.num_slots, slot_dim, device=frames.device)
        slots = None
        for t in range(burnin):
            feat_t = enc_features[:, t]
            slots, attn = self._sa(feat_t, slots, t)
            centroid = self._compute_slot_centroid(attn, grid_sz)
            pe_32 = self._reconstruct_pe(centroid)
            slots_pe = slots.clone()
            slots_pe[:, :, -self.POS_EMBED_DIM:] = slots_pe[:, :, -self.POS_EMBED_DIM:] + pe_32
            burnin_S.append(slots_pe)
            p = self._encode_pos_to_zd(centroid, pos_enc_dim)
            slot_buffer[:, t] = torch.cat([slots_pe, p], dim=-1)
        burnin_S = torch.stack(burnin_S, dim=1)

        freeze_C = getattr(self.config, 'freeze_C', False)
        global_C = self.predictor.compute_C(
            slot_buffer[:, :burnin, :, :static_dim]) if freeze_C else None
        pred_S_list = []
        energy_pairs = []
        cur_S = slot_buffer[:, burnin - 1]
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
        pred_S_full = torch.stack(pred_S_list, dim=1)  # [S_pe | p]

        # Target (EMA)
        with torch.no_grad():
            target_enc = self.target_encoder(frames)
            target_slots = []
            s_target = None
            for t in range(burnin + rollout):
                feat_t = target_enc[:, t]
                s_target, attn_t = self._sa(feat_t, s_target, t, attn_module=self.target_slot_attention)
                s_target = self._add_sd_pos_encoding(s_target, attn_t, grid_sz)
                target_slots.append(s_target)
            target_all_s = torch.stack(target_slots, dim=1)

        # 解码：pred_S_full = [S_pe | p]，取前 half_idx 维送入 decoder
        dec_burnin = torch.stack([self.decoder(burnin_S[:, t]) for t in range(burnin)], dim=1)
        dec_pred = torch.stack(
            [self.decoder(pred_S_full[:, t, :, :half_idx]) for t in range(rollout)], dim=1)
        dec_target = torch.stack(
            [self.decoder(target_all_s[:, burnin + t]) for t in range(rollout)], dim=1)

        # GRL
        all_S = torch.cat([burnin_S, pred_S_full[:, :, :, :half_idx]], dim=1)
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
                "predicted": pred_S_full[:, :, :, :half_idx],
                "target": target_all_s[:, burnin:],
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
