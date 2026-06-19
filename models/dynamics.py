import torch
import torch.nn as nn
from models.encoder import CNNEncoder, ResNetEncoder
from models.decoder import ISASpatialBroadcastDecoder
from models.attention import SlotAttentionTranslScaleEquiv
from models.predictor import SlotPredictor
from models.misc import GradientReversal, AffineCoupling, create_coordinate_grid


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

        enc_hidden = getattr(config, 'encoder_hidden', 64)
        if isinstance(enc_hidden, (list, tuple)):
            enc_hidden = enc_hidden[0]

        img_sz = getattr(config, 'img_size', 64)
        if isinstance(img_sz, (list, tuple)):
            img_sz = img_sz[0]

        in_channels = getattr(config, 'in_channels', 3)
        feat_dim = getattr(config, 'feat_dim', 64)
        self.appearance_dim = getattr(config, 'appearance_dim', 64)
        self.slot_dim = self.appearance_dim + 3
        self.static_dim = getattr(config, 'static_dim', 34)
        self.dynamic_dim = getattr(config, 'dynamic_dim', 33)
        qkv_size = getattr(config, 'sa_qkv_size', None) or getattr(config, 'qkv_size', self.slot_dim)
        num_slots = getattr(config, 'num_slots', 7)

        enc_type = getattr(config, 'encoder_type', 'cnn')
        if enc_type == 'cnn':
            self.encoder = CNNEncoder(
                in_channels=in_channels, hidden_channels=enc_hidden,
                out_dim=feat_dim, img_size=img_sz,
                pos_embedding=None, reduction='flatten',
            )
        else:
            self.encoder = ResNetEncoder(
                resnet_version=getattr(config, 'resnet_version', 18),
                pretrained=getattr(config, 'resnet_pretrained', False),
                out_dim=feat_dim, pos_embedding=None, reduction='none',
            )

        self.slot_attention = SlotAttentionTranslScaleEquiv(
            num_slots=num_slots,
            appearance_dim=self.appearance_dim,
            feat_dim=feat_dim,
            qkv_size=qkv_size,
            grid_enc_hidden=getattr(config, 'grid_enc_hidden', qkv_size * 2),
            mlp_size=getattr(config, 'sa_mlp_size', 256),
            num_iterations=getattr(config, 'slot_iters', 3),
            epsilon=1e-8,
            min_scale=0.001,
            max_scale=2.0,
            scales_factor=getattr(config, 'scales_factor', 5.0),
            init_with_fixed_scale=getattr(config, 'init_with_fixed_scale', None),
            add_rel_pos_to_values=getattr(config, 'add_rel_pos_to_values', True),
            softmax_temperature=getattr(config, 'softmax_temperature', 1.0),
            append_statistics=getattr(config, 'append_statistics', False),
        )

        dec_hidden = getattr(config, 'decoder_hidden', 64)
        if isinstance(dec_hidden, (list, tuple)):
            dec_hidden = dec_hidden[0]
        bs = getattr(config, 'broadcast_size', 16)
        if isinstance(bs, (list, tuple)):
            bs = bs[0]

        self.decoder = ISASpatialBroadcastDecoder(
            appearance_dim=self.appearance_dim,
            output_channels=in_channels,
            hidden_channels=dec_hidden,
            broadcast_size=bs,
            img_size=img_sz,
            num_slots=num_slots,
            predict_mask=True,
            scales_factor=getattr(config, 'scales_factor', 5.0),
        )

        # f_z 只作用在 appearance 部分
        self.f_z = AffineCoupling(
            static_dim=self.static_dim,
            dynamic_dim=self.appearance_dim - self.static_dim,
            hidden_dim=getattr(config, 'hidden_dim', 256),
        )

        self.predictor = SlotPredictor(config)

        # GRL + 反向预测器
        self.grl = GradientReversal()
        self.mlp_rev = nn.Linear(self.dynamic_dim, self.static_dim)

        # JEPA 组件
        if self.jepa:
            if enc_type == 'cnn':
                self.target_encoder = CNNEncoder(
                    in_channels=in_channels, hidden_channels=enc_hidden,
                    out_dim=feat_dim, img_size=img_sz,
                    pos_embedding=None, reduction='flatten',
                )
            else:
                self.target_encoder = ResNetEncoder(
                    resnet_version=getattr(config, 'resnet_version', 18),
                    pretrained=False,
                    out_dim=feat_dim, pos_embedding=None, reduction='none',
                )
            self.target_encoder.load_state_dict(self.encoder.state_dict())
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)

            self.target_slot_attention = SlotAttentionTranslScaleEquiv(
                num_slots=num_slots,
                appearance_dim=self.appearance_dim,
                feat_dim=feat_dim,
                qkv_size=qkv_size,
                grid_enc_hidden=getattr(config, 'grid_enc_hidden', qkv_size * 2),
                mlp_size=getattr(config, 'sa_mlp_size', 256),
                num_iterations=getattr(config, 'slot_iters', 3),
                scales_factor=getattr(config, 'scales_factor', 5.0),
                add_rel_pos_to_values=getattr(config, 'add_rel_pos_to_values', True),
                softmax_temperature=getattr(config, 'softmax_temperature', 1.0),
            )
            self.target_slot_attention.load_state_dict(self.slot_attention.state_dict())
            for p in self.target_slot_attention.parameters():
                p.requires_grad_(False)

        if hasattr(torch, 'compile'):
            self.decoder = torch.compile(self.decoder)
            self.encoder = torch.compile(self.encoder)
            if self.jepa:
                self.target_encoder = torch.compile(self.target_encoder)

        if getattr(config, 'freeze_slot', False):
            for name in ['encoder', 'slot_attention', 'decoder']:
                mod = getattr(self, name, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(False)
            print("Frozen: encoder + slot_attention + decoder (ISA)")

        if getattr(config, 'train_gru_only', False):
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            for p in self.decoder.parameters():
                p.requires_grad_(False)
            for name, p in self.slot_attention.named_parameters():
                if 'gru' not in name and 'mlp' not in name:
                    p.requires_grad_(False)
            print("train_gru_only: encoder+decoder frozen, SA only GRU+MLP trainable")

    def _encode_features(self, frames):
        B, T, C, H, W = frames.shape
        feat = self.encoder(frames)
        B, T, N, D = feat.shape
        grid_sz = int(N ** 0.5)
        grid = create_coordinate_grid(grid_sz, grid_sz, frames.device)
        grid = grid.view(1, 1, N, 2).expand(B, T, N, 2)
        feat_with_grid = torch.cat([feat, grid], dim=-1)
        return feat_with_grid

    def _sa(self, feat_t, slots, t, attn_module=None):
        mod = attn_module or self.slot_attention
        n_iters = self.slot_attention.num_iterations
        return mod(feat_t, slots, num_iterations=n_iters)

    def _forward_pretrain(self, frames):
        B, T, C, H, W = frames.shape
        burnin = self.config.burnin_frames

        feat = self._encode_features(frames)

        burnin_slots = []
        attn_list = []
        slots = None
        for t in range(burnin):
            slots, attn = self._sa(feat[:, t], slots, t)
            burnin_slots.append(slots)
            attn_list.append(attn)
        burnin_slots = torch.stack(burnin_slots, dim=1)
        attn_all = torch.stack(attn_list, dim=1)

        dec_results = [self.decoder(burnin_slots[:, t], return_rgb=True) for t in range(burnin)]
        dec_burnin = torch.stack([r[0] for r in dec_results], dim=1)
        dec_alpha = torch.stack([r[1] for r in dec_results], dim=2)

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
            "attn": attn_all,
            "alpha": dec_alpha,
            "rev_pred": None,
            "S_c": None,
            "energy_pairs": None,
        }

    def _forward_finetune(self, frames):
        B, T, C, H, W = frames.shape
        burnin = self.config.burnin_frames
        rollout = self.config.rollout_frames

        with torch.no_grad():
            feat = self._encode_features(frames)

        buf_sz = getattr(self.config, 'buffer_len', burnin + rollout)
        slot_dim_z = self.static_dim + self.dynamic_dim
        Z_buffer = torch.zeros(B, buf_sz, self.config.num_slots, slot_dim_z, device=frames.device)

        burnin_S = []
        burnin_Z = []
        slots = None
        for t in range(burnin):
            feat_t = feat[:, t]
            with torch.no_grad():
                slots, attn = self._sa(feat_t, slots, t)
            burnin_S.append(slots)
            Z_core = self.f_z(slots[:, :, :self.appearance_dim])
            Z_full = torch.cat([Z_core, slots[:, :, -3:]], dim=-1)
            burnin_Z.append(Z_full)
            Z_buffer[:, t] = Z_full
        burnin_S = torch.stack(burnin_S, dim=1)
        burnin_Z = torch.stack(burnin_Z, dim=1)

        freeze_C = getattr(self.config, 'freeze_C', False)
        global_C = self.predictor.compute_C(burnin_Z) if freeze_C else None
        pred_Z_list = []
        energy_pairs = []
        cur_Z = Z_full
        qp_metrics = {"q_next_list": [], "p_next_list": [],
                      "fresh_q_list": [], "fresh_p_list": []}
        for t in range(rollout):
            C_use = global_C if freeze_C else cur_Z[:, :, :self.static_dim]
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

        with torch.no_grad():
            target_S_list = []
            s = slots
            for t in range(burnin, burnin + rollout):
                feat_t = feat[:, t]
                s, attn_t = self._sa(feat_t, s, t)
                target_S_list.append(s)
            target_S = torch.stack(target_S_list, dim=1)

        dec_burnin = torch.stack([self.decoder(burnin_S[:, t]) for t in range(burnin)], dim=1)
        pred_S_list = []
        for t in range(rollout):
            Z_appearance = pred_Z[:, t, :, :self.appearance_dim]
            pos_depth = pred_Z[:, t, :, self.appearance_dim:]
            S_raw = self.f_z.inverse(Z_appearance)
            S = torch.cat([S_raw, pos_depth], dim=-1)
            pred_S_list.append(S)
        pred_S = torch.stack(pred_S_list, dim=1)
        dec_pred = torch.stack([self.decoder(pred_S[:, t]) for t in range(rollout)], dim=1)
        dec_target = torch.stack([self.decoder(target_S[:, t]) for t in range(rollout)], dim=1)

        all_Z = torch.cat([burnin_Z, pred_Z], dim=1)
        Z_dyn = all_Z[:, :, :, self.static_dim:]
        rev_pred = self.mlp_rev(self.grl(Z_dyn))
        Z_c = all_Z[:, :, :, :self.static_dim]

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
        slot_dim_z = self.static_dim + self.dynamic_dim

        feat = self._encode_features(frames)

        burnin_S = []
        buf_sz = getattr(self.config, 'buffer_len', burnin + rollout)
        slot_buffer = torch.zeros(B, buf_sz, self.config.num_slots, slot_dim_z, device=frames.device)
        slots = None
        for t in range(burnin):
            feat_t = feat[:, t]
            slots, attn = self._sa(feat_t, slots, t)
            burnin_S.append(slots)
            Z_core = self.f_z(slots[:, :, :self.appearance_dim])
            Z_full = torch.cat([Z_core, slots[:, :, -3:]], dim=-1)
            slot_buffer[:, t] = Z_full
        burnin_S = torch.stack(burnin_S, dim=1)

        freeze_C = getattr(self.config, 'freeze_C', False)
        global_C = self.predictor.compute_C(
            slot_buffer[:, :burnin, :, :self.static_dim]) if freeze_C else None
        pred_S_list = []
        energy_pairs = []
        cur_S = slot_buffer[:, burnin - 1]
        for t in range(rollout):
            C_use = global_C if freeze_C else cur_S[:, :, :self.static_dim]
            out = self.predictor(cur_S, slot_buffer[:, :burnin + t], C=C_use, return_energy=True)
            next_S, ep = out if isinstance(out, tuple) else (out, None)
            pred_S_list.append(next_S)
            if ep is not None:
                energy_pairs.append(ep)
            if burnin + t < buf_sz:
                slot_buffer[:, burnin + t] = next_S
            cur_S = next_S
        pred_S_full = torch.stack(pred_S_list, dim=1)

        with torch.no_grad():
            target_enc = self.target_encoder(frames)
            target_slots = []
            s_target = None
            for t in range(burnin + rollout):
                feat_t = target_enc[:, t]
                grid_sz_t = int(feat_t.shape[1] ** 0.5)
                grid_t = create_coordinate_grid(grid_sz_t, grid_sz_t, frames.device)
                grid_t = grid_t.view(1, 1, -1, 2).expand(B, 1, -1, 2)
                feat_with_grid_t = torch.cat([feat_t, grid_t.squeeze(1)], dim=-1)
                s_target, attn_t = self._sa(feat_with_grid_t, s_target, t,
                                            attn_module=self.target_slot_attention)
                target_slots.append(s_target)
            target_all_s = torch.stack(target_slots, dim=1)

        dec_burnin = torch.stack([self.decoder(burnin_S[:, t]) for t in range(burnin)], dim=1)
        pred_S_decoded = []
        for t in range(rollout):
            Z_appearance = pred_S_full[:, t, :, :self.appearance_dim]
            pos_depth = pred_S_full[:, t, :, self.appearance_dim:]
            S_raw = self.f_z.inverse(Z_appearance)
            S = torch.cat([S_raw, pos_depth], dim=-1)
            pred_S_decoded.append(S)
        pred_S_decoded = torch.stack(pred_S_decoded, dim=1)
        dec_pred = torch.stack(
            [self.decoder(pred_S_decoded[:, t]) for t in range(rollout)], dim=1)
        dec_target = torch.stack(
            [self.decoder(target_all_s[:, burnin + t]) for t in range(rollout)], dim=1)

        all_S = torch.cat([burnin_S, pred_S_decoded], dim=1)
        S_dyn = all_S[:, :, :, self.static_dim:]
        rev_pred = self.mlp_rev(self.grl(S_dyn))
        S_c = all_S[:, :, :, :self.static_dim]

        return {
            "outputs": {
                "video_burnin": dec_burnin,
                "video_pred": dec_pred,
                "video_target": dec_target,
            },
            "slots": {
                "corrected": burnin_S,
                "predicted": pred_S_decoded,
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

