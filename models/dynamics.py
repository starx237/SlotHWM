import torch
import torch.nn as nn
from models.encoder import CNNEncoder, ResNetEncoder
from models.decoder import ISASpatialBroadcastDecoder
from models.attention import SlotAttentionTranslScaleEquiv
from models.predictor import SlotPredictor
from models.misc import GradientReversal, AffineCoupling, Identity, create_coordinate_grid


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
        z_dyn_dim = self.appearance_dim - self.static_dim
        if z_dyn_dim > 0:
            self.f_z = AffineCoupling(
                static_dim=self.static_dim,
                dynamic_dim=z_dyn_dim,
                hidden_dim=getattr(config, 'hidden_dim', 256),
            )
        else:
            self.f_z = Identity()

        self.predictor = SlotPredictor(config)

        # GRL + 反向预测器
        self.grl = GradientReversal()
        self.mlp_rev = nn.Linear(self.dynamic_dim, self.static_dim)

        # GRU2: 帧间 slot 残差预测（从上一帧slot预测下一帧slot初值）
        self.gru2_hidden_dim = getattr(config, 'gru2_hidden_dim', 64)
        self.gru2 = nn.GRUCell(self.appearance_dim, self.gru2_hidden_dim)
        self.gru2_proj = nn.Linear(self.gru2_hidden_dim, self.appearance_dim)
        nn.init.zeros_(self.gru2_proj.weight)
        nn.init.zeros_(self.gru2_proj.bias)
        self.gru2_predict_full = getattr(config, 'gru2_predict_full', False)
        if self.gru2_predict_full:
            self.gru2_proj_posdepth = nn.Linear(self.gru2_hidden_dim, 3)
            nn.init.zeros_(self.gru2_proj_posdepth.weight)
            nn.init.zeros_(self.gru2_proj_posdepth.bias)

        # Depth→Spread Predictor: 迫使 ISA 把大小信息编码到 depth
        self.depth_spread_weight = getattr(config, 'depth_spread_weight', 0.0)
        if self.depth_spread_weight > 0:
            use_prior = getattr(config, 'depth_spread_prior', False)
            if use_prior:
                self.depth_spread_predictor = None
                self.depth_spread_a = nn.Parameter(torch.tensor(1.0))
                self.depth_spread_b = nn.Parameter(torch.tensor(1.0))
                self.depth_spread_c = nn.Parameter(torch.tensor(0.0))
                self.depth_spread_d = nn.Parameter(torch.tensor(0.0))
            else:
                hidden = getattr(config, 'depth_spread_hidden', 32)
                self.depth_spread_predictor = nn.Sequential(
                    nn.Linear(1, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                    nn.Linear(hidden, 2),
                )
                nn.init.xavier_uniform_(self.depth_spread_predictor[-1].weight, gain=0.1)
                nn.init.zeros_(self.depth_spread_predictor[-1].bias)

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

        needs_compile = (not getattr(config, 'continue_pretrain', False)
                         and not getattr(config, 'freeze_slot', False)
                         and not getattr(config, 'jepa', False))
        if hasattr(torch, 'compile') and needs_compile:
            self.decoder = torch.compile(self.decoder)
            self.encoder = torch.compile(self.encoder)

        if getattr(config, 'freeze_slot', False):
            for name in ['encoder', 'slot_attention', 'decoder', 'gru2', 'gru2_proj', 'gru2_proj_posdepth', 'depth_spread_predictor']:
                mod = getattr(self, name, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(False)
            if getattr(config, 'freeze_appearance', False):
                for p in self.predictor.C_time_attn.parameters():
                    p.requires_grad_(False)
            print("Frozen: encoder + slot_attention + decoder + gru2 + gru2_proj (ISA)" + 
                  (" + C_time_attn" if getattr(config, 'freeze_appearance', False) else ""))

        if getattr(config, 'continue_pretrain', False):
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            for p in self.decoder.parameters():
                p.requires_grad_(False)
            for p in self.slot_attention.parameters():
                p.requires_grad_(False)
            print("continue_pretrain: encoder+decoder+slot_attention frozen, GRU2 trainable")

    def _encode_features(self, frames):
        B, T, C, H, W = frames.shape
        feat = self.encoder(frames)
        B, T, N, D = feat.shape
        grid_sz = int(N ** 0.5)
        grid = create_coordinate_grid(grid_sz, grid_sz, frames.device)
        grid = grid.view(1, 1, N, 2).expand(B, T, N, 2)
        feat_with_grid = torch.cat([feat, grid], dim=-1)
        return feat_with_grid

    def _sa(self, feat_t, slots, t, attn_module=None, iters_first=None, iters_rest=None):
        mod = attn_module or self.slot_attention
        n_first = iters_first if iters_first is not None else getattr(self.config, 'iters_first', 3)
        n_rest = iters_rest if iters_rest is not None else getattr(self.config, 'iters_rest', 1)
        n_iters = n_first if t == 0 else n_rest
        return mod(feat_t, slots, num_iterations=n_iters)

    def _gru2_step(self, prev_appearance, gru2_hidden, prev_posdepth=None):
        B, N, D = prev_appearance.shape
        gru2_hidden = self.gru2(
            prev_appearance.reshape(-1, self.appearance_dim),
            gru2_hidden.reshape(-1, self.gru2_hidden_dim),
        )
        residual = self.gru2_proj(gru2_hidden).reshape(B, N, self.appearance_dim)
        new_appearance = prev_appearance + residual
        if self.gru2_predict_full and prev_posdepth is not None:
            residual_pd = self.gru2_proj_posdepth(gru2_hidden).reshape(B, N, 3)
            new_posdepth = prev_posdepth + residual_pd
            return new_appearance, new_posdepth, gru2_hidden
        return new_appearance, gru2_hidden

    def _forward_pretrain(self, frames):
        B, T, C, H, W = frames.shape
        burnin = self.config.burnin_frames

        feat = self._encode_features(frames)

        burnin_slots = []
        attn_list = []
        slots = None
        gru2_hidden = None
        prev_appearance = None
        prev_posdepth = None
        gru2_pred_posdepth_list = []
        use_gru2 = getattr(self.config, 'continue_pretrain', False) or self.gru2_predict_full
        for t in range(burnin):
            if t > 0 and use_gru2 and slots is not None:
                if self.gru2_predict_full:
                    new_appearance, new_posdepth, gru2_hidden = self._gru2_step(prev_appearance, gru2_hidden, prev_posdepth)
                    gru2_pred_posdepth_list.append(new_posdepth)
                    slots = torch.cat([new_appearance, new_posdepth], dim=-1)
                else:
                    new_appearance, gru2_hidden = self._gru2_step(prev_appearance, gru2_hidden)
                    slots = torch.cat([
                        new_appearance,
                        slots[:, :, -3:-1].contiguous(),
                        slots[:, :, -1:].contiguous(),
                    ], dim=-1)
            slots, attn = self._sa(feat[:, t], slots, t)
            if use_gru2:
                prev_appearance = slots[:, :, :-3].detach()
                if self.gru2_predict_full:
                    prev_posdepth = slots[:, :, -3:].detach()
                if t == 0:
                    BN = prev_appearance.shape[0] * prev_appearance.shape[1]
                    gru2_hidden = torch.zeros(BN, self.gru2_hidden_dim, device=frames.device)
                    gru2_hidden = self.gru2(
                        prev_appearance.reshape(-1, self.appearance_dim),
                        gru2_hidden,
                    )
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
            "gru2_pred_posdepth": gru2_pred_posdepth_list if gru2_pred_posdepth_list else None,
        }

    def _forward_finetune(self, frames):
        B, T, C, H, W = frames.shape
        burnin = self.config.burnin_frames
        rollout = self.config.rollout_frames

        with torch.no_grad():
            feat = self._encode_features(frames)

        buf_sz = getattr(self.config, 'buffer_len', burnin + rollout)
        slot_dim_z = self.static_dim + self.dynamic_dim

        burnin_S = []
        burnin_Z = []
        burnin_Z_buffer = []
        slots = None
        gru2_hidden = None
        prev_appearance = None
        prev_posdepth = None
        gru2_pred_posdepth_list = []
        for t in range(burnin):
            feat_t = feat[:, t]
            # Apply GRU2 for cross-frame slot propagation
            if t > 0 and slots is not None:
                if self.gru2_predict_full:
                    new_appearance, new_posdepth, gru2_hidden = self._gru2_step(prev_appearance, gru2_hidden, prev_posdepth)
                    gru2_pred_posdepth_list.append(new_posdepth)
                    slots = torch.cat([new_appearance, new_posdepth], dim=-1)
                else:
                    new_appearance, gru2_hidden = self._gru2_step(prev_appearance, gru2_hidden)
                    slots = torch.cat([
                        new_appearance,
                        slots[:, :, -3:-1].contiguous(),
                        slots[:, :, -1:].contiguous(),
                    ], dim=-1)
            with torch.no_grad():
                slots, attn = self._sa(feat_t, slots, t)
            # Store appearance for next GRU2 step
            prev_appearance = slots[:, :, :-3].detach()
            if self.gru2_predict_full:
                prev_posdepth = slots[:, :, -3:].detach()
            if t == 0:
                BN = prev_appearance.shape[0] * prev_appearance.shape[1]
                gru2_hidden = torch.zeros(BN, self.gru2_hidden_dim, device=frames.device)
                gru2_hidden = self.gru2(
                    prev_appearance.reshape(-1, self.appearance_dim),
                    gru2_hidden.reshape(-1, self.gru2_hidden_dim),
                )
            burnin_S.append(slots)
            Z_core = self.f_z(slots[:, :, :self.appearance_dim])
            Z_full = torch.cat([Z_core, slots[:, :, -3:]], dim=-1)
            burnin_Z.append(Z_full)
            burnin_Z_buffer.append(Z_full)
        burnin_S = torch.stack(burnin_S, dim=1)
        burnin_Z = torch.stack(burnin_Z, dim=1)

        freeze_C = getattr(self.config, 'freeze_C', False)
        global_C = self.predictor.compute_C(burnin_Z) if freeze_C else None
        pred_Z_list = []
        energy_pairs = []
        cur_Z = Z_full
        Z_buffer = list(burnin_Z_buffer)
        qp_metrics = {"q_next_list": [], "p_next_list": [],
                      "fresh_q_list": [], "fresh_p_list": []}
        for t in range(rollout):
            C_use = global_C if freeze_C else cur_Z[:, :, :self.static_dim]
            Z_buf_t = torch.stack(Z_buffer[:burnin + t], dim=1)
            out = self.predictor(cur_Z, Z_buf_t, C=C_use,
                                 return_energy=True, return_qp=True)
            next_Z, ep, (fresh_q, fresh_p), (q_next, p_next) = out
            pred_Z_list.append(next_Z)
            if ep is not None:
                energy_pairs.append(ep)
            Z_buffer.append(next_Z)
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
                if gru2_hidden is not None:
                    if self.gru2_predict_full and prev_posdepth is not None:
                        new_app, new_pd, gru2_hidden = self._gru2_step(prev_appearance, gru2_hidden, prev_posdepth)
                        s = torch.cat([new_app, new_pd], dim=-1)
                    else:
                        new_app, gru2_hidden = self._gru2_step(prev_appearance, gru2_hidden)
                        s = torch.cat([new_app, s[:, :, -3:-1].contiguous(), s[:, :, -1:].contiguous()], dim=-1)
                s, attn_t = self._sa(feat_t, s, t)
                prev_appearance = s[:, :, :-3].detach()
                if self.gru2_predict_full:
                    prev_posdepth = s[:, :, -3:].detach()
                target_S_list.append(s)
            target_S = torch.stack(target_S_list, dim=1)

        depth_max = getattr(self.config, 'depth_max', 0.30)
        delta_depth_max = getattr(self.config, 'delta_depth_max', 0.05)
        bnd_threshold = getattr(self.config, 'bnd_threshold', 0.75)
        with torch.no_grad():
            target_depth = target_S[:, :, :, -1]
            burnin_last_depth = burnin_S[:, -1, :, -1]
            is_bg_burnin = burnin_last_depth >= depth_max
            depth_mask = torch.ones(B, rollout, target_S.shape[2],
                                    dtype=torch.bool, device=frames.device)
            ever_touched_bnd = torch.zeros(B, target_S.shape[2],
                                           dtype=torch.bool, device=frames.device)
            for t in range(rollout):
                cur_d = target_depth[:, t, :]
                is_bg_now = cur_d >= depth_max
                fg_to_bg = ~is_bg_burnin & is_bg_now
                bg_to_fg = is_bg_burnin & ~is_bg_now
                if t == 0:
                    delta_d = (cur_d - burnin_last_depth).abs()
                else:
                    delta_d = (cur_d - target_depth[:, t - 1, :]).abs()
                big_delta = delta_d >= delta_depth_max
                cur_pos_x = target_S[:, t, :, -3].abs()
                cur_pos_y = target_S[:, t, :, -2].abs()
                near_bnd = (cur_pos_x > bnd_threshold) | (cur_pos_y > bnd_threshold)
                ever_touched_bnd = ever_touched_bnd | (near_bnd & ~is_bg_now)
                depth_mask[:, t, :] = ~(fg_to_bg | bg_to_fg | big_delta | ever_touched_bnd)

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
            "depth_mask": depth_mask,
            "rev_pred": rev_pred,
            "S_c": Z_c,
            "energy_pairs": energy_pairs if energy_pairs else None,
            "qp_metrics": {
                "loss_q": loss_q.item(),
                "loss_p": loss_p.item(),
            },
            "gru2_pred_posdepth": gru2_pred_posdepth_list if gru2_pred_posdepth_list else None,
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

