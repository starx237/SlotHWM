import torch
import torch.nn as nn
from models.encoder import CNNEncoder, ResNetEncoder
from models.decoder import SpatialBroadcastDecoder
from models.attention import SlotAttention
from models.slotpi_model import SlotPiModel
from models.misc import GradientReversal


class SlotPi(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
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
            hidden_channels=dec_hidden, broadcast_size=bs,
            num_slots=num_slots, predict_mask=True, use_alpha=True,
        )

        self.slotpi = SlotPiModel(config)

        # GRL + 反向预测器（idea.md §4）
        self.grl = GradientReversal()
        # 单层 Linear 确保 MLP_rev 无法轻易拟合同源映射，让 GRL 能真正起作用
        self.mlp_rev = nn.Linear(dynamic_dim, static_dim)

    def forward(self, frames):
        '''端到端前向 (B, T, C, H, W) → 重建帧 + slots + C + rev_pred'''
        B, T, C, H, W = frames.shape
        burnin = self.config.burnin_frames
        rollout = self.config.rollout_frames
        num_slots = self.config.num_slots
        static_dim = getattr(self.config, 'static_dim', 128)
        dynamic_dim = getattr(self.config, 'dynamic_dim', 128)
        slot_dim = static_dim + dynamic_dim
        buffer_len = getattr(self.config, 'buffer_len', 10)

        # Phase 1: 编码所有帧 → 特征
        enc_features = self.encoder(frames)

        # Phase 2: Burnin — 用 GT 帧提取 corrected slots
        buffer = torch.zeros(B, buffer_len, num_slots, slot_dim, device=frames.device)
        burnin_slots = []
        slots = None
        for t in range(burnin):
            feat_t = enc_features[:, t]
            slots, attn = self.slot_attention(feat_t, slots)
            burnin_slots.append(slots)
            buffer = torch.cat([buffer[:, 1:], slots.unsqueeze(1)], dim=1)

        # 计算静态上下文 C
        burnin_slots = torch.stack(burnin_slots, dim=1)
        C = self.slotpi.compute_C(burnin_slots)

        # Phase 3: Rollout — SlotPi 自回归预测未来 slots（同时计算能量对用于 L_energy）
        pred_slots_list = []
        energy_pairs = []
        cur_slots = slots
        for t in range(rollout):
            out = self.slotpi(cur_slots, buffer, C=C, return_energy=True)
            next_slots, ep = out if isinstance(out, tuple) else (out, None)
            pred_slots_list.append(next_slots)
            if ep is not None:
                energy_pairs.append(ep)
            buffer = torch.cat([buffer[:, 1:], next_slots.unsqueeze(1)], dim=1)
            cur_slots = next_slots
        pred_slots = torch.stack(pred_slots_list, dim=1)

        # Phase 4: 从 GT rollout 帧提取 target slots（无梯度，仅用于 slot_pred_loss）
        with torch.no_grad():
            target_slots_list = []
            s = burnin_slots[:, -1]
            for t in range(burnin, burnin + rollout):
                feat_t = enc_features[:, t]
                s, _ = self.slot_attention(feat_t, s)
                target_slots_list.append(s)
            target_slots = torch.stack(target_slots_list, dim=1)

        # Phase 5: 解码器重建帧
        dec_burnin = torch.stack([self.decoder(burnin_slots[:, t]) for t in range(burnin)], dim=1)
        dec_pred = torch.stack([self.decoder(s) for s in pred_slots_list], dim=1)
        with torch.no_grad():
            dec_target = torch.stack([self.decoder(s) for s in target_slots_list], dim=1)

        # GRL 反向预测（idea.md §4）：用 S^d 预测 S^c，梯度逆转迫使解耦
        all_slots = torch.cat([burnin_slots, pred_slots], dim=1)
        slots_dyn = all_slots[:, :, :, static_dim:]
        rev_pred = self.mlp_rev(self.grl(slots_dyn))
        S_c = all_slots[:, :, :, :static_dim]  # 静态特征，作为 rev 预测目标

        return {
            "outputs": {
                "video_burnin": dec_burnin,
                "video_pred": dec_pred,
                "video_target": dec_target,
            },
            "slots": {
                "corrected": burnin_slots,
                "predicted": pred_slots,
                "target": target_slots,
            },
            "C": C,
            "rev_pred": rev_pred,
            "S_c": S_c,
            "energy_pairs": energy_pairs if energy_pairs else None,
        }
