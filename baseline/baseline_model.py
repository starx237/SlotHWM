import torch
import torch.nn as nn
from models.encoder import CNNEncoder, ResNetEncoder
from models.decoder import SpatialBroadcastDecoder
from models.attention import SlotAttention
from .predictor import Predictor


class BaselineModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pretrain = getattr(config, 'pretrain', False)
        burnin = getattr(config, 'burnin_frames', 6)
        rollout = getattr(config, 'rollout_frames', 10)

        enc_hidden = getattr(config, 'encoder_hidden', 32)
        if isinstance(enc_hidden, (list, tuple)):
            enc_hidden = enc_hidden[0]
        img_sz = getattr(config, 'img_size', 64)
        if isinstance(img_sz, (list, tuple)):
            img_sz = img_sz[0]
        in_channels = getattr(config, 'in_channels', 3)
        slot_dim = getattr(config, 'static_dim', 128) + getattr(config, 'dynamic_dim', 128)
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

        config.slot_dim = slot_dim
        self.predictor = Predictor(config)

        if hasattr(torch, 'compile'):
            self.decoder = torch.compile(self.decoder)
            self.encoder = torch.compile(self.encoder)

    def _add_sd_pos_encoding(self, slots, attn, grid_sz):
        B, N, _ = slots.shape
        D_sta = self.config.static_dim
        D_dyn = self.config.dynamic_dim
        attn_2d = attn.view(B, N, grid_sz, grid_sz)
        gy = torch.linspace(-1, 1, grid_sz, device=slots.device)
        gx = torch.linspace(-1, 1, grid_sz, device=slots.device)
        gy, gx = torch.meshgrid(gy, gx, indexing='ij')
        pos_y = (attn_2d * gy[None, None]).sum(dim=[2, 3])
        pos_x = (attn_2d * gx[None, None]).sum(dim=[2, 3])
        half = D_dyn // 4
        freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=slots.device).float() / half))
        pe = torch.stack([
            torch.sin(pos_y.unsqueeze(-1) * freq), torch.cos(pos_y.unsqueeze(-1) * freq),
            torch.sin(pos_x.unsqueeze(-1) * freq), torch.cos(pos_x.unsqueeze(-1) * freq),
        ], dim=-1).reshape(B, N, D_dyn)
        out = slots.clone()
        out[:, :, D_sta:] = out[:, :, D_sta:] + pe
        return out

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        burnin = self.config.burnin_frames
        rollout = self.config.rollout_frames
        slot_dim = getattr(self.config, 'static_dim', 128) + getattr(self.config, 'dynamic_dim', 128)

        if self.pretrain:
            return self._forward_pretrain(frames)

        enc_features = self.encoder(frames)
        grid_sz = int(enc_features.shape[2] ** 0.5)

        buf_sz = getattr(self.config, 'buffer_len', burnin + rollout)
        buffer = torch.zeros(B, buf_sz, self.config.num_slots, slot_dim, device=frames.device)

        burnin_slots = []
        slots = None
        for t in range(burnin):
            feat_t = enc_features[:, t]
            slots, attn = self.slot_attention(feat_t, slots)
            slots = self._add_sd_pos_encoding(slots, attn, grid_sz)
            burnin_slots.append(slots)
            buffer[:, t] = slots
        burnin_slots = torch.stack(burnin_slots, dim=1)

        pred_slots = []
        cur = slots
        for t in range(rollout):
            nxt = self.predictor(cur, buffer[:, :burnin + t])
            pred_slots.append(nxt)
            if burnin + t < buf_sz:
                buffer[:, burnin + t] = nxt
            cur = nxt
        pred_slots = torch.stack(pred_slots, dim=1)

        with torch.no_grad():
            target_slots = []
            s = slots
            for t in range(burnin, burnin + rollout):
                feat_t = enc_features[:, t]
                s, attn = self.slot_attention(feat_t, s)
                s = self._add_sd_pos_encoding(s, attn, grid_sz)
                target_slots.append(s)
            target_slots = torch.stack(target_slots, dim=1)

        dec_burnin = torch.stack([self.decoder(burnin_slots[:, t]) for t in range(burnin)], dim=1)
        dec_pred = torch.stack([self.decoder(pred_slots[:, t]) for t in range(rollout)], dim=1)
        dec_target = torch.stack([self.decoder(target_slots[:, t]) for t in range(rollout)], dim=1)

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
        }

    def _forward_pretrain(self, frames):
        B, T, C, H, W = frames.shape
        burnin = self.config.burnin_frames

        enc_features = self.encoder(frames)
        grid_sz = int(enc_features.shape[2] ** 0.5)

        burnin_slots = []
        slots = None
        for t in range(burnin):
            feat_t = enc_features[:, t]
            slots, attn = self.slot_attention(feat_t, slots)
            slots = self._add_sd_pos_encoding(slots, attn, grid_sz)
            burnin_slots.append(slots)
        burnin_slots = torch.stack(burnin_slots, dim=1)

        dec_burnin = torch.stack([self.decoder(burnin_slots[:, t]) for t in range(burnin)], dim=1)

        return {
            "outputs": {
                "video_burnin": dec_burnin,
                "video_pred": None,
            },
            "slots": {
                "corrected": burnin_slots,
                "predicted": None,
            },
        }
