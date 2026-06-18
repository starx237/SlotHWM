import torch
import torch.nn as nn
from models.encoder import CNNEncoder
from isa_pretrain.models.isa_slot_attention import SlotAttentionTranslScaleEquiv
from isa_pretrain.models.isa_decoder import ISASpatialBroadcastDecoder
from isa_pretrain.models.isa_misc import create_coordinate_grid


class SlotDynamicsModelISA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.burnin = getattr(config, 'burnin_frames', 6)

        hidden_dim = getattr(config, 'encoder_hidden', 64)
        if isinstance(hidden_dim, (list, tuple)):
            hidden_dim = hidden_dim[0]

        img_sz = getattr(config, 'img_size', 64)
        if isinstance(img_sz, (list, tuple)):
            img_sz = img_sz[0]

        in_channels = getattr(config, 'in_channels', 3)
        feat_dim = getattr(config, 'feat_dim', hidden_dim)
        self.appearance_dim = getattr(config, 'appearance_dim', 124)
        self.slot_dim = self.appearance_dim + 4
        qkv_size = getattr(config, 'qkv_size', self.slot_dim)
        num_slots = getattr(config, 'num_slots', 7)

        self.encoder = CNNEncoder(
            in_channels=in_channels, hidden_channels=hidden_dim,
            out_dim=feat_dim, img_size=img_sz,
            pos_embedding=None, reduction='flatten',
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
        bs = getattr(config, 'broadcast_size', 8)
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

        if hasattr(torch, 'compile'):
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)

    def _encode_features(self, frames):
        B, T, C, H, W = frames.shape
        feat = self.encoder(frames)
        B, T, N, D = feat.shape
        grid_sz = int(N ** 0.5)
        grid = create_coordinate_grid(grid_sz, grid_sz, frames.device)
        grid = grid.view(1, 1, N, 2).expand(B, T, N, 2)
        feat_with_grid = torch.cat([feat, grid], dim=-1)
        return feat_with_grid

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        feat = self._encode_features(frames)

        burnin_iters = getattr(self.config, 'burnin_iters', self.slot_attention.num_iterations)
        rollout_iters = getattr(self.config, 'rollout_iters', self.slot_attention.num_iterations)
        burnin_slots = []
        slots = None
        for t in range(self.burnin):
            iters = burnin_iters if t == 0 else rollout_iters
            slots, attn = self.slot_attention(feat[:, t], slots, num_iterations=iters)
            burnin_slots.append(slots)
        burnin_slots = torch.stack(burnin_slots, dim=1)

        dec_burnin = []
        dec_alphas = []
        dec_rgbs = []
        for t in range(self.burnin):
            recon, alpha, rgb = self.decoder(burnin_slots[:, t], return_rgb=True)
            dec_burnin.append(recon)
            dec_alphas.append(alpha)
            dec_rgbs.append(rgb)
        dec_burnin = torch.stack(dec_burnin, dim=1)
        dec_alphas = torch.stack(dec_alphas, dim=1)
        dec_rgbs = torch.stack(dec_rgbs, dim=1)

        return {
            "attn": attn,
            "encoded_feat": feat,
            "decoder_alpha": dec_alphas[:, 0],
            "decoder_rgb": dec_rgbs[:, 0],
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
        }
