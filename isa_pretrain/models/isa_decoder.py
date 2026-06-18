import math
import torch
import torch.nn as nn
from isa_pretrain.models.isa_misc import create_coordinate_grid


class ISASpatialBroadcastDecoder(nn.Module):
    def __init__(self, appearance_dim=64, output_channels=3,
                 hidden_channels=64, broadcast_size=8, img_size=64,
                 num_slots=7, predict_mask=True, scales_factor=5.0):
        super().__init__()
        self.appearance_dim = appearance_dim
        self.broadcast_size = broadcast_size
        self.img_size = img_size
        self.num_slots = num_slots
        self.predict_mask = predict_mask
        self.scales_factor = scales_factor

        self.grid_proj = nn.Linear(2, appearance_dim)

        num_upsample = int(math.log2(img_size // broadcast_size))
        num_regular = 5 - num_upsample

        backbone = []
        c_in = appearance_dim
        for i in range(num_upsample):
            backbone.extend([
                nn.ConvTranspose2d(c_in, hidden_channels, kernel_size=5,
                                   stride=2, padding=2, output_padding=1),
                nn.ReLU(),
            ])
            c_in = hidden_channels
        for i in range(num_regular):
            backbone.extend([
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5,
                          stride=1, padding=2),
                nn.ReLU(),
            ])
        self.backbone = nn.Sequential(*backbone)

        out_c = output_channels + 1 if predict_mask else output_channels
        self.out_conv = nn.Conv2d(hidden_channels, out_c, kernel_size=3,
                                  stride=1, padding=1)

    def forward(self, slots, return_alphas=False, return_rgb=False):
        B, N, D = slots.shape
        appearance = slots[..., :-3]
        positions = slots[..., -3:-1]
        depth = slots[..., -1:]

        S = self.broadcast_size

        broadcast = appearance.view(B * N, self.appearance_dim, 1, 1)
        broadcast = broadcast.expand(-1, -1, S, S)

        grid = create_coordinate_grid(S, S, slots.device)
        grid = grid.unsqueeze(0).unsqueeze(0)
        grid = grid.expand(B, N, S, S, 2)

        relative_grid = grid - positions.view(B, N, 1, 1, 2)
        relative_grid = relative_grid / self.scales_factor
        relative_grid = relative_grid / (depth.view(B, N, 1, 1, 1) + 1e-8)

        pos_emb = self.grid_proj(relative_grid)
        pos_emb = pos_emb.permute(0, 1, 4, 2, 3)

        x = self.backbone(
            broadcast + pos_emb.reshape(B * N, self.appearance_dim, S, S)
        )
        out = self.out_conv(x)

        out = out.reshape(B, N, -1, self.img_size, self.img_size)

        if self.predict_mask:
            alpha = torch.softmax(out[:, :, -1:], dim=1)
            rgb = torch.sigmoid(out[:, :, :-1])
            blended = (rgb * alpha).sum(dim=1)
            out_img = blended.view(B, -1, self.img_size, self.img_size)
            if return_rgb:
                return out_img, alpha, rgb
            if return_alphas:
                return out_img, alpha
            return out_img
        else:
            return out.mean(dim=1).view(B, -1, self.img_size, self.img_size)
