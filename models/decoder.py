import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, slot_dim=128, output_channels=3,
                 hidden_channels=64, broadcast_size=8, img_size=128,
                 num_slots=None, predict_mask=True, use_alpha=True):
        super().__init__()
        self.broadcast_size = broadcast_size
        self.img_size = img_size
        self.predict_mask = predict_mask

        in_channels = slot_dim + 2

        num_upsample = int(math.log2(img_size // broadcast_size))
        backbone = []
        c_in = in_channels
        for i in range(num_upsample):
            c_out = hidden_channels if i < num_upsample - 2 else hidden_channels // 2
            backbone.extend([
                nn.ConvTranspose2d(c_in, c_out, kernel_size=5,
                                   stride=2, padding=2, output_padding=1),
                nn.ReLU(),
            ])
            c_in = c_out

        final_channels = hidden_channels // 2
        backbone.extend([
            nn.Conv2d(final_channels, final_channels, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
        ])
        self.backbone = nn.Sequential(*backbone)

        out_c = output_channels + 1 if predict_mask else output_channels
        self.out_conv = nn.Conv2d(final_channels, out_c, kernel_size=3,
                                  stride=1, padding=1)

    def forward(self, slots):
        B, N, D = slots.shape
        S = self.broadcast_size

        grid_x = torch.linspace(-1, 1, S, device=slots.device)
        grid_y = torch.linspace(-1, 1, S, device=slots.device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)
        grid = grid.expand(B, N, S, S, 2)

        slots_b = slots.view(B, N, D, 1, 1).expand(B, N, D, S, S)
        decoder_input = torch.cat([slots_b, grid.permute(0, 1, 4, 2, 3)], dim=2)
        decoder_input = decoder_input.reshape(B * N, -1, S, S)

        x = self.backbone(decoder_input)
        out = self.out_conv(x)  # (B*N, C+1, H, W)

        if self.predict_mask:
            out = out.reshape(B, N, -1, self.img_size, self.img_size)
            alpha = torch.softmax(out[:, :, -1:], dim=1)
            rgb = torch.sigmoid(out[:, :, :-1])
            blended = (rgb * alpha).sum(dim=1)
            output = blended.view(B, -1, self.img_size, self.img_size)
        else:
            out = out.reshape(B, N, -1, self.img_size, self.img_size)
            output = out.mean(dim=1)
            output = output.view(B, -1, self.img_size, self.img_size)

        return output
