# decoder.py — 解码器模块
# 提供空间广播解码器，将 Slot 特征解码为视频帧，支持 alpha 混合的可预测掩码

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialBroadcastDecoder(nn.Module):
    '''
    空间广播解码器。
    将每个 Slot 特征广播到 2D 空间网格上，通过 CNN 解码生成图像块，
    再通过 alpha 掩码混合所有 Slot 的输出得到最终图像。
    '''
    def __init__(self, slot_dim=128, output_channels=3,
                 hidden_channels=64, broadcast_size=8, num_slots=None,
                 predict_mask=True, use_alpha=True):
        super().__init__()
        self.slot_dim = slot_dim                  # Slot 特征维度
        self.output_channels = output_channels     # 输出图像的通道数（如 RGB=3）
        self.hidden_channels = hidden_channels     # CNN 隐藏通道数
        self.broadcast_size = broadcast_size       # 广播网格大小（如 8x8）
        self.num_slots = num_slots                 # Slot 数量（目前未使用）
        self.predict_mask = predict_mask           # 是否预测 alpha 掩码
        self.use_alpha = use_alpha                 # 是否使用 alpha 混合

        # 输入通道数 = Slot 特征维度 + 2个坐标通道
        in_channels = slot_dim + 2
        # 解码 CNN：5层卷积 + ReLU
        self.decoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )

        # 输出卷积层：若预测掩码则输出通道为 output_channels + 1（额外 alpha 通道）
        out_c = output_channels + 1 if predict_mask else output_channels
        self.out_conv = nn.Conv2d(hidden_channels, out_c, kernel_size=3, stride=1, padding=1)

    def forward(self, slots):
        '''
        Args:
            slots: (B, N, D) 当前帧的 Slot 特征
        Returns:
            解码后的图像 (B, C, H, W)
        '''
        B, N, D = slots.shape
        S = self.broadcast_size

        # 创建 2D 坐标网格 [-1, 1]
        grid_x = torch.linspace(-1, 1, S, device=slots.device)
        grid_y = torch.linspace(-1, 1, S, device=slots.device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)
        grid = grid.expand(B, N, S, S, 2)

        # 将 Slot 广播到网格的每个位置
        slots_b = slots.view(B, N, D, 1, 1).expand(B, N, D, S, S)
        # 拼接 Slot 特征和坐标编码
        decoder_input = torch.cat([slots_b, grid.permute(0, 1, 4, 2, 3)], dim=2)
        decoder_input = decoder_input.reshape(B * N, -1, S, S)

        # CNN 解码
        x = self.decoder_cnn(decoder_input)
        out = self.out_conv(x)

        if self.predict_mask:
            out = out.reshape(B, N, -1, S, S)
            alpha = torch.softmax(out[:, :, -1:], dim=1)
            rgb = out[:, :, :-1]
            blended = (rgb * alpha).sum(dim=1)
            output = blended.view(B, -1, S, S)
        else:
            out = out.reshape(B, N, -1, S, S)
            output = out.mean(dim=1)
            output = output.view(B, -1, S, S)

        return output