# encoder.py — 编码器模块
# 提供帧编码器、CNN骨干网络、CNN编码器和ResNet编码器，用于将视频帧转换为特征表示

import torch
import torch.nn as nn
import torchvision
from typing import Optional, List


class FrameEncoder(nn.Module):
    '''通用帧编码器基类，使用 backbone 网络提取特征，支持位置编码和特征展平'''
    def __init__(self, backbone, pos_embedding=None, reduction="flatten"):
        super().__init__()
        self.backbone = backbone          # 特征提取骨干网络（CNN/ResNet等）
        self.pos_embedding = pos_embedding # 可选的位置编码模块
        self.reduction = reduction         # 特征降维方式："flatten" 或 "none"

    def forward(self, frames):
        '''
        Args:
            frames: 输入视频帧 (B, T, C, H, W)
        Returns:
            编码后的特征 (B, T, N, D)，其中 N 为空间位置数，D 为特征维度
        '''
        B, T, C, H, W = frames.shape
        # 将时空维度合并，以便一次性通过 backbone
        frames = frames.reshape(B * T, C, H, W)
        features = self.backbone(frames)
        if self.reduction == "flatten":
            # 展平空间维度并转置为 (B*T, N, D)
            features = features.view(features.shape[0], features.shape[1], -1)
            features = features.permute(0, 2, 1)
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"未知的 reduction 类型: {self.reduction}")
        # 可选的位置编码
        if self.pos_embedding is not None:
            features = self.pos_embedding(features)
        _, N, D = features.shape
        # 恢复时空维度
        features = features.view(B, T, N, D)
        return features


class CNNBackbone(nn.Module):
    '''CNN 骨干网络，由5层卷积组成，用于提取图像特征'''
    def __init__(self, in_channels=3, hidden_channels=64, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class CNNEncoder(FrameEncoder):
    '''基于 CNN 的编码器，使用 CNNBackbone 提取特征'''
    def __init__(self, in_channels=3, hidden_channels=64, out_dim=128,
                 img_size=64, pos_embedding=None, reduction="flatten"):
        backbone = CNNBackbone(in_channels, hidden_channels, out_dim)
        super().__init__(backbone, pos_embedding, reduction)


class ResNetBackbone(nn.Module):
    '''ResNet 骨干网络，支持 ResNet18/34/50，可选预训练权重'''
    def __init__(self, resnet_version=18, pretrained=False, out_dim=256, pool=True):
        super().__init__()
        # 选择 ResNet 版本并加载预训练权重
        if resnet_version == 18:
            weights = "IMAGENET1K_V1" if pretrained else None
            self.resnet = torchvision.models.resnet18(weights=weights)
        elif resnet_version == 34:
            weights = "IMAGENET1K_V1" if pretrained else None
            self.resnet = torchvision.models.resnet34(weights=weights)
        elif resnet_version == 50:
            weights = "IMAGENET1K_V2" if pretrained else None
            self.resnet = torchvision.models.resnet50(weights=weights)
        else:
            raise ValueError(f"不支持的 ResNet 版本: {resnet_version}")

        # 移除最后的全连接层，使用恒等映射代替
        in_feat = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.pool = pool
        # 如果输出维度不匹配则添加线性映射层
        self.out_proj = nn.Linear(in_feat, out_dim) if in_feat != out_dim else nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        if self.pool:
            # 池化模式下：添加空间维度以保持一致性
            x = self.out_proj(x)
            return x.unsqueeze(-1).unsqueeze(-1)
        return x


class ResNetEncoder(FrameEncoder):
    '''基于 ResNet 的编码器，使用 ResNetBackbone 提取特征'''
    def __init__(self, resnet_version=18, pretrained=False, out_dim=256,
                 pos_embedding=None, reduction="none"):
        backbone = ResNetBackbone(resnet_version, pretrained, out_dim)
        super().__init__(backbone, pos_embedding, reduction)