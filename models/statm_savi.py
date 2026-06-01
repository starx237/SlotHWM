# statm_savi.py — STATM-SAVi 模型主模块
# 实现空间-时间注意力掩码式Slot视频目标分割模型（STATM-SAVi）
# 结合了空间注意力、时间Transformer和Slot Attention机制

import torch
import torch.nn as nn
from models.attention import SlotAttention, TimeSpaceTransformerBlock2
from models.misc import PositionEmbedding
from models.encoder import FrameEncoder, CNNEncoder, ResNetEncoder
from models.decoder import SpatialBroadcastDecoder
from typing import Optional


class CorrectorPredictorTuple:
    '''封装校正器（Slot Attention）和预测器（Transformer）的简单元组类'''
    def __init__(self, corrector, predictor):
        self.corrector = corrector
        self.predictor = predictor


class STATMSAVi(nn.Module):
    '''
    STATM-SAVi 模型主类。
    结合了编码器、Slot Attention 校正器、时空 Transformer 预测器和解码器。
    通过逐帧处理视频序列，提取 Slot 表示并做下一帧预测。
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config

        # === 编码器 ===
        if config.encoder_type == "cnn":
            # 使用 CNN 编码器
            self.encoder = CNNEncoder(
                in_channels=config.in_channels,
                hidden_channels=config.encoder_hidden,
                out_dim=config.slot_dim,
                img_size=config.img_size,
                pos_embedding=PositionEmbedding(),
                reduction="flatten",
            )
        elif config.encoder_type == "resnet":
            # 使用 ResNet 编码器
            self.encoder = ResNetEncoder(
                resnet_version=config.resnet_version,
                pretrained=config.resnet_pretrained,
                out_dim=config.slot_dim,
                pos_embedding=PositionEmbedding(),
                reduction="none",
            )

        # === 解码器 ===
        self.decoder = SpatialBroadcastDecoder(
            slot_dim=config.slot_dim,
            output_channels=config.in_channels,
            hidden_channels=config.decoder_hidden,
            broadcast_size=config.broadcast_size,
            num_slots=config.num_slots,
            predict_mask=True,
            use_alpha=True,
        )

        # === Slot Attention 校正器 ===
        self.slot_attention = SlotAttention(
            num_slots=config.num_slots,
            slot_dim=config.slot_dim,
            hidden_dim=config.slot_hidden,
            iters=config.slot_iters,
        )

        # === 时空 Transformer 预测器 ===
        self.predictor = nn.ModuleList([
            TimeSpaceTransformerBlock2(
                embed_dim=config.slot_dim,
                num_heads=config.num_heads,
                qkv_size=config.qkv_size,
                mlp_size=config.predictor_mlp_size,
                pre_norm=True,
            ) for _ in range(config.num_predictor_blocks)
        ])

        self.buffer_len = config.buffer_len
        # 将校正器和预测器组合成元组
        self._corrector_predictor = CorrectorPredictorTuple(
            self.slot_attention, self.predictor)

    def forward(self, frames, cond=None):
        '''
        Args:
            frames: 输入视频帧 (B, T, C, H, W)
            cond: 可选条件（当前未使用）
        Returns:
            包含解码视频、校正后 Slot 和预测 Slot 的字典
        '''
        B, T, C, H, W = frames.shape

        # 编码所有帧
        enc_features = self.encoder(frames)
        slots = None
        decoder_outputs = []
        pred_slots = []
        corr_slots = []

        # 初始化历史缓存 buffer (B, T, N, D)
        buffer = torch.zeros(B, self.buffer_len, self.config.num_slots,
                             self.config.slot_dim, device=frames.device)

        # 逐帧处理
        for t in range(T):
            feat_t = enc_features[:, t]
            # 校正：使用 Slot Attention 从当前帧特征中提取 Slot
            if slots is None:
                slots, attn = self.slot_attention(feat_t, None)
            else:
                slots, attn = self.slot_attention(feat_t, slots.detach())

            corr_slots.append(slots)

            # 预测下一帧的 Slot
            if t < T - 1:
                for block in self.predictor:
                    slots_pred = block(slots, buffer)
                pred_slots.append(slots_pred)
                buffer = torch.cat([buffer[:, 1:], slots_pred.unsqueeze(1)], dim=1)
            else:
                pred_slots.append(slots)

            # 解码当前帧
            dec_out = self.decoder(slots)
            decoder_outputs.append(dec_out)

        # 堆叠所有时间步的结果
        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        corr_slots = torch.stack(corr_slots, dim=1)
        pred_slots = torch.stack(pred_slots, dim=1)

        out = {
            "outputs": {"video": decoder_outputs},
            "slots": {"corrected": corr_slots, "predicted": pred_slots},
        }
        return out