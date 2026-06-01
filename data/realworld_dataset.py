# realworld_dataset.py
# 真实世界流固交互数据集加载模块（待实现）
# 包含真实拍摄的流体与物体交互视频数据

from .base_dataset import BaseVideoDataset

# TODO: 实现真实世界流固交互数据集加载逻辑
# 参考论文：603 个训练视频 + 25 个验证视频，30fps，10 秒以上时长
# 默认配置：6 个 slot，slot_dim=192，ResNet 编码器
# 10 帧预热 + 15 帧展开预测