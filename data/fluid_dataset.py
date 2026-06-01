# fluid_dataset.py
# 流体（Navier-Stokes）数据集加载模块（待实现）
# 包含基于 Navier-Stokes 方程的流体动力学模拟数据

from .base_dataset import BaseVideoDataset

# TODO: 实现流体（Navier-Stokes）数据集加载逻辑
# 参考论文：30 条训练轨迹，10 条测试轨迹
# 使用 patch_size=2 进行 Patch 嵌入，512 维特征，1024 个 token
# 网格大小 64x64，时间步长 16*dt