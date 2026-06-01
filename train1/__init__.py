# train1 模块：第一阶段训练
# 包含训练器、损失函数和优化器创建工具

from .trainer import Stage1Trainer
from .losses import ReconstructionLoss
from .optimizer import create_optimizer