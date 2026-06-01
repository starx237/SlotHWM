# train2 模块：第二阶段训练
# 包含 slot 预测训练器、SlotPi 损失函数和优化器创建工具

from .trainer import Stage2Trainer
from .losses import SlotPiLoss
from .optimizer import create_optimizer