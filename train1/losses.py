# train1/losses.py —— 第一阶段损失函数定义

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    # 重建损失：支持 MSE（均方误差）和 L1（绝对误差）两种损失函数
    def __init__(self, loss_type="mse", reduction="mean"):
        super().__init__()
        self.loss_type = loss_type      # 损失类型："mse" 或 "l1"
        self.reduction = reduction      # 归约方式："mean" / "sum" / "none"

    def forward(self, preds, targets):
        # 根据 loss_type 选择对应的损失函数计算预测值与目标值之间的误差
        if self.loss_type == "mse":
            loss = F.mse_loss(preds, targets, reduction=self.reduction)
        elif self.loss_type == "l1":
            loss = F.l1_loss(preds, targets, reduction=self.reduction)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
        return loss