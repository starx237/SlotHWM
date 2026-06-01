# train1/optimizer.py —— 第一阶段优化器与学习率调度器创建

import torch
from torch.optim import Adam, AdamW


def create_optimizer(model_params, config):
    # 根据配置创建优化器：支持 Adam 和 AdamW
    if config.optimizer == "adam":
        optimizer = Adam(model_params, lr=config.lr, betas=(config.beta1, config.beta2),
                         weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        optimizer = AdamW(model_params, lr=config.lr, betas=(config.beta1, config.beta2),
                          weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # 配置学习率调度器：支持余弦退火（含线性预热）、阶梯下降
    if config.get("lr_scheduler") == "cosine_warmup":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
        warmup_epochs = config.get("warmup_epochs", 5)
        total_epochs = config.get("total_epochs", 100)

        # 自定义预热 + 余弦退火调度函数
        def warmup_cosine_lr(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            return 0.5 * (1 + torch.cos(torch.tensor((epoch - warmup_epochs) / (total_epochs - warmup_epochs) * 3.14159)))

        scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lr)
    elif config.get("lr_scheduler") == "step":
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=config.get("step_size", 100), gamma=config.get("gamma", 0.96))
    else:
        scheduler = None

    return optimizer, scheduler