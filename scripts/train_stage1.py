# -*- coding: utf-8 -*-
# SlotPi 第一阶段训练入口
# 训练 STATM-SAVi 编码器和解码器，用于从视频帧中提取 Slot 表示

import os
import sys
import argparse
import yaml
import torch
from slotpi.train1 import Stage1Trainer, create_optimizer
from slotpi.models import STATMSAVi
from slotpi.data import BaseVideoDataset
from slotpi.config.base_config import get_config

def setup_cuda():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

def main():
    setup_cuda()
    parser = argparse.ArgumentParser(description='SlotPi Stage 1: STATM-SAVi Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--workdir', type=str, default='./experiments/stage1/default', help='Working directory')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    cfg = get_config(config_dict)

    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建模型
    model = STATMSAVi(cfg).to(device)

    # 构建优化器
    optimizer, scheduler = create_optimizer(model.parameters(), cfg)

    # 构建数据加载器
    train_dataset = BaseVideoDataset(cfg, split='train')
    val_dataset = BaseVideoDataset(cfg, split='val')
    train_loader = train_dataset.get_dataloader(cfg.batch_size)
    val_loader = val_dataset.get_dataloader(cfg.batch_size)

    # 构建训练器
    trainer = Stage1Trainer(model, optimizer, scheduler, cfg)

    # 开始训练
    os.makedirs(os.path.join(args.workdir, 'checkpoints'), exist_ok=True)
    trainer.train(train_loader, val_loader, cfg.num_epochs)

if __name__ == '__main__':
    main()