# -*- coding: utf-8 -*-
# SlotPi 第二阶段训练入口
# 训练 SlotPi 物理模块 + 时空推理模块，基于预提取 slots 或固定编码器

import os
import sys
import argparse
import yaml
import torch
from train2 import Stage2Trainer, create_optimizer
from models import SlotPiModel
from data import BaseVideoDataset
from config.base_config import get_config

def setup_cuda():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

def main():
    setup_cuda()
    parser = argparse.ArgumentParser(description='SlotPi Stage 2: Physics + Spatiotemporal Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--workdir', type=str, default='./experiments/stage2/default', help='Working directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume checkpoint path')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    cfg = get_config(config_dict, stage=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建模型
    model = SlotPiModel(cfg).to(device)

    # 加载预训练编码器（第二阶段通常固定编码器）
    if cfg.load_pretrained_encoder:
        state = torch.load(cfg.load_pretrained_encoder, map_location=device)
        model.stage1_encoder.load_state_dict(state, strict=False)
        if cfg.freeze_encoder:
            for p in model.stage1_encoder.parameters():
                p.requires_grad = False
        print(f'Loaded pretrained encoder from {cfg.load_pretrained_encoder}')

    # 构建优化器（只训练需要梯度的参数）
    params_to_train = [p for p in model.parameters() if p.requires_grad]
    optimizer, scheduler = create_optimizer(params_to_train, cfg)

    # 构建数据加载器
    train_dataset = BaseVideoDataset(cfg, split='train')
    val_dataset = BaseVideoDataset(cfg, split='val')
    train_loader = train_dataset.get_dataloader(cfg.batch_size)
    val_loader = val_dataset.get_dataloader(cfg.batch_size)

    # 构建训练器
    trainer = Stage2Trainer(model, optimizer, scheduler, cfg)

    # 恢复 checkpoint（如果需要）
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            start_epoch, _ = trainer.load_checkpoint(args.resume)

    os.makedirs(os.path.join(args.workdir, 'checkpoints'), exist_ok=True)
    trainer.train(train_loader, val_loader, cfg.num_epochs, start_epoch=start_epoch)

if __name__ == '__main__':
    main()