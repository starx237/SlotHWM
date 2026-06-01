# -*- coding: utf-8 -*-
# SlotPi 评估入口
# 评估已训练的 SlotPi 或 STATM-SAVi 模型

import os
import sys
import argparse
import yaml
import torch
from evaluation import Evaluator
from models import SlotPiModel, STATMSAVi
from data import BaseVideoDataset
from config.base_config import get_config

def setup_cuda():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

def main():
    setup_cuda()
    parser = argparse.ArgumentParser(description='SlotPi Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--stage', type=int, default=2, choices=[1, 2], help='Model stage (1=STATM-SAVi, 2=SlotPi)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--workdir', type=str, default='./experiments/eval/default', help='Working directory')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    cfg = get_config(config_dict, stage=args.stage)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.stage == 1:
        model = STATMSAVi(cfg).to(device)
    else:
        model = SlotPiModel(cfg).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    print(f'Loaded checkpoint from {args.checkpoint} (epoch {state.get("epoch", "?")})')

    val_dataset = BaseVideoDataset(cfg, split='val')
    val_loader = val_dataset.get_dataloader(cfg.batch_size, shuffle=False)

    evaluator = Evaluator(cfg)
    results = evaluator.evaluate(model, val_loader)

    print('Evaluation Results:')
    for key, value in results.items():
        print(f'  {key}: {value:.6f}')

if __name__ == '__main__':
    main()