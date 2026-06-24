"""
实验23: 最终根本原因分析

核心问题: 为什么跨 batch 无法泛化？

假设: OBJ3D 数据集中，物体的运动方向是随机的（每个场景随机分配运动方向），
所以从历史位置无法预测未来运动方向。唯一能预测运动方向的是 "物体身份"。

但 ISA 的 slot ordering 是不确定的 — 同一个物体在不同场景中可能被分配到不同 slot。
所以即使模型知道了 "slot 0 向左移动"，在下一个场景中 slot 0 可能是另一个物体。

验证:
1. 在 OBJ3D 数据集中，同一个物体的运动方向是否在不同场景中一致？
2. 如果不一致，那跨场景泛化在理论上就不可能
3. 如果一致，那问题在于模型架构

关键: OBJ3D 中每个场景有 1-3 个前景物体，每个物体有随机运动方向。
同一个物体在不同场景中的运动方向是随机的。
所以 "从位置推断运动方向" 在理论上就是不可能的。

那正确的做法是什么？
- 需要从 buffer 中学习速度模式（最近几帧的位移趋势）
- buffer 中有 6 帧 burnin，足够推断速度
- 但目前的模型从 buffer 推断速度的效果如何？

让我直接测试: 用最近的 2 帧 delta 作为预测
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')

import torch
import numpy as np
import yaml
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset

with open('/autodl-fs/data/SlotHWM/config/obj3d.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)

app_dim = cfg.appearance_dim
rollout = cfg.rollout_frames
burnin = cfg.burnin_frames

model = SlotDynamicsModel(cfg).cuda()
ckpt = torch.load(cfg.pretrained_path, map_location='cpu')
model_state = model.state_dict()
loaded = {}
for mk in model_state:
    mk_c = mk.replace('_orig_mod.', '')
    for ck in ckpt['model']:
        ck_c = ck.replace('_orig_mod.', '')
        if ck_c == mk_c and ckpt['model'][ck].shape == model_state[mk].shape:
            loaded[mk] = ckpt['model'][ck]
            break
model.load_state_dict(loaded, strict=False)
model.eval()

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=8, shuffle=True, num_workers=0)

# 测试不同的启发式预测方法
# 方法1: 恒速外推 (last 2 burnin frames)
# 方法2: 加速外推 (last 3 burnin frames, linear fit)
# 方法3: 直接复制 (零速度)

for method_name in ['zero_velocity', 'constant_velocity', 'linear_extrapolation']:
    total_pos_mse = 0
    total_depth_mse = 0
    n_batches = 0
    
    for i in range(30):
        batch = next(iter(loader))
        frames = batch['video'].cuda()
        with torch.no_grad():
            out = model(frames)
        
        target_S = out['slots']['target']  # (B, rollout, N, D)
        corrected_S = out['slots']['corrected']  # (B, burnin, N, D)
        depth_mask = out['depth_mask']  # (B, rollout, N)
        B = frames.shape[0]
        
        burnin_dyn = corrected_S[:, :, :, app_dim:]  # (B, burnin, N, 3)
        
        # 自回归预测
        cur = burnin_dyn[:, -1]  # (B, N, 3)
        prev = burnin_dyn[:, -2]  # (B, N, 3)
        
        for t in range(rollout):
            if method_name == 'zero_velocity':
                pred = cur  # 不动
            elif method_name == 'constant_velocity':
                vel = cur - prev
                pred = cur + vel
            elif method_name == 'linear_extrapolation':
                if burnin >= 3:
                    prev2 = burnin_dyn[:, -3] if t == 0 else prev  # 简化
                    vel1 = cur - prev
                    vel0 = prev - prev2
                    acc = vel1 - vel0
                    pred = cur + vel1 + 0.5 * acc
                else:
                    vel = cur - prev
                    pred = cur + vel
            
            target_dyn = target_S[:, t, :, app_dim:]
            mask_t = depth_mask[:, t].unsqueeze(-1).float()
            
            pos_mse = ((pred[..., :2] - target_dyn[..., :2])**2 * mask_t).sum() / (mask_t.sum()*2+1e-8)
            depth_mse = ((pred[..., 2:3] - target_dyn[..., 2:3])**2 * mask_t).sum() / (mask_t.sum()+1e-8)
            total_pos_mse += pos_mse.item()
            total_depth_mse += depth_mse.item()
            n_batches += 1
            
            # 更新 (用 target 做 teacher forcing 看理论上限)
            prev = cur
            cur = target_dyn  # teacher forcing: 用真实值
    
    avg_pos = total_pos_mse / n_batches
    avg_depth = total_depth_mse / n_batches
    print(f"{method_name:25s}: pos_mse={avg_pos:.6f} depth_mse={avg_depth:.6f}")

# 也测 teacher forcing 但不用 (free-running)
print("\n=== Free-running (no teacher forcing) ===")
for method_name in ['constant_velocity']:
    total_pos_mse = 0
    total_depth_mse = 0
    n_batches = 0
    
    for i in range(30):
        batch = next(iter(loader))
        frames = batch['video'].cuda()
        with torch.no_grad():
            out = model(frames)
        
        target_S = out['slots']['target']
        corrected_S = out['slots']['corrected']
        depth_mask = out['depth_mask']
        B = frames.shape[0]
        
        burnin_dyn = corrected_S[:, :, :, app_dim:]
        cur = burnin_dyn[:, -1].clone()
        prev = burnin_dyn[:, -2].clone()
        vel = cur - prev
        
        for t in range(rollout):
            pred = cur + vel  # 恒速
            
            target_dyn = target_S[:, t, :, app_dim:]
            mask_t = depth_mask[:, t].unsqueeze(-1).float()
            
            pos_mse = ((pred[..., :2] - target_dyn[..., :2])**2 * mask_t).sum() / (mask_t.sum()*2+1e-8)
            depth_mse = ((pred[..., 2:3] - target_dyn[..., 2:3])**2 * mask_t).sum() / (mask_t.sum()+1e-8)
            total_pos_mse += pos_mse.item()
            total_depth_mse += depth_mse.item()
            n_batches += 1
            
            # Free-running: 用预测值
            prev = cur
            cur = pred
    
    avg_pos = total_pos_mse / n_batches
    avg_depth = total_depth_mse / n_batches
    print(f"{method_name:25s}: pos_mse={avg_pos:.6f} depth_mse={avg_depth:.6f}")

# 对比: 未训练的模型 baseline
print("\n=== Model baseline (untrained predictor) ===")
total_pos = 0
total_depth = 0
n = 0
for i in range(30):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = model(frames)
    pred_dyn = out['slots']['predicted'][:, :, :, app_dim:]
    target_dyn = out['slots']['target'][:, :, :, app_dim:]
    mask = out['depth_mask'].unsqueeze(-1).float()
    pos_mse = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask).sum() / (mask.sum()*2+1e-8)
    depth_mse = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask).sum() / (mask.sum()+1e-8)
    total_pos += pos_mse.item()
    total_depth += depth_mse.item()
    n += 1
print(f"  pos_mse={total_pos/n:.6f} depth_mse={total_depth/n:.6f}")
