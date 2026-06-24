"""
实验7: 最核心的问题 - 为什么 predictor 无法学到 pos/depth 的变化？

假设: 问题不在 predictor 架构，而在训练信号本身。
可能原因:
1. recon_loss 对 rollout 部分被 detach (freeze_appearance=True)，导致
   decoder 无法提供关于 pos/depth 错误的反馈
2. slot_loss 直接计算 pred vs target 的 MSE，但这个信号太弱或被噪声淹没
3. 更根本: Z^d 只有3维 (pos_x, pos_y, depth)，而 Z^c 有64维。
   slot_loss 中 Z^c 部分 (detach) 贡献很大，Z^d 信号被淹没？

让我验证: 只用 Z^d 的 loss 训练，能否收敛？
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset

with open('/autodl-fs/data/SlotHWM/config/obj3d.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)

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

# 只训练 predictor 的 spatiotemporal 模块
for name, param in model.named_parameters():
    if 'spatiotemporal' not in name:
        param.requires_grad = False

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=8, shuffle=True, num_workers=0)

rollout = cfg.rollout_frames
app_dim = cfg.appearance_dim
lr = 1e-4

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

print("=== Training with ONLY Z^d (pos/depth) MSE loss ===")
for step in range(500):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    
    model.train()
    # 用 eval 模式的 target（避免 dropout 噪声）
    model.eval()
    with torch.no_grad():
        out_eval = model(frames)
        target_S = out_eval['slots']['target'][:, :rollout]
        depth_mask = out_eval['depth_mask'][:, :rollout]
    
    model.train()
    out = model(frames)
    pred_S = out['slots']['predicted'][:, :rollout]
    
    mask = depth_mask.unsqueeze(-1).float()
    pred_dyn = pred_S[:, :, :, app_dim:]
    target_dyn = target_S[:, :, :, app_dim:]
    
    # 只用 Z^d 的 MSE
    dyn_loss = ((pred_dyn - target_dyn)**2 * mask).sum() / (mask.sum() * pred_dyn.shape[-1] + 1e-8)
    
    optimizer.zero_grad()
    dyn_loss.backward()
    
    grad_norm = sum(p.grad.norm().item()**2 for p in model.predictor.parameters() if p.grad is not None)**0.5
    
    optimizer.step()
    
    if step % 50 == 0:
        with torch.no_grad():
            pos_mse = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask).sum() / (mask.sum()*2+1e-8)
            depth_mse = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask).sum() / (mask.sum()+1e-8)
        print(f"step {step:3d}: dyn_loss={dyn_loss.item():.6f} pos={pos_mse.item():.6f} depth={depth_mse.item():.6f} grad={grad_norm:.4f}")

# 现在测试: 用 eval target + 更大 lr
print("\n=== Training with ONLY Z^d loss, lr=1e-3 ===")
model2 = SlotDynamicsModel(cfg).cuda()
model2.load_state_dict(loaded, strict=False)
for name, param in model2.named_parameters():
    if 'spatiotemporal' not in name:
        param.requires_grad = False
optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=1e-3)

for step in range(500):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    
    model2.eval()
    with torch.no_grad():
        out_eval = model2(frames)
        target_S = out_eval['slots']['target'][:, :rollout]
        depth_mask = out_eval['depth_mask'][:, :rollout]
    
    model2.train()
    out = model2(frames)
    pred_S = out['slots']['predicted'][:, :rollout]
    
    mask = depth_mask.unsqueeze(-1).float()
    pred_dyn = pred_S[:, :, :, app_dim:]
    target_dyn = target_S[:, :, :, app_dim:]
    
    dyn_loss = ((pred_dyn - target_dyn)**2 * mask).sum() / (mask.sum() * pred_dyn.shape[-1] + 1e-8)
    
    optimizer2.zero_grad()
    dyn_loss.backward()
    optimizer2.step()
    
    if step % 50 == 0:
        with torch.no_grad():
            pos_mse = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask).sum() / (mask.sum()*2+1e-8)
            depth_mse = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask).sum() / (mask.sum()+1e-8)
        grad_norm = sum(p.grad.norm().item()**2 for p in model2.predictor.parameters() if p.grad is not None)**0.5
        print(f"step {step:3d}: dyn_loss={dyn_loss.item():.6f} pos={pos_mse.item():.6f} depth={depth_mse.item():.6f} grad={grad_norm:.4f}")
