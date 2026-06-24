"""
实验8: 只用第一帧 rollout 的 loss 训练，避免 BPTT 梯度爆炸
如果第一帧能收敛，说明问题在 BPTT；如果第一帧也不行，说明有更根本的问题
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

for name, param in model.named_parameters():
    if 'spatiotemporal' not in name:
        param.requires_grad = False

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=8, shuffle=True, num_workers=0)

app_dim = cfg.appearance_dim

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

print("=== Training with ONLY frame-0 Z^d loss (no BPTT) ===")
for step in range(500):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    
    model.eval()
    with torch.no_grad():
        out_eval = model(frames)
        target_S = out_eval['slots']['target'][:, :1]  # 只用第一帧！
        depth_mask = out_eval['depth_mask'][:, :1]
    
    model.train()
    out = model(frames)
    pred_S = out['slots']['predicted'][:, :1]  # 只用第一帧！
    
    mask = depth_mask.unsqueeze(-1).float()
    pred_dyn = pred_S[:, :, :, app_dim:]
    target_dyn = target_S[:, :, :, app_dim:]
    
    dyn_loss = ((pred_dyn - target_dyn)**2 * mask).sum() / (mask.sum() * pred_dyn.shape[-1] + 1e-8)
    
    optimizer.zero_grad()
    dyn_loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
    
    grad_norm = sum(p.grad.norm().item()**2 for p in model.predictor.parameters() if p.grad is not None)**0.5
    
    optimizer.step()
    
    if step % 50 == 0:
        with torch.no_grad():
            pos_mse = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask).sum() / (mask.sum()*2+1e-8)
            depth_mse = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask).sum() / (mask.sum()+1e-8)
        print(f"step {step:3d}: dyn_loss={dyn_loss.item():.6f} pos={pos_mse.item():.6f} depth={depth_mse.item():.6f} grad={grad_norm:.4f}")

print("\n=== Training with ONLY frame-0 Z^d loss, NO grad clip ===")
model2 = SlotDynamicsModel(cfg).cuda()
model2.load_state_dict(loaded, strict=False)
for name, param in model2.named_parameters():
    if 'spatiotemporal' not in name:
        param.requires_grad = False
optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=1e-4)

for step in range(500):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    
    model2.eval()
    with torch.no_grad():
        out_eval = model2(frames)
        target_S = out_eval['slots']['target'][:, :1]
        depth_mask = out_eval['depth_mask'][:, :1]
    
    model2.train()
    out = model2(frames)
    pred_S = out['slots']['predicted'][:, :1]
    
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
