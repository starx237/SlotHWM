"""
实验16: 长时间训练 + 移动平均 - 确认模型是否真的在改善
之前都是短训练，可能只是震荡中恰好看到了低点和高点
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
from collections import deque

with open('/autodl-fs/data/SlotHWM/config/obj3d.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)

app_dim = cfg.appearance_dim
rollout = cfg.rollout_frames

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

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

losses = deque(maxlen=50)
pos_losses = deque(maxlen=50)
depth_losses = deque(maxlen=50)

for step in range(2000):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    
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
    
    dyn_loss = ((pred_dyn - target_dyn)**2 * mask).sum() / (mask.sum() * pred_dyn.shape[-1] + 1e-8)
    
    optimizer.zero_grad()
    dyn_loss.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
    optimizer.step()
    
    with torch.no_grad():
        pos_mse = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask).sum() / (mask.sum()*2+1e-8)
        depth_mse = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask).sum() / (mask.sum()+1e-8)
    
    losses.append(dyn_loss.item())
    pos_losses.append(pos_mse.item())
    depth_losses.append(depth_mse.item())
    
    if step % 200 == 0:
        avg_loss = sum(losses) / len(losses)
        avg_pos = sum(pos_losses) / len(pos_losses)
        avg_depth = sum(depth_losses) / len(depth_losses)
        print(f"step {step:4d}: avg_dyn={avg_loss:.6f} avg_pos={avg_pos:.6f} avg_depth={avg_depth:.6f} (over last {len(losses)} steps)")

# 最终评估
print("\n=== Final eval ===")
model.eval()
torch.manual_seed(999)
batch = next(iter(ds.get_dataloader(batch_size=8, shuffle=True, num_workers=0)))
frames = batch['video'].cuda()
with torch.no_grad():
    out = model(frames)
pred_dyn = out['slots']['predicted'][:, :rollout, :, app_dim:]
target_dyn = out['slots']['target'][:, :rollout, :, app_dim:]
mask = out['depth_mask'][:, :rollout].unsqueeze(-1).float()
pos_mse = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask).sum() / (mask.sum()*2+1e-8)
depth_mse = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask).sum() / (mask.sum()+1e-8)
print(f"pos_mse={pos_mse.item():.6f}, depth_mse={depth_mse.item():.6f}")
for t in [0, 4, 9]:
    dm = ((pred_dyn[:,t,:,2:3] - target_dyn[:,t,:,2:3])**2 * mask[:,t]).sum() / (mask[:,t].sum()+1e-8)
    pm = ((pred_dyn[:,t,:,:2] - target_dyn[:,t,:,:2])**2 * mask[:,t]).sum() / (mask[:,t].sum()*2+1e-8)
    print(f"  frame {t}: pos={pm.item():.6f}, depth={dm.item():.6f}")

# 对比 baseline (未训练)
model_base = SlotDynamicsModel(cfg).cuda()
model_base.load_state_dict(loaded, strict=False)
model_base.eval()
with torch.no_grad():
    out_base = model_base(frames)
pred_dyn_base = out_base['slots']['predicted'][:, :rollout, :, app_dim:]
target_dyn_base = out_base['slots']['target'][:, :rollout, :, app_dim:]
mask_base = out_base['depth_mask'][:, :rollout].unsqueeze(-1).float()
pos_mse_base = ((pred_dyn_base[..., :2] - target_dyn_base[..., :2])**2 * mask_base).sum() / (mask_base.sum()*2+1e-8)
depth_mse_base = ((pred_dyn_base[..., 2:3] - target_dyn_base[..., 2:3])**2 * mask_base).sum() / (mask_base.sum()+1e-8)
print(f"\nBaseline: pos_mse={pos_mse_base.item():.6f}, depth_mse={depth_mse_base.item():.6f}")
