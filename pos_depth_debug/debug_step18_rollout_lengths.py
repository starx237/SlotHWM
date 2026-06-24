"""
实验18: 确认 BPTT 不稳定性 - 逐步增加 rollout 长度

之前实验8显示单帧 loss 也震荡，但那是在不同 batch 上。
这次用同一个 batch + eval target + 单帧 loss
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

app_dim = cfg.appearance_dim

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

torch.manual_seed(42)
batch = next(iter(ds.get_dataloader(batch_size=8, shuffle=False, num_workers=0)))
frames = batch['video'].cuda()

model.eval()
with torch.no_grad():
    out = model(frames)
target_S = out['slots']['target']
depth_mask = out['depth_mask']

# 测试不同 rollout 长度: 1, 2, 5, 10
for max_t in [1, 2, 5, 10]:
    model_fresh = SlotDynamicsModel(cfg).cuda()
    model_fresh.load_state_dict(loaded, strict=False)
    for name, param in model_fresh.named_parameters():
        if 'spatiotemporal' not in name:
            param.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_fresh.parameters()), lr=1e-4)
    
    target_t = target_S[:, :max_t]
    mask_t = depth_mask[:, :max_t].unsqueeze(-1).float()
    
    best_loss = float('inf')
    for step in range(200):
        model_fresh.train()
        out = model_fresh(frames)
        pred_dyn = out['slots']['predicted'][:, :max_t, :, app_dim:]
        target_dyn = target_t[:, :, :, app_dim:]
        
        dyn_loss = ((pred_dyn - target_dyn)**2 * mask_t).sum() / (mask_t.sum() * pred_dyn.shape[-1] + 1e-8)
        
        optimizer.zero_grad()
        dyn_loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model_fresh.parameters()), 1.0)
        optimizer.step()
        
        if dyn_loss.item() < best_loss:
            best_loss = dyn_loss.item()
    
    model_fresh.eval()
    with torch.no_grad():
        out = model_fresh(frames)
        pred_dyn = out['slots']['predicted'][:, :max_t, :, app_dim:]
        target_dyn = target_t[:, :, :, app_dim:]
        dyn_loss = ((pred_dyn - target_dyn)**2 * mask_t).sum() / (mask_t.sum() * pred_dyn.shape[-1] + 1e-8)
        pos_mse = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask_t).sum() / (mask_t.sum()*2+1e-8)
        depth_mse = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask_t).sum() / (mask_t.sum()+1e-8)
    
    print(f"rollout={max_t:2d}: best={best_loss:.6f} final={dyn_loss.item():.6f} pos={pos_mse.item():.6f} depth={depth_mse.item():.6f}")

# 关键实验: 用 detach 截断 BPTT - 每步独立计算 loss，不回传到前一步
print("\n=== Detached BPTT: each step loss is independent ===")
model_det = SlotDynamicsModel(cfg).cuda()
model_det.load_state_dict(loaded, strict=False)
for name, param in model_det.named_parameters():
    if 'spatiotemporal' not in name:
        param.requires_grad = False
optimizer_det = torch.optim.Adam(filter(lambda p: p.requires_grad, model_det.parameters()), lr=1e-4)

for step in range(200):
    model_det.train()
    out = model_det(frames)
    pred_dyn = out['slots']['predicted'][:, :, :, app_dim:]
    target_dyn_full = target_S[:, :, :, app_dim:]
    mask_full = depth_mask.unsqueeze(-1).float()
    
    # 每一步独立：用 target 作为上一步的输入（teacher forcing），
    # 但 gradient 仍然通过 predictor 参数
    # 实际上无法用现有代码实现 teacher forcing...
    # 退而求其次: 只计算每步的 loss (不累积梯度)
    
    total_loss = 0
    for t in range(10):
        mask_t = mask_full[:, t]
        loss_t = ((pred_dyn[:, t] - target_dyn_full[:, t])**2 * mask_t).sum() / (mask_t.sum() * 3 + 1e-8)
        total_loss = total_loss + loss_t.detach().requires_grad_(True)  # 这不对...
    
    # 换个方式: 只用第一帧 loss
    mask_0 = mask_full[:, 0]
    loss_0 = ((pred_dyn[:, 0] - target_dyn_full[:, 0])**2 * mask_0).sum() / (mask_0.sum() * 3 + 1e-8)
    
    optimizer_det.zero_grad()
    loss_0.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model_det.parameters()), 1.0)
    optimizer_det.step()
    
    if step % 50 == 0:
        print(f"step {step:3d}: frame0_loss={loss_0.item():.6f}")

# 评估
model_det.eval()
with torch.no_grad():
    out = model_det(frames)
    pred_dyn = out['slots']['predicted'][:, :, :, app_dim:]
    target_dyn = target_S[:, :, :, app_dim:]
    mask_full = depth_mask.unsqueeze(-1).float()
    pos_mse = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask_full).sum() / (mask_full.sum()*2+1e-8)
    depth_mse = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask_full).sum() / (mask_full.sum()+1e-8)
print(f"\nDetached BPTT result: pos_mse={pos_mse.item():.6f}, depth_mse={depth_mse.item():.6f}")
for t in [0, 4, 9]:
    dm = ((pred_dyn[:,t,:,2:3] - target_dyn[:,t,:,2:3])**2 * mask_full[:,t]).sum() / (mask_full[:,t].sum()+1e-8)
    pm = ((pred_dyn[:,t,:,:2] - target_dyn[:,t,:,:2])**2 * mask_full[:,t]).sum() / (mask_full[:,t].sum()*2+1e-8)
    print(f"  frame {t}: pos={pm.item():.6f}, depth={dm.item():.6f}")
