"""
实验15: 分析 slot_loss 震荡的根本原因

从之前的实验看到:
1. 直接优化 pred_S 可以收敛
2. 简单 MLP 自回归训练会震荡，但能学到东西
3. 实际训练中时空模块和哈密顿模块都震荡

可能的根本原因:
A. 梯度裁剪不够 → 但 grad clip 1.0 仍然震荡
B. 学习率太大 → 但 1e-4 已经很小
C. 自回归误差累积 → 每步的小误差在 rollout 中指数放大
D. **teacher forcing vs free-running**: 训练时用 pred 作为下一步输入，
   误差累积导致梯度信号不一致

让我测试: 用 teacher forcing (target 作为下一步输入) 训练时空模块
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from models.attention import TimeSpaceTransformerBlock2
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

# 只训练 spatiotemporal 模块
for name, param in model.named_parameters():
    if 'spatiotemporal' not in name:
        param.requires_grad = False

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=8, shuffle=True, num_workers=0)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

print("=== Teacher forcing training ===")
for step in range(500):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    
    model.eval()
    with torch.no_grad():
        out_eval = model(frames)
        target_S = out_eval['slots']['target'][:, :rollout]
        depth_mask = out_eval['depth_mask'][:, :rollout]
        corrected_S = out_eval['slots']['corrected']
    
    # Teacher forcing: 用 target 的 Z^d 作为每步的输入
    # 而不是用上一步的预测
    # 但这需要修改 predictor 的 forward...
    
    # 实际上，我需要手动做 rollout
    # 更简单的方法: 只计算单步预测 loss (frame t → frame t+1)
    # 用 target[t] 作为输入，预测 target[t+1]
    
    model.train()
    
    # 获取 burnin 和 target 的 Z 空间表示
    with torch.no_grad():
        # burnin Z
        burnin_Z = []
        for t in range(burnin):
            S_t = corrected_S[:, t]
            Z_app = model.f_z(S_t[:, :, :app_dim])
            Z_t = torch.cat([Z_app, S_t[:, :, app_dim:]], dim=-1)
            burnin_Z.append(Z_t)
        burnin_Z = torch.stack(burnin_Z, dim=1)
        
        # target Z
        target_Z = []
        for t in range(rollout):
            S_t = target_S[:, t]
            Z_app = model.f_z(S_t[:, :, :app_dim])
            Z_t = torch.cat([Z_app, S_t[:, :, app_dim:]], dim=-1)
            target_Z.append(Z_t)
        target_Z = torch.stack(target_Z, dim=1)
    
    # Teacher forcing: 对每一步，用 target Z 作为输入
    total_loss = 0
    C = model.predictor.compute_C(burnin_Z)
    
    for t in range(rollout):
        if t == 0:
            cur_Z = burnin_Z[:, -1]  # burnin 最后一帧
        else:
            cur_Z = target_Z[:, t-1]  # teacher forcing!
        
        # 构建 buffer
        buf_start = max(0, t - burnin + 1)
        if t == 0:
            buffer_Z = burnin_Z
        else:
            # burnin + target[0:t]
            buffer_Z = torch.cat([burnin_Z, target_Z[:, :t]], dim=1)
            buffer_Z = buffer_Z[:, -(burnin + t):]  # 截断到 buffer_len
        
        # Forward predictor
        pred_Z = model.predictor(cur_Z, buffer_Z, C=C)
        
        # Loss: 只在 Z^d 空间
        pred_dyn = pred_Z[:, :, app_dim:]
        target_dyn_t = target_Z[:, t, :, app_dim:]
        
        mask_t = depth_mask[:, t].unsqueeze(-1).float()
        loss_t = ((pred_dyn - target_dyn_t)**2 * mask_t).sum() / (mask_t.sum() * pred_dyn.shape[-1] + 1e-8)
        total_loss = total_loss + loss_t
    
    total_loss = total_loss / rollout
    
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
    optimizer.step()
    
    if step % 50 == 0:
        print(f"step {step:3d}: loss={total_loss.item():.6f}")

# 评估 free-running (不用 teacher forcing)
print("\n=== Free-running evaluation ===")
model.eval()
torch.manual_seed(999)
batch = next(iter(ds.get_dataloader(batch_size=4, shuffle=True, num_workers=0)))
frames = batch['video'].cuda()

with torch.no_grad():
    out = model(frames)
    pred_S = out['slots']['predicted']
    target_S = out['slots']['target']
    depth_mask = out['depth_mask']
    mask = depth_mask.unsqueeze(-1).float()
    
    pred_dyn = pred_S[:, :, :, app_dim:]
    target_dyn = target_S[:, :, :, app_dim:]
    
    pos_mse = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask).sum() / (mask.sum()*2+1e-8)
    depth_mse = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask).sum() / (mask.sum()+1e-8)
    
    print(f"pos_mse={pos_mse.item():.6f}, depth_mse={depth_mse.item():.6f}")
    for t in [0, 4, 9]:
        d_err = ((pred_dyn[:,t,:,2:3] - target_dyn[:,t,:,2:3])**2 * mask[:,t]).sum() / (mask[:,t].sum()+1e-8)
        p_err = ((pred_dyn[:,t,:,:2] - target_dyn[:,t,:,:2])**2 * mask[:,t]).sum() / (mask[:,t].sum()*2+1e-8)
        print(f"  frame {t}: pos={p_err.item():.6f}, depth={d_err.item():.6f}")
