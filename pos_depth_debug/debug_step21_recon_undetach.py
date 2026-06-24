"""
实验21: 核心验证 - 解除 recon_rollout 的 detach，让梯度流过 decoder → predictor

这是最终验证: recon 信号是否比 slot_loss 更有效
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
rollout = cfg.rollout_frames
burnin = cfg.burnin_frames

# 方案A: slot_loss only (当前方式)
model_A = SlotDynamicsModel(cfg).cuda()
ckpt = torch.load(cfg.pretrained_path, map_location='cpu')
model_state = model_A.state_dict()
loaded = {}
for mk in model_state:
    mk_c = mk.replace('_orig_mod.', '')
    for ck in ckpt['model']:
        ck_c = ck.replace('_orig_mod.', '')
        if ck_c == mk_c and ckpt['model'][ck].shape == model_state[mk].shape:
            loaded[mk] = ckpt['model'][ck]
            break
model_A.load_state_dict(loaded, strict=False)

for name, param in model_A.named_parameters():
    if 'spatiotemporal' not in name:
        param.requires_grad = False

# 方案B: recon_rollout 不 detach (允许梯度流过 decoder → predictor)
model_B = SlotDynamicsModel(cfg).cuda()
model_B.load_state_dict(loaded, strict=False)

# predictor + decoder 都可训练
for name, param in model_B.named_parameters():
    if 'spatiotemporal' not in name and 'decoder' not in name:
        param.requires_grad = False

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=4, shuffle=True, num_workers=0)

opt_A = torch.optim.Adam(filter(lambda p: p.requires_grad, model_A.parameters()), lr=1e-4)
opt_B = torch.optim.Adam(filter(lambda p: p.requires_grad, model_B.parameters()), lr=1e-4)

print("=== A: slot_loss only vs B: slot_loss + recon (undetached) ===")
for step in range(300):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    
    # A: 只用 slot_loss
    model_A.eval()
    with torch.no_grad():
        out_A = model_A(frames)
    target_S = out_A['slots']['target'][:, :rollout]
    depth_mask = out_A['depth_mask'][:, :rollout]
    
    model_A.train()
    out_A = model_A(frames)
    pred_dyn = out_A['slots']['predicted'][:, :rollout, :, app_dim:]
    target_dyn = target_S[:, :, :, app_dim:]
    mask = depth_mask.unsqueeze(-1).float()
    loss_A = ((pred_dyn - target_dyn)**2 * mask).sum() / (mask.sum() * pred_dyn.shape[-1] + 1e-8)
    
    opt_A.zero_grad()
    loss_A.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model_A.parameters()), 1.0)
    opt_A.step()
    
    # B: slot_loss + recon (undetached)
    model_B.train()
    out_B = model_B(frames)
    
    # slot_loss
    pred_dyn_B = out_B['slots']['predicted'][:, :rollout, :, app_dim:]
    target_dyn_B = out_B['slots']['target'][:, :rollout, :, app_dim:]
    slot_loss_B = ((pred_dyn_B - target_dyn_B)**2 * mask).sum() / (mask.sum() * pred_dyn_B.shape[-1] + 1e-8)
    
    # recon_loss (不 detach!)
    recon_pred = out_B['outputs']['video_pred']
    target_rollout = frames[:, burnin:burnin+rollout]
    recon_loss_B = F.mse_loss(recon_pred, target_rollout)
    
    total_B = slot_loss_B + recon_loss_B  # 都有梯度
    
    opt_B.zero_grad()
    total_B.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model_B.parameters()), 1.0)
    opt_B.step()
    
    if step % 30 == 0:
        # 检查 B 的 predictor 梯度来源
        st_grad_from_slot = 0
        st_grad_from_recon = 0
        
        # 重新计算看梯度分解
        model_B.zero_grad()
        
        # slot 部分
        slot_loss_B.backward(retain_graph=True)
        slot_gn = sum(p.grad.norm().item()**2 for p in model_B.predictor.spatiotemporal_module.parameters() if p.grad is not None)**0.5
        model_B.zero_grad()
        
        # recon 部分
        recon_loss_B.backward(retain_graph=True)
        recon_gn = sum(p.grad.norm().item()**2 for p in model_B.predictor.spatiotemporal_module.parameters() if p.grad is not None)**0.5
        model_B.zero_grad()
        
        print(f"step {step:3d}: A={loss_A.item():.6f} B_total={total_B.item():.6f} B_slot={slot_loss_B.item():.6f} B_recon={recon_loss_B.item():.6f} | st_grad: slot={slot_gn:.4f} recon={recon_gn:.4f}")

# 最终评估
print("\n=== Final evaluation ===")
for name, mdl in [("A (slot only)", model_A), ("B (slot+recon)", model_B)]:
    mdl.eval()
    torch.manual_seed(999)
    batch = next(iter(ds.get_dataloader(batch_size=4, shuffle=True, num_workers=0)))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = mdl(frames)
    pred_dyn = out['slots']['predicted'][:, :rollout, :, app_dim:]
    target_dyn = out['slots']['target'][:, :rollout, :, app_dim:]
    mask = out['depth_mask'][:, :rollout].unsqueeze(-1).float()
    pos_mse = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask).sum() / (mask.sum()*2+1e-8)
    depth_mse = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask).sum() / (mask.sum()+1e-8)
    print(f"  {name}: pos={pos_mse.item():.6f} depth={depth_mse.item():.6f}")
