"""
实验21b: 简化版 - slot only vs slot+recon
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

def load_model():
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
    return model

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=4, shuffle=True, num_workers=0)

# A: slot_loss only
model_A = load_model()
for name, param in model_A.named_parameters():
    if 'spatiotemporal' not in name:
        param.requires_grad = False
opt_A = torch.optim.Adam(filter(lambda p: p.requires_grad, model_A.parameters()), lr=1e-4)

# B: slot_loss + recon (decoder also trainable)
model_B = load_model()
for name, param in model_B.named_parameters():
    if 'spatiotemporal' not in name and 'decoder' not in name:
        param.requires_grad = False
opt_B = torch.optim.Adam(filter(lambda p: p.requires_grad, model_B.parameters()), lr=1e-4)

for step in range(300):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    
    # A
    model_A.eval()
    with torch.no_grad():
        out_A_eval = model_A(frames)
    target_S = out_A_eval['slots']['target'][:, :rollout]
    depth_mask = out_A_eval['depth_mask'][:, :rollout]
    
    model_A.train()
    out_A = model_A(frames)
    pred_dyn = out_A['slots']['predicted'][:, :rollout, :, app_dim:]
    target_dyn = target_S[:, :, :, app_dim:]
    mask = depth_mask.unsqueeze(-1).float()
    loss_A = ((pred_dyn - target_dyn)**2 * mask).sum() / (mask.sum() * 3 + 1e-8)
    
    opt_A.zero_grad()
    loss_A.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model_A.parameters()), 1.0)
    opt_A.step()
    
    # B: slot + recon
    model_B.train()
    out_B = model_B(frames)
    
    pred_dyn_B = out_B['slots']['predicted'][:, :rollout, :, app_dim:]
    target_dyn_B = out_B['slots']['target'][:, :rollout, :, app_dim:]
    mask_B = out_B['depth_mask'][:, :rollout].unsqueeze(-1).float()
    slot_loss_B = ((pred_dyn_B - target_dyn_B)**2 * mask_B).sum() / (mask_B.sum() * 3 + 1e-8)
    
    recon_pred = out_B['outputs']['video_pred']
    target_rollout = frames[:, burnin:burnin+rollout]
    recon_loss_B = F.mse_loss(recon_pred, target_rollout)
    
    total_B = slot_loss_B + 10.0 * recon_loss_B  # recon 权重更大
    
    opt_B.zero_grad()
    total_B.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model_B.parameters()), 1.0)
    opt_B.step()
    
    if step % 50 == 0:
        print(f"step {step:3d}: A={loss_A.item():.6f} | B_slot={slot_loss_B.item():.6f} B_recon={recon_loss_B.item():.6f}")

# Final eval
print("\n=== Final ===")
for name, mdl in [("A (slot)", model_A), ("B (slot+recon)", model_B)]:
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
    recon = F.mse_loss(out['outputs']['video_pred'], frames[:, burnin:burnin+rollout])
    print(f"  {name}: pos={pos_mse.item():.6f} depth={depth_mse.item():.6f} recon={recon.item():.6f}")
