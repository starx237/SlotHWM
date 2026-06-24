"""
实验1: 屏蔽depth，仅预测pos_x/pos_y，训练时空模块看能否收敛
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

# A: 只用 pos loss (depth 维度不参与)
# B: 用 pos + depth loss (完整 dyn loss)
model_A = load_model()
model_B = load_model()

for name, param in model_A.named_parameters():
    if 'spatiotemporal' not in name:
        param.requires_grad = False
for name, param in model_B.named_parameters():
    if 'spatiotemporal' not in name:
        param.requires_grad = False

opt_A = torch.optim.Adam(filter(lambda p: p.requires_grad, model_A.parameters()), lr=1e-4)
opt_B = torch.optim.Adam(filter(lambda p: p.requires_grad, model_B.parameters()), lr=1e-4)

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=8, shuffle=True, num_workers=0)

losses_A = deque(maxlen=50)
losses_B = deque(maxlen=50)

for step in range(500):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    
    # A: pos only
    model_A.eval()
    with torch.no_grad():
        out_A_eval = model_A(frames)
    target_A = out_A_eval['slots']['target'][:, :rollout]
    dm_A = out_A_eval['depth_mask'][:, :rollout]
    
    model_A.train()
    out_A = model_A(frames)
    pred_A = out_A['slots']['predicted'][:, :rollout]
    mask_A = dm_A.unsqueeze(-1).float()
    
    # 只用 pos (前2维) 的 loss
    pred_pos = pred_A[:, :, :, app_dim:app_dim+2]
    target_pos = target_A[:, :, :, app_dim:app_dim+2]
    loss_A = ((pred_pos - target_pos)**2 * mask_A).sum() / (mask_A.sum() * 2 + 1e-8)
    
    opt_A.zero_grad()
    loss_A.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model_A.parameters()), 1.0)
    opt_A.step()
    losses_A.append(loss_A.item())
    
    # B: pos + depth
    model_B.eval()
    with torch.no_grad():
        out_B_eval = model_B(frames)
    target_B = out_B_eval['slots']['target'][:, :rollout]
    dm_B = out_B_eval['depth_mask'][:, :rollout]
    
    model_B.train()
    out_B = model_B(frames)
    pred_B = out_B['slots']['predicted'][:, :rollout]
    mask_B = dm_B.unsqueeze(-1).float()
    
    pred_dyn = pred_B[:, :, :, app_dim:]
    target_dyn = target_B[:, :, :, app_dim:]
    loss_B = ((pred_dyn - target_dyn)**2 * mask_B).sum() / (mask_B.sum() * 3 + 1e-8)
    
    opt_B.zero_grad()
    loss_B.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model_B.parameters()), 1.0)
    opt_B.step()
    losses_B.append(loss_B.item())
    
    if step % 50 == 0:
        avg_A = sum(losses_A) / len(losses_A)
        avg_B = sum(losses_B) / len(losses_B)
        print(f"step {step:3d}: A(pos_only)={loss_A.item():.6f}(avg={avg_A:.6f}) | B(pos+depth)={loss_B.item():.6f}(avg={avg_B:.6f})")

# 评估
print("\n=== Evaluation ===")
for name, mdl in [("A(pos_only)", model_A), ("B(pos+depth)", model_B)]:
    mdl.eval()
    torch.manual_seed(999)
    batch = next(iter(ds.get_dataloader(batch_size=8, shuffle=True, num_workers=0)))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = mdl(frames)
    pred_dyn = out['slots']['predicted'][:, :rollout, :, app_dim:]
    target_dyn = out['slots']['target'][:, :rollout, :, app_dim:]
    mask = out['depth_mask'][:, :rollout].unsqueeze(-1).float()
    pos_mse = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask).sum() / (mask.sum()*2+1e-8)
    depth_mse = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask).sum() / (mask.sum()+1e-8)
    print(f"  {name}: pos={pos_mse.item():.6f} depth={depth_mse.item():.6f}")
    
    # 逐帧
    for t in [0, 4, 9]:
        pm = ((pred_dyn[:,t,:,:2] - target_dyn[:,t,:,:2])**2 * mask[:,t]).sum() / (mask[:,t].sum()*2+1e-8)
        dm = ((pred_dyn[:,t,:,2:3] - target_dyn[:,t,:,2:3])**2 * mask[:,t]).sum() / (mask[:,t].sum()+1e-8)
        print(f"    frame {t}: pos={pm.item():.6f} depth={dm.item():.6f}")

# baseline
model_base = load_model()
model_base.eval()
torch.manual_seed(999)
batch = next(iter(ds.get_dataloader(batch_size=8, shuffle=True, num_workers=0)))
frames = batch['video'].cuda()
with torch.no_grad():
    out = model_base(frames)
pred_dyn = out['slots']['predicted'][:, :rollout, :, app_dim:]
target_dyn = out['slots']['target'][:, :rollout, :, app_dim:]
mask = out['depth_mask'][:, :rollout].unsqueeze(-1).float()
pos_mse = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask).sum() / (mask.sum()*2+1e-8)
depth_mse = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask).sum() / (mask.sum()+1e-8)
print(f"\n  Baseline: pos={pos_mse.item():.6f} depth={depth_mse.item():.6f}")
