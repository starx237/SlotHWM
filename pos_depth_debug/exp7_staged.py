"""
实验7: 分阶段训练
Phase 1: 只学 pos (屏蔽depth loss), 200步
Phase 2: 只学 depth (bnd_mask+decay, pos loss weight降低), 200步
Phase 3: 联合微调 (bnd_mask+decay, pos+depth), 100步
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')
import warnings; warnings.filterwarnings('ignore')
import torch
import numpy as np
import yaml
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset

with open('config/obj3d.yaml') as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))

app_dim = cfg.appearance_dim
depth_max = getattr(cfg, 'depth_max', 0.30)
boundary_threshold = 0.75
rollout = cfg.rollout_frames

def make():
    m = SlotDynamicsModel(cfg).cuda()
    ckpt = torch.load(cfg.pretrained_path, map_location='cpu')
    sd = m.state_dict()
    ld = {}
    for mk in sd:
        mc = mk.replace('_orig_mod.','')
        for ck in ckpt['model']:
            cc = ck.replace('_orig_mod.','')
            if cc==mc and ckpt['model'][ck].shape==sd[mk].shape:
                ld[mk]=ckpt['model'][ck]; break
    m.load_state_dict(ld, strict=False)
    for n,p in m.named_parameters():
        if 'spatiotemporal' not in n: p.requires_grad=False
    return m

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)

data_list = []
for seed in range(6):
    torch.manual_seed(seed)
    data_list.append(next(iter(ds.get_dataloader(batch_size=2,shuffle=False,num_workers=0)))['video'].cuda())

def compute_boundary_mask(slots, depth_mask):
    B, T, N, D = slots.shape
    pos_x = slots[:, :, :, app_dim]
    pos_y = slots[:, :, :, app_dim+1]
    depth = slots[:, :, :, app_dim+2]
    boundary_mask = (pos_x.abs() < boundary_threshold) & (pos_y.abs() < boundary_threshold) & (depth < depth_max)
    return (depth_mask & boundary_mask).float()

decay_weights = torch.tensor([0.9**t for t in range(rollout)], device='cuda')

m = make()
opt = torch.optim.Adam(filter(lambda p:p.requires_grad, m.parameters()), lr=1e-4)

def evaluate(mdl, ev_data, label=""):
    mdl.eval()
    with torch.no_grad():
        out = mdl(ev_data)
    pred = out['slots']['predicted'][:, :rollout, :, app_dim:]
    tgt = out['slots']['target'][:, :rollout, :, app_dim:]
    dm = out['depth_mask'][:, :rollout].unsqueeze(-1).float()
    bnd = compute_boundary_mask(out['slots']['target'][:, :rollout], out['depth_mask'][:, :rollout]).unsqueeze(-1)
    
    pos_mse = ((pred[...,:2]-tgt[...,:2])**2*dm).sum()/(dm.sum()*2+1e-8)
    depth_mse = ((pred[...,2:3]-tgt[...,2:3])**2*dm).sum()/(dm.sum()+1e-8)
    bnd_pos = ((pred[...,:2]-tgt[...,:2])**2*bnd).sum()/(bnd.sum()*2+1e-8)
    bnd_depth = ((pred[...,2:3]-tgt[...,2:3])**2*bnd).sum()/(bnd.sum()+1e-8)
    print(f"  {label:30s}: pos={pos_mse.item():.6f} depth={depth_mse.item():.6f} | bnd_pos={bnd_pos.item():.6f} bnd_depth={bnd_depth.item():.6f}")
    return {'pos': pos_mse.item(), 'depth': depth_mse.item(), 'bnd_pos': bnd_pos.item(), 'bnd_depth': bnd_depth.item()}

torch.manual_seed(99)
ev = next(iter(ds.get_dataloader(batch_size=2,shuffle=False,num_workers=0)))['video'].cuda()

print("=== Phase 1: pos only (200 steps) ===")
for step in range(200):
    frames = data_list[step % len(data_list)]
    m.train()
    out = m(frames)
    
    target = out['slots']['target'][:, :rollout].detach()
    pred = out['slots']['predicted'][:, :rollout]
    dm = out['depth_mask'][:, :rollout].detach()
    m0 = dm.unsqueeze(-1).float()
    
    pred_dyn = pred[:, :, :, app_dim:]
    tgt_dyn = target[:, :, :, app_dim:]
    
    # 只用 pos loss
    loss = ((pred_dyn[...,:2] - tgt_dyn[...,:2])**2 * m0).sum() / (m0.sum()*2+1e-8)
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad, m.parameters()), 1.0)
    opt.step()
    
    if step % 100 == 0:
        print(f"  step {step}: pos_loss={loss.item():.6f}", flush=True)

evaluate(m, ev, "After Phase1 (pos only)")

print("\n=== Phase 2: depth only + bnd_mask + decay (300 steps) ===")
# 降低学习率
for pg in opt.param_groups:
    pg['lr'] = 5e-5

for step in range(300):
    frames = data_list[(step+200) % len(data_list)]
    m.train()
    out = m(frames)
    
    target = out['slots']['target'][:, :rollout].detach()
    pred = out['slots']['predicted'][:, :rollout]
    dm = out['depth_mask'][:, :rollout].detach()
    
    bnd_mask = compute_boundary_mask(target, dm)
    bnd = bnd_mask.unsqueeze(-1)
    dw = decay_weights.view(1, -1, 1, 1)
    
    pred_dyn = pred[:, :, :, app_dim:]
    tgt_dyn = target[:, :, :, app_dim:]
    
    # 只用 depth loss (bnd_mask + decay)
    depth_loss = ((pred_dyn[...,2:3] - tgt_dyn[...,2:3])**2 * bnd * dw).sum() / (bnd.sum()+1e-8)
    # 加一个很小的 pos loss 防止 pos 漂移
    pos_loss = ((pred_dyn[...,:2] - tgt_dyn[...,:2])**2 * bnd).sum() / (bnd.sum()*2+1e-8)
    loss = depth_loss + 0.1 * pos_loss
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad, m.parameters()), 1.0)
    opt.step()
    
    if step % 100 == 0:
        print(f"  step {step}: depth_loss={depth_loss.item():.6f} pos_loss={pos_loss.item():.6f}", flush=True)

evaluate(m, ev, "After Phase2 (depth+bnd+0.1pos)")

print("\n=== Phase 3: joint finetune (bnd+decay) (200 steps) ===")
for pg in opt.param_groups:
    pg['lr'] = 2e-5

for step in range(200):
    frames = data_list[(step+500) % len(data_list)]
    m.train()
    out = m(frames)
    
    target = out['slots']['target'][:, :rollout].detach()
    pred = out['slots']['predicted'][:, :rollout]
    dm = out['depth_mask'][:, :rollout].detach()
    
    bnd_mask = compute_boundary_mask(target, dm)
    bnd = bnd_mask.unsqueeze(-1)
    dw = decay_weights.view(1, -1, 1, 1)
    
    pred_dyn = pred[:, :, :, app_dim:]
    tgt_dyn = target[:, :, :, app_dim:]
    
    pos_loss = ((pred_dyn[...,:2] - tgt_dyn[...,:2])**2 * bnd).sum() / (bnd.sum()*2+1e-8)
    depth_loss = ((pred_dyn[...,2:3] - tgt_dyn[...,2:3])**2 * bnd * dw).sum() / (bnd.sum()+1e-8)
    loss = pos_loss + depth_loss
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad, m.parameters()), 1.0)
    opt.step()
    
    if step % 100 == 0:
        print(f"  step {step}: pos={pos_loss.item():.6f} depth={depth_loss.item():.6f}", flush=True)

evaluate(m, ev, "After Phase3 (joint finetune)")

# 对比基线
print("\n=== Baselines ===")
evaluate(make(), ev, "Base (no training)")
evaluate(make(), ev, "A(orig, 500step reference)")

# 逐帧详细
print("\n=== Per-frame breakdown ===")
m.eval()
with torch.no_grad():
    out = m(ev)
pred = out['slots']['predicted'][:, :rollout, :, app_dim:]
tgt = out['slots']['target'][:, :rollout, :, app_dim:]
bnd = compute_boundary_mask(out['slots']['target'][:, :rollout], out['depth_mask'][:, :rollout]).unsqueeze(-1)
for t in range(rollout):
    b_t = bnd[:,t]
    bp = ((pred[:,t,:,:2]-tgt[:,t,:,:2])**2*b_t).sum()/(b_t.sum()*2+1e-8)
    bd = ((pred[:,t,:,2:3]-tgt[:,t,:,2:3])**2*b_t).sum()/(b_t.sum()+1e-8)
    print(f"  t={t}: bnd_pos={bp.item():.6f} bnd_depth={bd.item():.6f}")
