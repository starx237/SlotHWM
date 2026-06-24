"""
实验9: 最佳方案 pos1_d3 (bnd_mask+decay, depth_w=3) 长训练验证
800步，更多数据
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
for seed in range(10):
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

pos_w, depth_w = 1.0, 3.0

for step in range(800):
    frames = data_list[step % len(data_list)]
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
    loss = pos_w * pos_loss + depth_w * depth_loss
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad, m.parameters()), 1.0)
    opt.step()
    
    if step % 100 == 0:
        print(f"step {step:3d}: pos={pos_loss.item():.6f} depth={depth_loss.item():.6f} total={loss.item():.6f}", flush=True)

# 评估 - 多个种子取平均
print("\n=== Evaluation (multi-seed) ===")
m.eval()
all_pos, all_depth, all_bnd_pos, all_bnd_depth = [], [], [], []

for seed in [42, 99, 123, 456, 789]:
    torch.manual_seed(seed)
    ev = next(iter(ds.get_dataloader(batch_size=2,shuffle=False,num_workers=0)))['video'].cuda()
    with torch.no_grad():
        out = m(ev)
    pred = out['slots']['predicted'][:, :rollout, :, app_dim:]
    tgt = out['slots']['target'][:, :rollout, :, app_dim:]
    dm = out['depth_mask'][:, :rollout].unsqueeze(-1).float()
    bnd = compute_boundary_mask(out['slots']['target'][:, :rollout], out['depth_mask'][:, :rollout]).unsqueeze(-1)
    
    all_pos.append(((pred[...,:2]-tgt[...,:2])**2*dm).sum()/(dm.sum()*2+1e-8).item())
    all_depth.append(((pred[...,2:3]-tgt[...,2:3])**2*dm).sum()/(dm.sum()+1e-8).item())
    all_bnd_pos.append(((pred[...,:2]-tgt[...,:2])**2*bnd).sum()/(bnd.sum()*2+1e-8).item())
    all_bnd_depth.append(((pred[...,2:3]-tgt[...,2:3])**2*bnd).sum()/(bnd.sum()+1e-8).item())

# 基线
base_pos, base_depth, base_bnd_pos, base_bnd_depth = [], [], [], []
m0 = make()
m0.eval()
for seed in [42, 99, 123, 456, 789]:
    torch.manual_seed(seed)
    ev = next(iter(ds.get_dataloader(batch_size=2,shuffle=False,num_workers=0)))['video'].cuda()
    with torch.no_grad():
        out = m0(ev)
    pred = out['slots']['predicted'][:, :rollout, :, app_dim:]
    tgt = out['slots']['target'][:, :rollout, :, app_dim:]
    dm = out['depth_mask'][:, :rollout].unsqueeze(-1).float()
    bnd = compute_boundary_mask(out['slots']['target'][:, :rollout], out['depth_mask'][:, :rollout]).unsqueeze(-1)
    
    base_pos.append(((pred[...,:2]-tgt[...,:2])**2*dm).sum()/(dm.sum()*2+1e-8).item())
    base_depth.append(((pred[...,2:3]-tgt[...,2:3])**2*dm).sum()/(dm.sum()+1e-8).item())
    base_bnd_pos.append(((pred[...,:2]-tgt[...,:2])**2*bnd).sum()/(bnd.sum()*2+1e-8).item())
    base_bnd_depth.append(((pred[...,2:3]-tgt[...,2:3])**2*bnd).sum()/(bnd.sum()+1e-8).item())

print(f"{'':20s} {'pos':>10s} {'depth':>10s} {'bnd_pos':>10s} {'bnd_depth':>10s}")
print(f"{'Base (avg)':20s} {np.mean(base_pos):10.6f} {np.mean(base_depth):10.6f} {np.mean(base_bnd_pos):10.6f} {np.mean(base_bnd_depth):10.6f}")
print(f"{'Trained (avg)':20s} {np.mean(all_pos):10.6f} {np.mean(all_depth):10.6f} {np.mean(all_bnd_pos):10.6f} {np.mean(all_bnd_depth):10.6f}")

# 逐帧详细
print("\n=== Per-frame breakdown ===")
torch.manual_seed(99)
ev = next(iter(ds.get_dataloader(batch_size=2,shuffle=False,num_workers=0)))['video'].cuda()
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
