"""
实验4+5: 严格 loss mask 训练 + 单独学 depth
A: 原始 loss (pos+depth)
B: 严格 boundary mask loss (pos+depth, 物体靠近边界时mask掉)
C: 严格 boundary mask + 只学 depth (pos不参与loss)
D: 只学 depth (无 boundary mask)
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

# 预加载数据
torch.manual_seed(0)
d1 = next(iter(ds.get_dataloader(batch_size=2,shuffle=False,num_workers=0)))['video'].cuda()
torch.manual_seed(1)
d2 = next(iter(ds.get_dataloader(batch_size=2,shuffle=False,num_workers=0)))['video'].cuda()
torch.manual_seed(2)
d3 = next(iter(ds.get_dataloader(batch_size=2,shuffle=False,num_workers=0)))['video'].cuda()
batches = [d1, d2, d3]

def compute_boundary_mask(slots, depth_mask):
    """
    当物体的 pos 靠近边界或 depth 超过 depth_max 时，mask 掉该 slot
    slots: (B, T, N, D)
    depth_mask: (B, T, N) 原始 depth_mask
    返回: (B, T, N) 增强版 mask
    """
    B, T, N, D = slots.shape
    pos_x = slots[:, :, :, app_dim]     # (B, T, N)
    pos_y = slots[:, :, :, app_dim+1]   # (B, T, N)
    depth = slots[:, :, :, app_dim+2]   # (B, T, N)
    
    boundary_mask = (pos_x.abs() < boundary_threshold) & (pos_y.abs() < boundary_threshold) & (depth < depth_max)
    enhanced_mask = depth_mask & boundary_mask
    return enhanced_mask.float()

models = {
    'A(orig)': make(),
    'B(bnd_mask)': make(),
    'C(depth_only+bnd)': make(),
    'D(depth_only)': make(),
}
opts = {k: torch.optim.Adam(filter(lambda p:p.requires_grad, m.parameters()), lr=1e-4) for k, m in models.items()}

# 300步训练
for step in range(300):
    frames = batches[step % len(batches)]
    
    for name, mdl in models.items():
        mdl.eval()
        with torch.no_grad():
            out = mdl(frames)
        
        target = out['slots']['target'][:, :rollout]
        pred = out['slots']['predicted'][:, :rollout]
        dm = out['depth_mask'][:, :rollout]
        
        # 计算边界增强 mask (基于 target)
        bnd_mask = compute_boundary_mask(target, dm)  # (B, T, N)
        mask = bnd_mask.unsqueeze(-1)  # (B, T, N, 1)
        
        pred_dyn = pred[:, :, :, app_dim:]
        tgt_dyn = target[:, :, :, app_dim:]
        
        if name == 'A(orig)':
            # 原始: pos+depth, 原始 depth_mask
            m0 = dm.unsqueeze(-1).float()
            loss = ((pred_dyn - tgt_dyn)**2 * m0).sum() / (m0.sum()*3+1e-8)
        elif name == 'B(bnd_mask)':
            # 严格mask: pos+depth, 边界增强mask
            loss = ((pred_dyn - tgt_dyn)**2 * mask).sum() / (mask.sum()*3+1e-8)
        elif name == 'C(depth_only+bnd)':
            # 只学 depth, 边界增强mask
            pred_d = pred_dyn[..., 2:3]
            tgt_d = tgt_dyn[..., 2:3]
            loss = ((pred_d - tgt_d)**2 * mask).sum() / (mask.sum()+1e-8)
        elif name == 'D(depth_only)':
            # 只学 depth, 原始mask
            m0 = dm.unsqueeze(-1).float()
            pred_d = pred_dyn[..., 2:3]
            tgt_d = tgt_dyn[..., 2:3]
            loss = ((pred_d - tgt_d)**2 * m0).sum() / (m0.sum()+1e-8)
        
        mdl.train()
        opt = opts[name]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad, mdl.parameters()), 1.0)
        opt.step()
        
        if step % 50 == 0:
            print(f"step {step:3d} {name}: {loss.item():.6f}", flush=True)

# 评估
print("\n=== Evaluation ===")
torch.manual_seed(99)
ev = next(iter(ds.get_dataloader(batch_size=2,shuffle=False,num_workers=0)))['video'].cuda()

for name, mdl in [('Base', make())] + list(models.items()):
    mdl.eval()
    with torch.no_grad():
        out = mdl(ev)
    
    pred = out['slots']['predicted'][:, :rollout, :, app_dim:]
    tgt = out['slots']['target'][:, :rollout, :, app_dim:]
    dm = out['depth_mask'][:, :rollout].unsqueeze(-1).float()
    bnd = compute_boundary_mask(out['slots']['target'][:, :rollout], out['depth_mask'][:, :rollout]).unsqueeze(-1)
    
    pos_mse = ((pred[...,:2]-tgt[...,:2])**2*dm).sum()/(dm.sum()*2+1e-8)
    depth_mse = ((pred[...,2:3]-tgt[...,2:3])**2*dm).sum()/(dm.sum()+1e-8)
    depth_mse_bnd = ((pred[...,2:3]-tgt[...,2:3])**2*bnd).sum()/(bnd.sum()+1e-8)
    pos_mse_bnd = ((pred[...,:2]-tgt[...,:2])**2*bnd).sum()/(bnd.sum()*2+1e-8)
    
    print(f"  {name:20s}: pos={pos_mse.item():.6f} depth={depth_mse.item():.6f} | bnd_pos={pos_mse_bnd.item():.6f} bnd_depth={depth_mse_bnd.item():.6f}")
