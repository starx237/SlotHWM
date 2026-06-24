"""
实验4+5: 严格 loss mask 训练 + 单独学 depth
A: 原始 loss (pos+depth)
B: 严格 boundary mask loss (pos+depth)
C: 严格 boundary mask + 只学 depth
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

torch.manual_seed(0)
d1 = next(iter(ds.get_dataloader(batch_size=2,shuffle=False,num_workers=0)))['video'].cuda()
torch.manual_seed(1)
d2 = next(iter(ds.get_dataloader(batch_size=2,shuffle=False,num_workers=0)))['video'].cuda()
torch.manual_seed(2)
d3 = next(iter(ds.get_dataloader(batch_size=2,shuffle=False,num_workers=0)))['video'].cuda()
batches = [d1, d2, d3]

def compute_boundary_mask(slots, depth_mask):
    B, T, N, D = slots.shape
    pos_x = slots[:, :, :, app_dim]
    pos_y = slots[:, :, :, app_dim+1]
    depth = slots[:, :, :, app_dim+2]
    boundary_mask = (pos_x.abs() < boundary_threshold) & (pos_y.abs() < boundary_threshold) & (depth < depth_max)
    return (depth_mask & boundary_mask).float()

models = {
    'A(orig)': make(),
    'B(bnd_mask)': make(),
    'C(depth_bnd)': make(),
    'D(depth_only)': make(),
}
opts = {k: torch.optim.Adam(filter(lambda p:p.requires_grad, m.parameters()), lr=1e-4) for k, m in models.items()}

# 训练
for step in range(300):
    frames = batches[step % len(batches)]
    
    for name, mdl in models.items():
        # 一次 train forward，用 detach 的 target
        mdl.train()
        out = mdl(frames)
        
        target = out['slots']['target'][:, :rollout].detach()
        pred = out['slots']['predicted'][:, :rollout]
        dm = out['depth_mask'][:, :rollout].detach()
        
        bnd_mask = compute_boundary_mask(target, dm)
        m0 = dm.unsqueeze(-1).float()
        bnd = bnd_mask.unsqueeze(-1)
        
        pred_dyn = pred[:, :, :, app_dim:]
        tgt_dyn = target[:, :, :, app_dim:]
        
        if name == 'A(orig)':
            loss = ((pred_dyn - tgt_dyn)**2 * m0).sum() / (m0.sum()*3+1e-8)
        elif name == 'B(bnd_mask)':
            loss = ((pred_dyn - tgt_dyn)**2 * bnd).sum() / (bnd.sum()*3+1e-8)
        elif name == 'C(depth_bnd)':
            loss = ((pred_dyn[...,2:3] - tgt_dyn[...,2:3])**2 * bnd).sum() / (bnd.sum()+1e-8)
        elif name == 'D(depth_only)':
            loss = ((pred_dyn[...,2:3] - tgt_dyn[...,2:3])**2 * m0).sum() / (m0.sum()+1e-8)
        
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
    
    # 逐帧
    for t in [0, 4, 9]:
        m_t = dm[:,t]
        b_t = bnd[:,t]
        pm = ((pred[:,t,:,:2]-tgt[:,t,:,:2])**2*m_t).sum()/(m_t.sum()*2+1e-8)
        dm_t = ((pred[:,t,:,2:3]-tgt[:,t,:,2:3])**2*m_t).sum()/(m_t.sum()+1e-8)
        print(f"    frame {t}: pos={pm.item():.6f} depth={dm_t.item():.6f}")
