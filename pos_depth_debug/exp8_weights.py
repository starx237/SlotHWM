"""
实验8: 找最优 pos/depth 权重比 (bnd_mask+decay)
之前结果: E(bnd+decay) bnd_depth=0.000143 但 bnd_pos=0.002374
试不同的 pos_weight 来平衡
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

# 测试不同 pos_weight
configs = [
    ('pos10_d1', 10.0, 1.0),   # 重 pos
    ('pos5_d1', 5.0, 1.0),
    ('pos3_d1', 3.0, 1.0),
    ('pos1_d1', 1.0, 1.0),     # 等权
    ('pos1_d3', 1.0, 3.0),     # 重 depth
    ('pos1_d5', 1.0, 5.0),
]

results = {}
for cfg_name, pos_w, depth_w in configs:
    print(f"\n--- {cfg_name} (pos_w={pos_w}, depth_w={depth_w}) ---")
    m = make()
    opt = torch.optim.Adam(filter(lambda p:p.requires_grad, m.parameters()), lr=1e-4)
    
    for step in range(400):
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
    
    # 评估
    m.eval()
    torch.manual_seed(99)
    ev = next(iter(ds.get_dataloader(batch_size=2,shuffle=False,num_workers=0)))['video'].cuda()
    with torch.no_grad():
        out = m(ev)
    pred = out['slots']['predicted'][:, :rollout, :, app_dim:]
    tgt = out['slots']['target'][:, :rollout, :, app_dim:]
    bnd = compute_boundary_mask(out['slots']['target'][:, :rollout], out['depth_mask'][:, :rollout]).unsqueeze(-1)
    
    bnd_pos = ((pred[...,:2]-tgt[...,:2])**2*bnd).sum()/(bnd.sum()*2+1e-8)
    bnd_depth = ((pred[...,2:3]-tgt[...,2:3])**2*bnd).sum()/(bnd.sum()+1e-8)
    
    results[cfg_name] = (bnd_pos.item(), bnd_depth.item())
    print(f"  bnd_pos={bnd_pos.item():.6f} bnd_depth={bnd_depth.item():.6f}")

print("\n=== Summary ===")
print(f"{'Config':15s} {'bnd_pos':>10s} {'bnd_depth':>10s}")
print("-"*40)
# 也加基线
print(f"{'Base':15s} {'0.0216':>10s} {'0.0003':>10s}")
for k, (p, d) in results.items():
    print(f"{k:15s} {p:10.6f} {d:10.6f}")
