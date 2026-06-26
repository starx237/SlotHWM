#!/usr/bin/env python3
"""
失败版：R²(spread)≈-4, spread/depth≈1.13
关键区别：直接 model.load_state_dict + DataLoader(batch_size=1, shuffle=False)
没有 setup_cuda，没有 Trainer，没有 ds.get_dataloader
"""
import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
import torch, numpy as np
from types import SimpleNamespace
import yaml

sys.path.insert(0, '.')
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset
from torch.utils.data import DataLoader

with open('config/pretrain_phase2.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg_dict['continue_pretrain'] = True
cfg_dict['workdir'] = '/tmp/fail_eval'
cfg = SimpleNamespace(**cfg_dict)

model = SlotDynamicsModel(cfg)
ckpt = torch.load('experiments/phase2_depth_spread/checkpoints/best.pt', map_location='cpu')
model.load_state_dict(ckpt['model'], strict=False)
model.eval().cuda()

app_dim = model.appearance_dim
ds = OBJ3DDataset(data_path='data/obj3d', num_frames=1, subsample=2, stride=4)
dl = ds.get_dataloader(batch_size=1, shuffle=False, num_workers=0)

all_d, all_s, all_c = [], [], []
with torch.no_grad():
    for i, batch in enumerate(dl):
        if i >= 3000: break
        frames = batch["video"].cuda()
        out = model(frames)
        slots_t = out['slots']['corrected'][0, 0]
        alpha = out['alpha']
        a = alpha[0].squeeze()
        N, H, W = a.shape
        gy, gx = torch.meshgrid(torch.linspace(-1,1,H,device='cuda'), torch.linspace(-1,1,W,device='cuda'), indexing='ij')
        a_sum = a.sum(dim=[-2,-1], keepdim=True) + 1e-8
        a_norm = a / a_sum
        cx = (a_norm * gx.unsqueeze(0)).sum(dim=[-2,-1])
        cy = (a_norm * gy.unsqueeze(0)).sum(dim=[-2,-1])
        sp = torch.sqrt((a_norm * ((gx.unsqueeze(0)-cx.unsqueeze(-1).unsqueeze(-1))**2 +
                                    (gy.unsqueeze(0)-cy.unsqueeze(-1).unsqueeze(-1))**2)).sum(dim=[-2,-1]))
        depth = slots_t[:, app_dim+2]
        a_max = a.amax(dim=[-2,-1]); cov = a.sum(dim=[-2,-1])
        fg = (cov > 20) & (cov < 1500) & (sp > 0.01) & (a_max > 0.3) & (depth < 0.5)
        for s in range(N):
            if fg[s] and depth[s] > 0.04:
                all_d.append(depth[s].item())
                all_s.append(sp[s].item())
                all_c.append(cov[s].item() / (H*W))
        if i % 500 == 0:
            print(f"  {i}/3000 done", flush=True)

dm = np.array(all_d); sm = np.array(all_s); cm = np.array(all_c)
mask = dm > 0.04
dm, sm, cm = dm[mask], sm[mask], cm[mask]
d2m = dm**2

a_val = model.depth_spread_a.item()
c_val = model.depth_spread_c.item()
b_val = model.depth_spread_b.item()
d_val = model.depth_spread_d.item()
r2_s_prior = 1 - np.sum((sm - (a_val*dm+c_val))**2) / np.sum((sm - sm.mean())**2)
r2_c_prior = 1 - np.sum((cm - (b_val*d2m+d_val))**2) / np.sum((cm - cm.mean())**2)
coef_s = np.polyfit(dm, sm, 1)
r2_s_poly = 1 - np.sum((sm - np.polyval(coef_s, dm))**2) / np.sum((sm - sm.mean())**2)

print(f"\n=== FAIL VERSION ===")
print(f"n_fg={len(dm)}, depth mean={dm.mean():.4f}")
print(f"spread/depth median={np.median(sm/dm):.4f}")
print(f"polyfit: spread y={coef_s[0]:.4f}x+{coef_s[1]:.4f} R2={r2_s_poly:.4f}")
print(f"prior:   R2(spread)={r2_s_prior:.4f}")
