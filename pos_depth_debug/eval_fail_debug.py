#!/usr/bin/env python3
"""
失败版：加详细debug输出定位卡死位置
"""
import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
print("[DEBUG] 1. imports starting", flush=True)

import torch, numpy as np
from types import SimpleNamespace
import yaml

print("[DEBUG] 2. basic imports done", flush=True)

sys.path.insert(0, '.')
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset
from torch.utils.data import DataLoader

print("[DEBUG] 3. project imports done", flush=True)

with open('config/pretrain_phase2.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg_dict['continue_pretrain'] = True
cfg_dict['workdir'] = '/tmp/fail_eval'
cfg = SimpleNamespace(**cfg_dict)

print("[DEBUG] 4. config loaded", flush=True)

model = SlotDynamicsModel(cfg)
print("[DEBUG] 5. model created", flush=True)

ckpt = torch.load('experiments/phase2_depth_spread/checkpoints/best.pt', map_location='cpu')
print("[DEBUG] 6. checkpoint loaded", flush=True)

model.load_state_dict(ckpt['model'], strict=False)
print("[DEBUG] 7. state_dict loaded", flush=True)

model.eval().cuda()
print("[DEBUG] 8. model on cuda + eval mode", flush=True)

app_dim = model.appearance_dim
ds = OBJ3DDataset(data_path='data/OBJ3D', num_frames=16, subsample=2)
print("[DEBUG] 9. dataset created", flush=True)

dl = ds.get_dataloader(batch_size=1, shuffle=False, num_workers=0)
print("[DEBUG] 10. dataloader created", flush=True)

all_d, all_s, all_c = [], [], []
with torch.no_grad():
    for i, batch in enumerate(dl):
        print(f"[DEBUG] 11.{i} batch loaded, keys={list(batch.keys())}", flush=True)
        if i >= 3000: break
        frames = batch["video"].cuda()
        print(f"[DEBUG] 11.{i} frames shape={frames.shape} on cuda", flush=True)
        out = model(frames)
        torch.cuda.synchronize()
        print(f"[DEBUG] 11.{i} model forward + sync done", flush=True)
        slots_t = out['slots']['corrected'][0, 0]
        torch.cuda.synchronize()
        print(f"[DEBUG] 11.{i} slots_t done, shape={slots_t.shape}", flush=True)
        alpha = out['alpha']
        torch.cuda.synchronize()
        print(f"[DEBUG] 11.{i} alpha done, shape={alpha.shape}", flush=True)
        a = alpha[0].squeeze()
        N, H, W = a.shape
        print(f"[DEBUG] 11.{i} a squeezed, N={N} H={H} W={W}", flush=True)
        torch.cuda.synchronize()
        print(f"[DEBUG] 11.{i} sync done", flush=True)
        gy, gx = torch.meshgrid(torch.linspace(-1,1,H,device='cuda'), torch.linspace(-1,1,W,device='cuda'), indexing='ij')
        print(f"[DEBUG] 11.{i} meshgrid done", flush=True)
        a_sum = a.sum(dim=[-2,-1], keepdim=True) + 1e-8
        a_norm = a / a_sum
        cx = (a_norm * gx.unsqueeze(0)).sum(dim=[-2,-1])
        cy = (a_norm * gy.unsqueeze(0)).sum(dim=[-2,-1])
        print(f"[DEBUG] 11.{i} cx/cy done", flush=True)
        sp = torch.sqrt((a_norm * ((gx.unsqueeze(0)-cx.unsqueeze(-1).unsqueeze(-1))**2 +
                                    (gy.unsqueeze(0)-cy.unsqueeze(-1).unsqueeze(-1))**2)).sum(dim=[-2,-1]))
        print(f"[DEBUG] 11.{i} sp done", flush=True)
        depth = slots_t[:, app_dim+2]
        a_max = a.amax(dim=[-2,-1]); cov = a.sum(dim=[-2,-1])
        print(f"[DEBUG] 11.{i} depth/a_max/cov done", flush=True)
        fg = (cov > 20) & (cov < 1500) & (sp > 0.01) & (a_max > 0.3) & (depth < 0.5)
        for s in range(N):
            if fg[s] and depth[s] > 0.04:
                all_d.append(depth[s].item())
                all_s.append(sp[s].item())
                all_c.append(cov[s].item() / (H*W))
        if i % 500 == 0:
            print(f"  {i}/3000 done", flush=True)
        if i >= 2:
            print("[DEBUG] first 3 iters done, breaking early", flush=True)
            break

print("[DEBUG] 12. loop done", flush=True)
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

print(f"\n=== FAIL VERSION (3 samples only) ===")
print(f"n_fg={len(dm)}, depth mean={dm.mean():.4f}")
print(f"spread/depth median={np.median(sm/dm):.4f}")
print(f"polyfit: spread y={coef_s[0]:.4f}x+{coef_s[1]:.4f} R2={r2_s_poly:.4f}")
print(f"prior:   R2(spread)={r2_s_prior:.4f}")
