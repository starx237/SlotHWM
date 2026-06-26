#!/usr/bin/env python3
import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
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
cfg_dict['workdir'] = '/tmp/simple_eval2'
cfg = SimpleNamespace(**cfg_dict)

model = SlotDynamicsModel(cfg)
ckpt = torch.load('experiments/phase2_depth_spread/checkpoints/step_30000.pt', map_location='cpu')
model.load_state_dict(ckpt['model'], strict=False)
model.eval().cuda()

app_dim = model.appearance_dim
ds = OBJ3DDataset(data_path='data/OBJ3D', num_frames=7, subsample=2)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

all_d, all_s, all_c = [], [], []

with torch.no_grad():
    for i, batch in enumerate(dl):
        if i >= 3000: break
        frames = batch["video"].cuda()
        out = model(frames)
        slots_t = out['slots']['corrected'][0, 0]
        alpha = out['alpha']
        a = alpha[0]
        while a.dim() > 3:
            a = a.squeeze(0)
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
coef_s = np.polyfit(dm, sm, 1)
r2_poly = 1 - np.sum((sm - np.polyval(coef_s, dm))**2) / np.sum((sm - sm.mean())**2)
r2_prior = 1 - np.sum((sm - (1.4385*dm+0.003452))**2) / np.sum((sm - sm.mean())**2)

print(f"\n3000 samples: n_fg={len(dm)}")
print(f"  depth mean={dm.mean():.4f}, spread/depth median={np.median(sm/dm):.4f}")
print(f"  polyfit: y={coef_s[0]:.4f}x+{coef_s[1]:.4f}, R2={r2_poly:.4f}")
print(f"  prior R2={r2_prior:.4f}")
