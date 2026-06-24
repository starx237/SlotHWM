"""
实验3v2: 长时程跟踪 - 放宽边界阈值和运动阈值
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')
import warnings; warnings.filterwarnings('ignore')
import torch
import numpy as np
import yaml, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset

with open('config/obj3d.yaml') as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))

app_dim = cfg.appearance_dim
depth_max = getattr(cfg, 'depth_max', 0.30)

model = SlotDynamicsModel(cfg).cuda()
ckpt = torch.load(cfg.pretrained_path, map_location='cpu')
sd = model.state_dict()
ld = {}
for mk in sd:
    mc = mk.replace('_orig_mod.','')
    for ck in ckpt['model']:
        cc = ck.replace('_orig_mod.','')
        if cc==mc and ckpt['model'][ck].shape==sd[mk].shape:
            ld[mk]=ckpt['model'][ck]; break
model.load_state_dict(ld, strict=False)
model.eval()

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=50, stride=1, subsample=1)

# 先看看 stride=1 时 pos 的数值范围
loader = ds.get_dataloader(batch_size=1, shuffle=False, num_workers=0)
batch = next(iter(loader))
frames = batch['video'].cuda()
with torch.no_grad():
    out = model(frames)
corrected = out['slots']['corrected']
target = out['slots']['target']
all_slots = torch.cat([corrected, target], dim=1)[0]  # (50, N, D)

N = all_slots.shape[1]
print(f"50-frame slots shape: {all_slots.shape}")
for s in range(N):
    d = all_slots[:, s, app_dim+2].cpu().numpy()
    px = all_slots[:, s, app_dim].cpu().numpy()
    py = all_slots[:, s, app_dim+1].cpu().numpy()
    if np.mean(d[:5]) >= depth_max: continue
    print(f"  slot {s}: pos_x=[{px.min():.3f},{px.max():.3f}] pos_y=[{py.min():.3f},{py.max():.3f}] depth=[{d.min():.4f},{d.max():.4f}]")
