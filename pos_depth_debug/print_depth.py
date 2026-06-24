"""
直接打印 depth 数值，看变化模式
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

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=1, shuffle=False, num_workers=0)

# 打印前5个样本的完整depth数据
for i in range(5):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = model(frames)
    all_slots = torch.cat([out['slots']['corrected'], out['slots']['target']], dim=1)[0]
    
    print(f"\n=== Sample {i} ===")
    N = all_slots.shape[1]
    for s in range(N):
        depth = all_slots[:, s, app_dim+2].cpu().numpy()
        pos_x = all_slots[:, s, app_dim].cpu().numpy()
        pos_y = all_slots[:, s, app_dim+1].cpu().numpy()
        
        if np.mean(depth) >= depth_max:
            print(f"  slot {s}: BACKGROUND (depth={depth.mean():.4f})")
            continue
        
        dd = np.diff(depth)
        n_ext = sum(1 for k in range(1,len(dd)) if dd[k]*dd[k-1] < 0)
        
        print(f"  slot {s}: depth={[f'{d:.4f}' for d in depth]}")
        print(f"          delta=[{', '.join(f'{d:+.5f}' for d in dd)}]")
        print(f"          pos_x range=[{pos_x.min():.3f},{pos_x.max():.3f}] pos_y=[{pos_y.min():.3f},{pos_y.max():.3f}] extrema={n_ext}")
