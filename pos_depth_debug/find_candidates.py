"""
查找 pos 线性但 depth 非单调/震荡的物体
放宽筛选条件，先统计数据分布
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

all_data = []

for i in range(200):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = model(frames)
    
    corrected = out['slots']['corrected']
    target = out['slots']['target']
    all_slots = torch.cat([corrected, target], dim=1)[0]  # (16, N, D)
    
    N = all_slots.shape[1]
    for s in range(N):
        depth = all_slots[:, s, app_dim+2].cpu().numpy()
        pos_x = all_slots[:, s, app_dim].cpu().numpy()
        pos_y = all_slots[:, s, app_dim+1].cpu().numpy()
        
        if np.mean(depth) >= depth_max:
            continue
        
        total_move = np.sqrt((pos_x[-1]-pos_x[0])**2 + (pos_y[-1]-pos_y[0])**2)
        if total_move < 0.1:
            continue
        
        fr = np.arange(16)
        r2_px = 1 - np.sum((pos_x-np.polyval(np.polyfit(fr,pos_x,1),fr))**2)/(np.sum((pos_x-pos_x.mean())**2)+1e-8) if pos_x.std()>1e-6 else 1.0
        r2_py = 1 - np.sum((pos_y-np.polyval(np.polyfit(fr,pos_y,1),fr))**2)/(np.sum((pos_y-pos_y.mean())**2)+1e-8) if pos_y.std()>1e-6 else 1.0
        pos_r2 = min(r2_px, r2_py)
        
        # depth 极值点数
        d_delta = np.diff(depth)
        n_extrema = 0
        for k in range(1, len(d_delta)):
            if d_delta[k] * d_delta[k-1] < 0:
                n_extrema += 1
        
        # depth 是否单调
        is_monotone = np.all(d_delta >= -1e-5) or np.all(d_delta <= 1e-5)
        is_unimodal = n_extrema <= 1
        
        # depth R²
        r2_d = 1 - np.sum((depth-np.polyval(np.polyfit(fr,depth,1),fr))**2)/(np.sum((depth-depth.mean())**2)+1e-8) if depth.std()>1e-6 else 1.0
        
        # 边界信息
        bnd_thresh = 0.75
        ever_near_boundary = (np.abs(pos_x).max() >= bnd_thresh) or (np.abs(pos_y).max() >= bnd_thresh)
        first_bnd_frame = -1
        for t in range(16):
            if abs(pos_x[t]) >= bnd_thresh or abs(pos_y[t]) >= bnd_thresh:
                first_bnd_frame = t
                break
        
        all_data.append({
            'sample': i, 'slot': s,
            'pos_r2': pos_r2, 'depth_r2': r2_d,
            'n_extrema': n_extrema,
            'is_monotone': is_monotone, 'is_unimodal': is_unimodal,
            'total_move': total_move,
            'near_boundary': ever_near_boundary,
            'first_bnd_frame': first_bnd_frame,
            'pos_x': pos_x, 'pos_y': pos_y, 'depth': depth,
        })

print(f"Total moving FG objects: {len(all_data)}")

# 统计
pos_good = [d for d in all_data if d['pos_r2'] > 0.85]
print(f"pos R²>0.85: {len(pos_good)}")

depth_monotone = [d for d in pos_good if d['is_monotone']]
depth_unimodal = [d for d in pos_good if d['is_unimodal']]
depth_complex = [d for d in pos_good if not d['is_unimodal']]
print(f"  depth monotone: {len(depth_monotone)}")
print(f"  depth unimodal (incl monotone): {len(depth_unimodal)}")
print(f"  depth complex (>1 extrema): {len(depth_complex)}")

# 放宽: pos R²>0.8 且 depth 有≥2个极值点
relaxed = [d for d in all_data if d['pos_r2'] > 0.8 and d['n_extrema'] >= 2]
print(f"\nRelaxed (pos R²>0.8, depth extrema>=2): {len(relaxed)}")

# 再放宽: 只要 depth 不单调不单峰
very_relaxed = [d for d in all_data if d['pos_r2'] > 0.7 and not d['is_unimodal']]
print(f"Very relaxed (pos R²>0.7, not unimodal): {len(very_relaxed)}")

# 极端宽松
extreme = [d for d in all_data if d['pos_r2'] > 0.7 and d['n_extrema'] >= 1]
print(f"Extreme (pos R²>0.7, extrema>=1): {len(extreme)}")

# 检查边界效应
print("\n--- Boundary effect ---")
for d in all_data[:50]:
    if d['pos_r2'] > 0.8 and d['n_extrema'] >= 1:
        bnd_info = f"near_bnd@t={d['first_bnd_frame']}" if d['near_boundary'] else "in_frame"
        print(f"  s{d['sample']}_sl{d['slot']}: pos_R²={d['pos_r2']:.3f} depth_R²={d['depth_r2']:.3f} extrema={d['n_extrema']} {bnd_info}")

# 按depth复杂度排序，看极端案例
all_data_sorted = sorted(all_data, key=lambda d: (-d['n_extrema'], d['depth_r2']))
print("\n--- Top by extrema count ---")
for d in all_data_sorted[:20]:
    bnd_info = f"bnd@t={d['first_bnd_frame']}" if d['near_boundary'] else "in_frame"
    print(f"  s{d['sample']}_sl{d['slot']}: pos_R²={d['pos_r2']:.3f} depth_R²={d['depth_r2']:.3f} extrema={d['n_extrema']} monotone={d['is_monotone']} {bnd_info} move={d['total_move']:.2f}")
