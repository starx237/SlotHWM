"""
实验3: 长时程depth跟踪 (50帧)
选5个典型物体，绘制depth变化折线图
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
model_state = model.state_dict()
loaded = {}
for mk in model_state:
    mk_c = mk.replace('_orig_mod.', '')
    for ck in ckpt['model']:
        ck_c = ck.replace('_orig_mod.', '')
        if ck_c == mk_c and ckpt['model'][ck].shape == model_state[mk].shape:
            loaded[mk] = ckpt['model'][ck]
            break
model.load_state_dict(loaded, strict=False)
model.eval()

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=50, stride=1, subsample=1)
loader = ds.get_dataloader(batch_size=1, shuffle=False, num_workers=0)

boundary_threshold = 0.7

# 收集长时程轨迹
all_tracks = []
for i in range(30):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = model(frames)
    
    corrected = out['slots']['corrected']
    target = out['slots']['target']
    all_slots = torch.cat([corrected, target], dim=1)[0]  # (50, N, D)
    
    N = all_slots.shape[1]
    for s in range(N):
        depth_vals = all_slots[:, s, app_dim+2].cpu().numpy()
        pos_x = all_slots[:, s, app_dim].cpu().numpy()
        pos_y = all_slots[:, s, app_dim+1].cpu().numpy()
        
        if np.mean(depth_vals[:10]) >= depth_max:
            continue
        
        # 找严格在画面内的长段
        in_frame = (np.abs(pos_x) < boundary_threshold) & (np.abs(pos_y) < boundary_threshold) & (depth_vals < depth_max)
        segments = []
        start = None
        for t in range(len(in_frame)):
            if in_frame[t]:
                if start is None: start = t
            else:
                if start is not None:
                    segments.append((start, t-1))
                    start = None
        if start is not None:
            segments.append((start, len(in_frame)-1))
        
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start + 1
            move = np.sqrt((pos_x[seg_end]-pos_x[seg_start])**2 + (pos_y[seg_end]-pos_y[seg_start])**2)
            if seg_len >= 20 and move > 0.3:
                all_tracks.append({
                    'sample': i,
                    'slot': s,
                    'seg_start': seg_start,
                    'seg_end': seg_end,
                    'seg_len': seg_len,
                    'pos_x': pos_x[seg_start:seg_end+1],
                    'pos_y': pos_y[seg_start:seg_end+1],
                    'depth': depth_vals[seg_start:seg_end+1],
                    'total_move': move,
                })
    
    if i % 10 == 0:
        print(f"Scanned {i} samples, found {len(all_tracks)} long tracks so far", flush=True)

print(f"\nTotal long tracks (>=20 frames, move>0.3): {len(all_tracks)}")

# 选5个不同样本的代表
all_tracks.sort(key=lambda t: t['total_move'], reverse=True)
selected = []
used_samples = set()
for t in all_tracks:
    if t['sample'] not in used_samples and len(selected) < 5:
        selected.append(t)
        used_samples.add(t['sample'])

print(f"Selected {len(selected)} representative tracks:")
for j, t in enumerate(selected):
    print(f"  [{j}] s{t['sample']}_sl{t['slot']} frames={t['seg_start']}-{t['seg_end']}({t['seg_len']}) move={t['total_move']:.3f}")

# 绘制折线图
fig, axes = plt.subplots(5, 3, figsize=(18, 16))
fig.suptitle('Long-term Depth Tracking (50 frames, strict boundary filtering)', fontsize=14)

for j, t in enumerate(selected):
    frames_range = np.arange(t['seg_len'])
    depth = t['depth']
    pos_x = t['pos_x']
    pos_y = t['pos_y']
    
    # Depth
    axes[j, 0].plot(frames_range, depth, 'b-o', markersize=3, label='depth')
    
    # 线性拟合
    coef1 = np.polyfit(frames_range, depth, 1)
    fit1 = np.polyval(coef1, frames_range)
    r2_1 = 1 - np.sum((depth-fit1)**2)/(np.sum((depth-depth.mean())**2)+1e-8)
    axes[j, 0].plot(frames_range, fit1, 'r--', label=f'linear R²={r2_1:.3f}')
    
    # 二次拟合
    coef2 = np.polyfit(frames_range, depth, 2)
    fit2 = np.polyval(coef2, frames_range)
    r2_2 = 1 - np.sum((depth-fit2)**2)/(np.sum((depth-depth.mean())**2)+1e-8)
    axes[j, 0].plot(frames_range, fit2, 'g--', label=f'quad R²={r2_2:.3f}')
    
    # inv-depth 拟合
    inv_depth = 1.0/(depth+1e-6)
    coef_inv = np.polyfit(frames_range, inv_depth, 1)
    fit_inv = 1.0/(np.polyval(coef_inv, frames_range)+1e-6)
    r2_inv = 1 - np.sum((depth-fit_inv)**2)/(np.sum((depth-depth.mean())**2)+1e-8)
    axes[j, 0].plot(frames_range, fit_inv, 'm--', label=f'inv-depth R²={r2_inv:.3f}')
    
    axes[j, 0].set_title(f's{t["sample"]}_sl{t["slot"]} ({t["seg_len"]}f, move={t["total_move"]:.2f})')
    axes[j, 0].set_ylabel('Depth')
    axes[j, 0].legend(fontsize=8)
    axes[j, 0].grid(True, alpha=0.3)
    
    # Pos X
    axes[j, 1].plot(frames_range, pos_x, 'r-o', markersize=3)
    axes[j, 1].set_title(f'pos_x (s{t["sample"]}_sl{t["slot"]})')
    axes[j, 1].set_ylabel('pos_x')
    axes[j, 1].grid(True, alpha=0.3)
    
    # Pos Y
    axes[j, 2].plot(frames_range, pos_y, 'g-o', markersize=3)
    axes[j, 2].set_title(f'pos_y (s{t["sample"]}_sl{t["slot"]})')
    axes[j, 2].set_ylabel('pos_y')
    axes[j, 2].grid(True, alpha=0.3)

axes[-1, 0].set_xlabel('Frame')
axes[-1, 1].set_xlabel('Frame')
axes[-1, 2].set_xlabel('Frame')
plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/long_term_depth.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: pos_depth_debug/long_term_depth.png")

# 额外: depth delta 分析
fig2, axes2 = plt.subplots(5, 2, figsize=(14, 16))
fig2.suptitle('Depth Delta Analysis (strict boundary)', fontsize=14)

for j, t in enumerate(selected):
    depth = t['depth']
    pos_x = t['pos_x']
    frames_range = np.arange(t['seg_len'])
    
    # depth delta
    d_delta = np.diff(depth)
    axes2[j, 0].plot(frames_range[1:], d_delta, 'b-o', markersize=3)
    axes2[j, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes2[j, 0].set_title(f'Δdepth (s{t["sample"]}_sl{t["slot"]})')
    axes2[j, 0].set_ylabel('Δdepth')
    axes2[j, 0].grid(True, alpha=0.3)
    
    # depth vs pos_x scatter
    axes2[j, 1].scatter(pos_x, depth, c=frames_range, cmap='viridis', s=20)
    axes2[j, 1].set_title(f'depth vs pos_x (s{t["sample"]}_sl{t["slot"]})')
    axes2[j, 1].set_xlabel('pos_x')
    axes2[j, 1].set_ylabel('depth')
    axes2[j, 1].grid(True, alpha=0.3)

axes2[-1, 0].set_xlabel('Frame')
axes2[-1, 1].set_xlabel('pos_x')
plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/depth_delta_long.png', dpi=150, bbox_inches='tight')
print("Saved: pos_depth_debug/depth_delta_long.png")

# 统计摘要
print("\n=== Summary ===")
print(f"Total long tracks found: {len(all_tracks)}")
if all_tracks:
    lens = [t['seg_len'] for t in all_tracks]
    moves = [t['total_move'] for t in all_tracks]
    print(f"Track length: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.1f}")
    print(f"Movement: min={min(moves):.2f}, max={max(moves):.2f}, mean={np.mean(moves):.2f}")
    
    # 对所有长时程轨迹做线性度分析
    lin_r2s = []
    quad_r2s = []
    for t in all_tracks:
        depth = t['depth']
        if depth.std() < 1e-6: continue
        frames_range = np.arange(len(depth))
        c1 = np.polyfit(frames_range, depth, 1)
        r2_1 = 1 - np.sum((depth-np.polyval(c1,frames_range))**2)/(np.sum((depth-depth.mean())**2)+1e-8)
        c2 = np.polyfit(frames_range, depth, 2)
        r2_2 = 1 - np.sum((depth-np.polyval(c2,frames_range))**2)/(np.sum((depth-depth.mean())**2)+1e-8)
        lin_r2s.append(r2_1)
        quad_r2s.append(r2_2)
    
    print(f"\nAll long tracks R²(lin): mean={np.mean(lin_r2s):.4f}, median={np.median(lin_r2s):.4f}")
    print(f"All long tracks R²(quad): mean={np.mean(quad_r2s):.4f}, median={np.median(quad_r2s):.4f}")
