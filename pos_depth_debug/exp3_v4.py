"""
实验3v4: 长时程跟踪 - 用模型forward获取slots，找连续稳定段
不用逐帧编码，直接用模型输出，寻找物体始终在画面内的连续段
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

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=1, shuffle=True, num_workers=0)

boundary_threshold = 0.75
all_segments = []

for i in range(100):
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
        
        if np.mean(depth[:3]) >= depth_max:
            continue
        
        move = np.sqrt((pos_x[-1]-pos_x[0])**2 + (pos_y[-1]-pos_y[0])**2)
        
        # 找严格在画面内的连续段
        in_frame = (np.abs(pos_x) < boundary_threshold) & (np.abs(pos_y) < boundary_threshold) & (depth < depth_max)
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
        
        for seg_s, seg_e in segments:
            seg_len = seg_e - seg_s + 1
            seg_move = np.sqrt((pos_x[seg_e]-pos_x[seg_s])**2 + (pos_y[seg_e]-pos_y[seg_s])**2)
            if seg_len >= 8 and seg_move > 0.1:
                all_segments.append({
                    'sample': i, 'slot': s,
                    'seg_start': seg_s, 'seg_end': seg_e, 'seg_len': seg_len,
                    'pos_x': pos_x[seg_s:seg_e+1].copy(),
                    'pos_y': pos_y[seg_s:seg_e+1].copy(),
                    'depth': depth[seg_s:seg_e+1].copy(),
                    'total_move': seg_move,
                })

print(f"Total in-frame segments: {len(all_segments)}")
all_segments.sort(key=lambda t: t['total_move'], reverse=True)

# 选5个不同样本的代表
selected = []
used = set()
for t in all_segments:
    if t['sample'] not in used and len(selected) < 5:
        selected.append(t)
        used.add(t['sample'])

print(f"Selected {len(selected)} representative tracks:")
for j, t in enumerate(selected):
    print(f"  [{j}] s{t['sample']}_sl{t['slot']} seg={t['seg_len']}f move={t['total_move']:.3f}")

# ============ 深入分析depth规律 ============
print("\n=== Depth pattern analysis (all segments) ===")
lin_r2s, quad_r2s, inv_r2s, cubic_r2s = [], [], [], []
depth_deltas = []
depth_ranges = []

for t in all_segments:
    depth = t['depth']
    if depth.std() < 1e-6: continue
    fr = np.arange(len(depth))
    
    c1 = np.polyfit(fr, depth, 1)
    r2_1 = 1 - np.sum((depth-np.polyval(c1,fr))**2)/(np.sum((depth-depth.mean())**2)+1e-8)
    
    c2 = np.polyfit(fr, depth, 2)
    r2_2 = 1 - np.sum((depth-np.polyval(c2,fr))**2)/(np.sum((depth-depth.mean())**2)+1e-8)
    
    inv_d = 1.0/(depth+1e-6)
    ci = np.polyfit(fr, inv_d, 1)
    fi = 1.0/(np.polyval(ci,fr)+1e-6)
    r2_i = 1 - np.sum((depth-fi)**2)/(np.sum((depth-depth.mean())**2)+1e-8)
    
    c3 = np.polyfit(fr, depth, 3)
    r2_3 = 1 - np.sum((depth-np.polyval(c3,fr))**2)/(np.sum((depth-depth.mean())**2)+1e-8)
    
    lin_r2s.append(r2_1)
    quad_r2s.append(r2_2)
    inv_r2s.append(r2_i)
    cubic_r2s.append(r2_3)
    
    depth_deltas.append(np.diff(depth))
    depth_ranges.append(depth.max() - depth.min())

print(f"Segments analyzed: {len(lin_r2s)}")
print(f"R²(linear):  mean={np.mean(lin_r2s):.4f} median={np.median(lin_r2s):.4f}")
print(f"R²(quad):    mean={np.mean(quad_r2s):.4f} median={np.median(quad_r2s):.4f}")
print(f"R²(inv-d):   mean={np.mean(inv_r2s):.4f} median={np.median(inv_r2s):.4f}")
print(f"R²(cubic):   mean={np.mean(cubic_r2s):.4f} median={np.median(cubic_r2s):.4f}")

# depth delta 统计
all_deltas = np.concatenate(depth_deltas)
print(f"\nDepth delta stats:")
print(f"  mean={all_deltas.mean():.6f} std={all_deltas.std():.6f}")
print(f"  |delta| mean={np.abs(all_deltas).mean():.6f} median={np.median(np.abs(all_deltas)):.6f}")
print(f"  max={all_deltas.max():.6f} min={all_deltas.min():.6f}")
print(f"  sign changes: {np.mean(np.diff(np.sign(all_deltas))!=0)*100:.1f}%")

print(f"\nDepth range (max-min) per segment: mean={np.mean(depth_ranges):.4f} median={np.median(depth_ranges):.4f}")

# 检查: depth变化是否与pos变化相关
print("\n=== Depth-pos correlation ===")
corr_dp = []
for t in all_segments:
    if t['depth'].std() < 1e-6: continue
    dp = np.diff(t['depth'])
    dx = np.diff(t['pos_x'])
    dy = np.diff(t['pos_y'])
    dist = np.sqrt(dx**2 + dy**2)
    if dist.std() < 1e-8: continue
    c = np.corrcoef(np.abs(dp), dist)[0,1]
    corr_dp.append(c)
if corr_dp:
    print(f"corr(|Δdepth|, |Δpos|): mean={np.mean(corr_dp):.4f} median={np.median(corr_dp):.4f}")

# ============ 绘图 ============
fig, axes = plt.subplots(len(selected), 3, figsize=(18, 4*len(selected)))
if len(selected) == 1:
    axes = axes.reshape(1, -1)
fig.suptitle('Depth Tracking (strict boundary, stride=4)', fontsize=14)

for j, t in enumerate(selected):
    fr = np.arange(t['seg_len'])
    depth = t['depth']
    px = t['pos_x']
    py = t['pos_y']
    
    ax = axes[j, 0]
    ax.plot(fr, depth, 'b-o', markersize=3, label='depth')
    if depth.std() > 1e-6:
        c1 = np.polyfit(fr, depth, 1); f1 = np.polyval(c1, fr)
        r2_1 = 1-np.sum((depth-f1)**2)/(np.sum((depth-depth.mean())**2)+1e-8)
        ax.plot(fr, f1, 'r--', label=f'lin R²={r2_1:.3f}')
        
        c2 = np.polyfit(fr, depth, 2); f2 = np.polyval(c2, fr)
        r2_2 = 1-np.sum((depth-f2)**2)/(np.sum((depth-depth.mean())**2)+1e-8)
        ax.plot(fr, f2, 'g--', label=f'quad R²={r2_2:.3f}')
        
        inv_d = 1.0/(depth+1e-6)
        ci = np.polyfit(fr, inv_d, 1); fi = 1.0/(np.polyval(ci,fr)+1e-6)
        r2_i = 1-np.sum((depth-fi)**2)/(np.sum((depth-depth.mean())**2)+1e-8)
        ax.plot(fr, fi, 'm--', label=f'inv R²={r2_i:.3f}')
    
    ax.set_title(f's{t["sample"]}_sl{t["slot"]} ({t["seg_len"]}f, move={t["total_move"]:.2f})')
    ax.set_ylabel('Depth'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    axes[j, 1].plot(fr, px, 'r-o', markersize=3)
    axes[j, 1].set_title('pos_x'); axes[j, 1].grid(True, alpha=0.3)
    
    axes[j, 2].plot(fr, py, 'g-o', markersize=3)
    axes[j, 2].set_title('pos_y'); axes[j, 2].grid(True, alpha=0.3)

axes[-1, 0].set_xlabel('Frame'); axes[-1, 1].set_xlabel('Frame'); axes[-1, 2].set_xlabel('Frame')
plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/long_term_depth_v4.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: long_term_depth_v4.png")

# Delta图
fig2, axes2 = plt.subplots(len(selected), 2, figsize=(14, 4*len(selected)))
if len(selected) == 1:
    axes2 = axes2.reshape(1, -1)
fig2.suptitle('Depth Delta & Depth-pos relationship', fontsize=14)

for j, t in enumerate(selected):
    depth = t['depth']; px = t['pos_x']
    fr = np.arange(t['seg_len'])
    
    dd = np.diff(depth)
    axes2[j, 0].plot(fr[1:], dd, 'b-o', markersize=3)
    axes2[j, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes2[j, 0].set_title(f'Δdepth (s{t["sample"]}_sl{t["slot"]})')
    axes2[j, 0].grid(True, alpha=0.3)
    
    sc = axes2[j, 1].scatter(px, depth, c=fr, cmap='viridis', s=30)
    axes2[j, 1].set_xlabel('pos_x'); axes2[j, 1].set_ylabel('depth')
    axes2[j, 1].set_title(f'depth vs pos_x')
    plt.colorbar(sc, ax=axes2[j, 1], label='frame')
    axes2[j, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/depth_delta_v4.png', dpi=150, bbox_inches='tight')
print("Saved: depth_delta_v4.png")

# 保存数据
np.save('/autodl-fs/data/SlotHWM/pos_depth_debug/segments_v4.npy', all_segments, allow_pickle=True)
np.save('/autodl-fs/data/SlotHWM/pos_depth_debug/selected_v4.npy', selected, allow_pickle=True)
