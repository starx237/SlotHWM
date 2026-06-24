"""
深入分析: depth 为什么全部非单调？
1. 看在画面内的物体，depth 是否单调
2. 逐帧分析 depth 变化模式
3. 可视化典型案例（含 slot 分解）
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
bnd_thresh = 0.75

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

# 收集所有物体数据
all_data = []
for i in range(200):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = model(frames)
    corrected = out['slots']['corrected']
    target = out['slots']['target']
    all_slots = torch.cat([corrected, target], dim=1)[0]
    
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
        
        # 找完全在画面内的连续段
        in_frame = (np.abs(pos_x) < bnd_thresh) & (np.abs(pos_y) < bnd_thresh) & (depth < depth_max)
        segs = []
        start = None
        for t in range(16):
            if in_frame[t]:
                if start is None: start = t
            else:
                if start is not None:
                    segs.append((start, t-1))
                    start = None
        if start is not None:
            segs.append((start, 15))
        
        for seg_s, seg_e in segs:
            seg_len = seg_e - seg_s + 1
            if seg_len < 6:
                continue
            d_seg = depth[seg_s:seg_e+1]
            dd = np.diff(d_seg)
            n_ext = sum(1 for k in range(1,len(dd)) if dd[k]*dd[k-1] < 0)
            is_mono = np.all(dd >= -1e-5) or np.all(dd <= 1e-5)
            
            fr = np.arange(seg_len)
            px = pos_x[seg_s:seg_e+1]
            py = pos_y[seg_s:seg_e+1]
            r2_px = 1 - np.sum((px-np.polyval(np.polyfit(fr,px,1),fr))**2)/(np.sum((px-px.mean())**2)+1e-8) if px.std()>1e-6 else 1.0
            r2_py = 1 - np.sum((py-np.polyval(np.polyfit(fr,py,1),fr))**2)/(np.sum((py-py.mean())**2)+1e-8) if py.std()>1e-6 else 1.0
            r2_d = 1 - np.sum((d_seg-np.polyval(np.polyfit(fr,d_seg,1),fr))**2)/(np.sum((d_seg-d_seg.mean())**2)+1e-8) if d_seg.std()>1e-6 else 1.0
            
            all_data.append({
                'sample': i, 'slot': s,
                'seg_start': seg_s, 'seg_end': seg_e, 'seg_len': seg_len,
                'pos_r2': min(r2_px, r2_py), 'depth_r2': r2_d,
                'n_extrema': n_ext, 'is_monotone': is_mono,
                'pos_x': px, 'pos_y': py, 'depth': d_seg,
                'total_move': total_move,
            })

print(f"Total in-frame segments (>=6 frames): {len(all_data)}")

mono = [d for d in all_data if d['is_monotone']]
uni = [d for d in all_data if d['n_extrema'] <= 1]
complex_d = [d for d in all_data if d['n_extrema'] >= 2]
print(f"  depth monotone: {len(mono)}")
print(f"  depth unimodal: {len(uni)}")
print(f"  depth complex: {len(complex_d)}")

# 逐个统计极值点数分布
from collections import Counter
ext_counts = Counter(d['n_extrema'] for d in all_data)
print(f"  extrema distribution: {dict(sorted(ext_counts.items()))}")

# 找 "pos 线性 + depth 震荡" 的典型案例
pos_good = [d for d in all_data if d['pos_r2'] > 0.85]
print(f"\npos R²>0.85 in-frame: {len(pos_good)}")
mono2 = [d for d in pos_good if d['is_monotone']]
print(f"  depth monotone: {len(mono2)}")

# 按 depth 非单调程度排序
pos_good_sorted = sorted(pos_good, key=lambda d: (-d['n_extrema'], d['depth_r2']))
print(f"\nTop cases (pos linear, depth complex, in-frame):")
for d in pos_good_sorted[:10]:
    print(f"  s{d['sample']}_sl{d['slot']} seg=[{d['seg_start']},{d['seg_end']}]({d['seg_len']}f): pos_R²={d['pos_r2']:.3f} depth_R²={d['depth_r2']:.3f} extrema={d['n_extrema']}")

# 也看 depth 最单调的
pos_good_mono = sorted(pos_good, key=lambda d: (d['n_extrema'], -d['depth_r2']))
print(f"\nBest cases (pos linear, depth simple):")
for d in pos_good_mono[:10]:
    print(f"  s{d['sample']}_sl{d['slot']} seg=[{d['seg_start']},{d['seg_end']}]({d['seg_len']}f): pos_R²={d['pos_r2']:.3f} depth_R²={d['depth_r2']:.3f} extrema={d['n_extrema']} mono={d['is_monotone']}")

# ========== 可视化 ==========
# 选6个案例: 3个depth复杂 + 3个depth简单
cases_complex = pos_good_sorted[:3]
cases_simple = pos_good_mono[:3]
# 确保不重复
seen = set()
selected = []
for d in cases_complex + cases_simple:
    key = (d['sample'], d['slot'])
    if key not in seen:
        seen.add(key)
        selected.append(d)
    if len(selected) >= 6:
        break

print(f"\nVisualizing {len(selected)} cases:")
for d in selected:
    print(f"  s{d['sample']}_sl{d['slot']} seg=[{d['seg_start']},{d['seg_end']}] pos_R²={d['pos_r2']:.3f} depth_R²={d['depth_r2']:.3f} extrema={d['n_extrema']}")

fig, axes = plt.subplots(len(selected), 4, figsize=(22, 3.5*len(selected)))
if len(selected) == 1:
    axes = axes.reshape(1, -1)
fig.suptitle('Pos Linear vs Depth Pattern (in-frame segments only)', fontsize=14, y=1.01)

for j, d in enumerate(selected):
    si, sl = d['sample'], d['slot']
    seg_s, seg_e = d['seg_start'], d['seg_end']
    fr = np.arange(d['seg_len'])
    
    # 重新获取完整的16帧数据
    torch.manual_seed(0)
    sample = ds[si]
    frames = sample['video'].unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(frames)
    all_slots = torch.cat([out['slots']['corrected'], out['slots']['target']], dim=1)[0]
    
    full_depth = all_slots[:, sl, app_dim+2].cpu().numpy()
    full_px = all_slots[:, sl, app_dim].cpu().numpy()
    full_py = all_slots[:, sl, app_dim+1].cpu().numpy()
    
    # 1) 完整16帧的 depth (标注 in-frame 段)
    ax = axes[j, 0]
    fr_full = np.arange(16)
    ax.plot(fr_full, full_depth, 'g-o', markersize=3)
    # 标注 in-frame 段
    ax.axvspan(seg_s, seg_e, alpha=0.15, color='green', label='in-frame')
    ax.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)
    # 标注边界外的点
    for t in range(16):
        if abs(full_px[t]) >= bnd_thresh or abs(full_py[t]) >= bnd_thresh:
            ax.plot(t, full_depth[t], 'rx', markersize=10, markeredgewidth=2)
    label = 'complex' if d['n_extrema'] >= 2 else 'simple'
    ax.set_title(f's{si}_sl{sl} depth ({label}, ext={d["n_extrema"]}, R²={d["depth_r2"]:.2f})', fontsize=9)
    ax.set_ylabel('depth')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # 2) in-frame 段的 pos + depth
    ax = axes[j, 1]
    ax.plot(fr, d['pos_x'], 'r-o', markersize=3, label='pos_x')
    ax.plot(fr, d['pos_y'], 'b-o', markersize=3, label='pos_y')
    ax2 = ax.twinx()
    ax2.plot(fr, d['depth'], 'g-s', markersize=3, alpha=0.7, label='depth')
    ax2.set_ylabel('depth', color='green')
    ax.set_title(f'in-frame pos+depth (pos_R²={d["pos_r2"]:.2f})', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')
    ax2.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 3) Slot alpha mask 可视化 - burnin 最后一帧 (t=5) 和 rollout 中间帧 (t=10)
    for k, t_idx in enumerate([5, 10]):
        ax = axes[j, 2+k]
        # 获取该帧的所有 slot alpha
        with torch.no_grad():
            # 用 decoder 解码单个 slot
            s_data = all_slots[t_idx].unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, D)
            recon = model.decoder(s_data)  # (1, 1, 4, H, W)
            rgb = recon[0, 0, :3].cpu().numpy().transpose(1, 2, 0)
            alpha = recon[0, 0, 3].cpu().numpy()
            img = rgb * alpha[:,:,np.newaxis]
            img = np.clip((img + 1) / 2, 0, 1)  # 假设 [-1,1] 范围
            ax.imshow(img)
        # 标注目标 slot
        ax.set_title(f't={t_idx} slot{sl} alpha', fontsize=9)
        ax.axis('off')

plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/pos_linear_depth_patterns.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: pos_depth_debug/pos_linear_depth_patterns.png")

# 额外: 全局统计
print("\n=== Global stats ===")
print(f"{'n_extrema':>10s} {'count':>6s} {'avg_depth_R2':>12s}")
for ne in sorted(ext_counts.keys()):
    subset = [d for d in all_data if d['n_extrema'] == ne]
    avg_r2 = np.mean([d['depth_r2'] for d in subset]) if subset else 0
    print(f"{ne:10d} {len(subset):6d} {avg_r2:12.4f}")
