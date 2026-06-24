"""
v3: 放宽边界阈值到0.85，看哪些物体 pos 线性但 depth 震荡
直接可视化完整16帧，标注边界和slot
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
bnd_thresh = 0.85

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
    all_slots = torch.cat([out['slots']['corrected'], out['slots']['target']], dim=1)[0]
    
    N = all_slots.shape[1]
    for s in range(N):
        depth = all_slots[:, s, app_dim+2].cpu().numpy()
        pos_x = all_slots[:, s, app_dim].cpu().numpy()
        pos_y = all_slots[:, s, app_dim+1].cpu().numpy()
        
        if np.mean(depth) >= depth_max:
            continue
        
        total_move = np.sqrt((pos_x[-1]-pos_x[0])**2 + (pos_y[-1]-pos_y[0])**2)
        if total_move < 0.15:
            continue
        
        fr = np.arange(16)
        r2_px = 1 - np.sum((pos_x-np.polyval(np.polyfit(fr,pos_x,1),fr))**2)/(np.sum((pos_x-pos_x.mean())**2)+1e-8) if pos_x.std()>1e-6 else 1.0
        r2_py = 1 - np.sum((pos_y-np.polyval(np.polyfit(fr,pos_y,1),fr))**2)/(np.sum((pos_y-pos_y.mean())**2)+1e-8) if pos_y.std()>1e-6 else 1.0
        pos_r2 = min(r2_px, r2_py)
        
        # depth 极值点
        dd = np.diff(depth)
        n_ext = sum(1 for k in range(1,len(dd)) if dd[k]*dd[k-1] < 0)
        is_mono = np.all(dd >= -1e-5) or np.all(dd <= 1e-5)
        r2_d = 1 - np.sum((depth-np.polyval(np.polyfit(fr,depth,1),fr))**2)/(np.sum((depth-depth.mean())**2)+1e-8) if depth.std()>1e-6 else 1.0
        
        # 是否始终在画面内
        ever_out = (np.abs(pos_x).max() >= bnd_thresh) or (np.abs(pos_y).max() >= bnd_thresh)
        first_out = -1
        for t in range(16):
            if abs(pos_x[t]) >= bnd_thresh or abs(pos_y[t]) >= bnd_thresh:
                first_out = t; break
        
        all_data.append({
            'sample': i, 'slot': s,
            'pos_r2': pos_r2, 'depth_r2': r2_d,
            'n_extrema': n_ext, 'is_monotone': is_mono,
            'total_move': total_move,
            'ever_out': ever_out, 'first_out': first_out,
            'pos_x': pos_x, 'pos_y': pos_y, 'depth': depth,
        })

# 统计
print(f"Total moving FG objects (bnd=0.85): {len(all_data)}")
mono = [d for d in all_data if d['is_monotone']]
print(f"depth monotone: {len(mono)}")
from collections import Counter
ext_dist = Counter(d['n_extrema'] for d in all_data)
print(f"extrema distribution: {dict(sorted(ext_dist.items()))}")

# 区分始终在画面内 vs 曾出画面
in_frame = [d for d in all_data if not d['ever_out']]
out_frame = [d for d in all_data if d['ever_out']]
print(f"\nAlways in frame: {len(in_frame)}")
mono_in = [d for d in in_frame if d['is_monotone']]
print(f"  depth monotone: {len(mono_in)}")
ext_in = Counter(d['n_extrema'] for d in in_frame)
print(f"  extrema distribution: {dict(sorted(ext_in.items()))}")
avg_r2_in = np.mean([d['depth_r2'] for d in in_frame]) if in_frame else 0
print(f"  avg depth R²: {avg_r2_in:.4f}")

print(f"\nEver near boundary: {len(out_frame)}")
mono_out = [d for d in out_frame if d['is_monotone']]
print(f"  depth monotone: {len(mono_out)}")
ext_out = Counter(d['n_extrema'] for d in out_frame)
print(f"  extrema distribution: {dict(sorted(ext_out.items()))}")
avg_r2_out = np.mean([d['depth_r2'] for d in out_frame]) if out_frame else 0
print(f"  avg depth R²: {avg_r2_out:.4f}")

# 选典型案例可视化
# 优先选始终在画面内、pos线性好、depth最复杂的
in_frame_sorted = sorted(in_frame, key=lambda d: (-d['n_extrema'], d['depth_r2']))
out_frame_sorted = sorted(out_frame, key=lambda d: (-d['n_extrema'], d['depth_r2']))

# 也选 depth 最简单的做对比
in_frame_best = sorted(in_frame, key=lambda d: (d['n_extrema'], -d['depth_r2']))

selected = []
used = set()
# 3个 in-frame 复杂
for d in in_frame_sorted:
    key = (d['sample'], d['slot'])
    if key not in used:
        used.add(key)
        selected.append(('in_complex', d))
    if len([s for s in selected if s[0]=='in_complex']) >= 3:
        break

# 3个 in-frame 简单
for d in in_frame_best:
    key = (d['sample'], d['slot'])
    if key not in used:
        used.add(key)
        selected.append(('in_simple', d))
    if len([s for s in selected if s[0]=='in_simple']) >= 3:
        break

# 3个 out-frame 复杂
for d in out_frame_sorted:
    key = (d['sample'], d['slot'])
    if key not in used:
        used.add(key)
        selected.append(('out_complex', d))
    if len([s for s in selected if s[0]=='out_complex']) >= 3:
        break

print(f"\nSelected {len(selected)} cases for visualization:")
for label, d in selected:
    print(f"  [{label}] s{d['sample']}_sl{d['slot']}: pos_R²={d['pos_r2']:.3f} depth_R²={d['depth_r2']:.3f} ext={d['n_extrema']} first_out={d['first_out']}")

# 可视化
n = len(selected)
fig, axes = plt.subplots(n, 5, figsize=(28, 3.5*n))
if n == 1:
    axes = axes.reshape(1, -1)
fig.suptitle('Pos Linear + Depth Pattern Analysis', fontsize=14, y=1.01)

for j, (label, d) in enumerate(selected):
    si, sl = d['sample'], d['slot']
    fr = np.arange(16)
    
    # 获取完整数据
    sample = ds[si]
    frames_vid = sample['video']  # (16, 3, 64, 64)
    frames = frames_vid.unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(frames)
    all_slots = torch.cat([out['slots']['corrected'], out['slots']['target']], dim=1)[0]
    
    # Col 0: 原始视频帧 (t=0, 5, 10, 15 拼接)
    ax = axes[j, 0]
    imgs = []
    for t in [0, 5, 10, 15]:
        img = frames_vid[t].permute(1, 2, 0).numpy()
        img = np.clip((img + 1) / 2, 0, 1)
        imgs.append(img)
    composite = np.concatenate(imgs, axis=1)
    ax.imshow(composite)
    ax.set_title(f's{si} frames 0,5,10,15', fontsize=8)
    ax.axis('off')
    
    # Col 1: pos 轨迹
    ax = axes[j, 1]
    ax.plot(fr, d['pos_x'], 'r-o', markersize=3, label='pos_x')
    ax.plot(fr, d['pos_y'], 'b-o', markersize=3, label='pos_y')
    ax.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)
    if d['first_out'] >= 0:
        ax.axvline(x=d['first_out'], color='red', linestyle=':', alpha=0.5, label=f'boundary@t={d["first_out"]}')
    ax.set_title(f'pos (R²={d["pos_r2"]:.2f})', fontsize=9)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    
    # Col 2: depth 曲线
    ax = axes[j, 2]
    ax.plot(fr, d['depth'], 'g-o', markersize=3, label='depth')
    # 标注极值点
    for k in range(1, 15):
        if (d['depth'][k] > d['depth'][k-1] and d['depth'][k] > d['depth'][k+1]):
            ax.plot(k, d['depth'][k], 'rv', markersize=8)  # 局部极大
        elif (d['depth'][k] < d['depth'][k-1] and d['depth'][k] < d['depth'][k+1]):
            ax.plot(k, d['depth'][k], 'r^', markersize=8)  # 局部极小
    ax.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)
    if d['first_out'] >= 0:
        ax.axvline(x=d['first_out'], color='red', linestyle=':', alpha=0.5)
    tag = 'COMPLEX' if d['n_extrema'] >= 2 else 'SIMPLE'
    ax.set_title(f'depth {tag} (R²={d["depth_r2"]:.2f}, ext={d["n_extrema"]})', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Col 3-4: Slot 重建 (t=5 和 t=10)
    for k, t_idx in enumerate([5, 10]):
        ax = axes[j, 3+k]
        s_data = all_slots[t_idx].unsqueeze(0).unsqueeze(0).cuda()
        with torch.no_grad():
            recon = model.decoder(s_data)
        rgb = recon[0, 0, :3].cpu().numpy().transpose(1, 2, 0)
        alpha = recon[0, 0, 3].cpu().numpy()
        # 尝试不同的范围
        img = np.clip((rgb + 1) / 2, 0, 1) * alpha[:,:,np.newaxis]
        ax.imshow(img)
        ax.set_title(f'sl{sl} t={t_idx}', fontsize=8)
        ax.axis('off')

plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/pos_linear_depth_patterns_v3.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: pos_depth_debug/pos_linear_depth_patterns_v3.png")
