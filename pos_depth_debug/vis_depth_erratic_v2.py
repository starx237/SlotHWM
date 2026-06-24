"""
可视化: pos平稳但depth震荡的物体
用 decoder return_alphas 获取每个 slot 的 mask
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

# 找案例
cases = []
for si in range(50):
    sample = ds[si]
    frames = sample['video'].unsqueeze(0).cuda()
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
        if total_move < 0.1:
            continue
        
        dd = np.abs(np.diff(depth))
        max_jump = dd.max()
        max_jump_idx = dd.argmax()
        
        # pos 线性度
        fr = np.arange(16)
        r2_px = 1 - np.sum((pos_x-np.polyval(np.polyfit(fr,pos_x,1),fr))**2)/(np.sum((pos_x-pos_x.mean())**2)+1e-8) if pos_x.std()>1e-6 else 1.0
        r2_py = 1 - np.sum((pos_y-np.polyval(np.polyfit(fr,pos_y,1),fr))**2)/(np.sum((pos_y-pos_y.mean())**2)+1e-8) if pos_y.std()>1e-6 else 1.0
        
        cases.append({
            'sample': si, 'slot': s,
            'max_jump': max_jump, 'max_jump_idx': max_jump_idx,
            'total_move': total_move, 'pos_r2': min(r2_px, r2_py),
            'pos_x': pos_x, 'pos_y': pos_y, 'depth': depth,
        })

# 选5个: pos R2 最好但 depth 有大跳变的
cases_sorted = sorted(cases, key=lambda c: (-c['pos_r2'], -c['max_jump']))
seen = set()
selected = []
for c in cases_sorted:
    key = c['sample']
    if key not in seen and c['max_jump'] > 0.005:
        seen.add(key)
        selected.append(c)
    if len(selected) >= 5:
        break

print(f"Selected {len(selected)} cases:")
for c in selected:
    print(f"  s{c['sample']}_sl{c['slot']}: pos_R²={c['pos_r2']:.3f} max_jump={c['max_jump']:.4f}@t={c['max_jump_idx']}→{c['max_jump_idx']+1} move={c['total_move']:.2f}")

# 可视化
n = len(selected)
fig, axes = plt.subplots(n, 5, figsize=(30, 4*n))
if n == 1:
    axes = axes.reshape(1, -1)
fig.suptitle('Pos Linear but Depth Erratic — ISA Depth Encoding Inconsistency\n(Green=burnin, Red markers=depth local extrema, Arrows=max frame jump)', fontsize=12, y=1.03)

for j, c in enumerate(selected):
    si, sl = c['sample'], c['slot']
    sample = ds[si]
    frames_vid = sample['video']
    frames = frames_vid.unsqueeze(0).cuda()
    
    with torch.no_grad():
        out = model(frames)
    all_slots = torch.cat([out['slots']['corrected'], out['slots']['target']], dim=1)[0]
    
    fr = np.arange(16)
    
    # Col 0: 原始视频 (4帧)
    ax = axes[j, 0]
    imgs = []
    for t in [0, 5, 10, 15]:
        img = frames_vid[t].permute(1, 2, 0).numpy()
        imgs.append(np.clip((img + 1) / 2, 0, 1))
    ax.imshow(np.concatenate(imgs, axis=1))
    ax.set_title(f'Sample {si}', fontsize=9)
    ax.axis('off')
    
    # Col 1: 所有slot的depth
    ax = axes[j, 1]
    N = all_slots.shape[1]
    for s in range(N):
        d = all_slots[:, s, app_dim+2].cpu().numpy()
        if np.mean(d) >= depth_max:
            continue
        lw = 2.5 if s == sl else 0.7
        al = 1.0 if s == sl else 0.35
        ax.plot(fr, d, '-o', markersize=2, linewidth=lw, alpha=al, label=f'sl{s}')
    ax.axvline(x=5.5, color='green', linestyle='--', alpha=0.5)
    ax.set_title(f'All slots depth (sl{sl} bold)', fontsize=9)
    ax.legend(fontsize=5, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('depth')
    
    # Col 2: pos + depth (双轴)
    ax = axes[j, 2]
    ln1 = ax.plot(fr, c['pos_x'], 'r-o', markersize=3, label='pos_x')
    ln2 = ax.plot(fr, c['pos_y'], 'b-o', markersize=3, label='pos_y')
    ax.axvline(x=5.5, color='green', linestyle='--', alpha=0.5)
    ax.set_ylabel('pos')
    ax2 = ax.twinx()
    ln3 = ax2.plot(fr, c['depth'], 'g-s', markersize=4, linewidth=2, label='depth')
    # 极值点
    d = c['depth']
    for k in range(1, 15):
        if d[k] > d[k-1] and d[k] > d[k+1]:
            ax2.plot(k, d[k], 'rv', markersize=10)
        elif d[k] < d[k-1] and d[k] < d[k+1]:
            ax2.plot(k, d[k], 'r^', markersize=10)
    # 最大跳变
    ji = c['max_jump_idx']
    ax2.annotate(f'Δ={c["max_jump"]:.3f}', xy=(ji+0.5, d[ji]),
                xytext=(ji+2, d[ji]+0.01),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=8, color='red')
    ax2.set_ylabel('depth', color='green')
    lns = ln1 + ln2 + ln3
    ax.legend(lns, [l.get_label() for l in lns], fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'sl{sl} pos+depth (R²={c["pos_r2"]:.2f})', fontsize=9)
    
    # Col 3-4: Slot mask + 场景 (2帧: 正常帧 + 跳变帧)
    for k, t_idx in enumerate([5, c['max_jump_idx']]):
        ax = axes[j, 3+k]
        if t_idx >= 16:
            ax.text(0.5, 0.5, 'N/A', ha='center')
            continue
        
        slots_frame = all_slots[t_idx].unsqueeze(0).cuda()
        with torch.no_grad():
            recon_img, alphas = model.decoder(slots_frame, return_alphas=True)
        
        # 背景=原始帧
        bg = frames_vid[t_idx].permute(1, 2, 0).numpy()
        bg = np.clip((bg + 1) / 2, 0, 1)
        
        # 叠加目标 slot 的 alpha mask (红框)
        slot_alpha = alphas[0, sl, 0].cpu().numpy()
        
        # 在原始帧上用红色半透明标注 slot 区域
        overlay = bg.copy()
        mask = slot_alpha > 0.3
        overlay[mask] = overlay[mask] * 0.5 + np.array([1, 0, 0]) * 0.5
        
        ax.imshow(overlay)
        ax.set_title(f't={t_idx} sl{sl} mask', fontsize=9)
        ax.axis('off')

plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/depth_erratic_visual.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: pos_depth_debug/depth_erratic_visual.png")

# 统计
all_jumps = [c['max_jump'] for c in cases]
if all_jumps:
    print(f"\nDepth max jump stats (all moving objects, n={len(all_jumps)}):")
    print(f"  mean={np.mean(all_jumps):.4f} median={np.median(all_jumps):.4f}")
    print(f"  90th={np.percentile(all_jumps,90):.4f} 95th={np.percentile(all_jumps,95):.4f}")
    
    jump_indices = [c['max_jump_idx'] for c in cases]
    print(f"\n  Jump position:")
    for pos in range(15):
        count = sum(1 for idx in jump_indices if idx == pos)
        if count > 0:
            print(f"    t={pos}→{pos+1}: {count}")
