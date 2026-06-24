"""
更直观的可视化: 选几个有代表性大跳变的案例
每行: 视频帧 | 所有slot depth | 目标slot pos+depth | 目标slot在正常帧和跳变帧的mask对比
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

# 收集
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
        if total_move < 0.15:
            continue
        
        dd = np.abs(np.diff(depth))
        max_jump = dd.max()
        max_jump_idx = dd.argmax()
        
        fr = np.arange(16)
        r2_px = 1 - np.sum((pos_x-np.polyval(np.polyfit(fr,pos_x,1),fr))**2)/(np.sum((pos_x-pos_x.mean())**2)+1e-8) if pos_x.std()>1e-6 else 1.0
        r2_py = 1 - np.sum((pos_y-np.polyval(np.polyfit(fr,pos_y,1),fr))**2)/(np.sum((pos_y-pos_y.mean())**2)+1e-8) if pos_y.std()>1e-6 else 1.0
        
        cases.append({
            'sample': si, 'slot': s,
            'max_jump': max_jump, 'max_jump_idx': max_jump_idx,
            'total_move': total_move, 'pos_r2': min(r2_px, r2_py),
            'pos_x': pos_x, 'pos_y': pos_y, 'depth': depth,
        })

# 选6个: 不同位置的大跳变
# 3个 burnin 内跳变 + 3个 rollout 内跳变
burnin_jumps = sorted([c for c in cases if c['max_jump_idx'] < 5 and c['max_jump'] > 0.01 and c['pos_r2'] > 0.8],
                      key=lambda c: -c['max_jump'])
rollout_jumps = sorted([c for c in cases if c['max_jump_idx'] >= 5 and c['max_jump'] > 0.01 and c['pos_r2'] > 0.8],
                       key=lambda c: -c['max_jump'])

selected = []
seen = set()
for c in burnin_jumps[:3]:
    if c['sample'] not in seen:
        seen.add(c['sample'])
        selected.append(('burnin_jump', c))
for c in rollout_jumps[:3]:
    if c['sample'] not in seen:
        seen.add(c['sample'])
        selected.append(('rollout_jump', c))

print(f"Selected {len(selected)} cases:")
for label, c in selected:
    print(f"  [{label}] s{c['sample']}_sl{c['slot']}: pos_R²={c['pos_r2']:.3f} jump={c['max_jump']:.4f}@t={c['max_jump_idx']}→{c['max_jump_idx']+1}")

# 可视化
n = len(selected)
fig, axes = plt.subplots(n, 6, figsize=(36, 4.5*n))
if n == 1:
    axes = axes.reshape(1, -1)
fig.suptitle('Depth Encoding Artifacts: pos is linear but depth has large frame-to-frame jumps\n(Red overlay = target slot mask, green dashed = burnin|rollout)', fontsize=12, y=1.03)

for j, (label, c) in enumerate(selected):
    si, sl = c['sample'], c['slot']
    sample = ds[si]
    frames_vid = sample['video']
    frames = frames_vid.unsqueeze(0).cuda()
    
    with torch.no_grad():
        out = model(frames)
    all_slots = torch.cat([out['slots']['corrected'], out['slots']['target']], dim=1)[0]
    
    fr = np.arange(16)
    ji = c['max_jump_idx']
    
    # Col 0: 原始视频 (4帧)
    ax = axes[j, 0]
    imgs = []
    for t in [0, 5, 10, 15]:
        img = frames_vid[t].permute(1, 2, 0).numpy()
        imgs.append(np.clip((img + 1) / 2, 0, 1))
    ax.imshow(np.concatenate(imgs, axis=1))
    ax.set_title(f's{si} [{label}]', fontsize=9)
    ax.axis('off')
    
    # Col 1: 所有slot depth
    ax = axes[j, 1]
    N = all_slots.shape[1]
    for s in range(N):
        d = all_slots[:, s, app_dim+2].cpu().numpy()
        if np.mean(d) >= depth_max:
            continue
        lw = 2.5 if s == sl else 0.7
        al = 1.0 if s == sl else 0.3
        ax.plot(fr, d, '-o', markersize=2, linewidth=lw, alpha=al, label=f'sl{s}')
    ax.axvline(x=5.5, color='green', linestyle='--', alpha=0.5)
    ax.set_title(f'All depths (sl{sl} bold)', fontsize=9)
    ax.legend(fontsize=5, ncol=3)
    ax.grid(True, alpha=0.3)
    
    # Col 2: pos + depth
    ax = axes[j, 2]
    ax.plot(fr, c['pos_x'], 'r-o', markersize=3, label='pos_x')
    ax.plot(fr, c['pos_y'], 'b-o', markersize=3, label='pos_y')
    ax.axvline(x=5.5, color='green', linestyle='--', alpha=0.5)
    ax.set_ylabel('pos')
    ax2 = ax.twinx()
    ax2.plot(fr, c['depth'], 'g-s', markersize=4, linewidth=2, label='depth')
    d = c['depth']
    for k in range(1, 15):
        if d[k] > d[k-1] and d[k] > d[k+1]:
            ax2.plot(k, d[k], 'rv', markersize=10)
        elif d[k] < d[k-1] and d[k] < d[k+1]:
            ax2.plot(k, d[k], 'r^', markersize=10)
    ax2.axvspan(ji, ji+1, alpha=0.2, color='red')
    ax2.annotate(f'Δ={c["max_jump"]:.3f}', xy=(ji+0.5, max(d[ji], d[ji+1])),
                xytext=(ji+2, max(d[ji], d[ji+1])+0.01),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=8, color='red')
    ax2.set_ylabel('depth', color='green')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'sl{sl} pos+depth (pos_R²={c["pos_r2"]:.2f})', fontsize=9)
    
    # Col 3-5: 3帧的 slot mask 对比 (跳变前, 跳变后, + 一个正常帧)
    # 选3个时间点: 跳变前, 跳变后, 跳变后2帧
    t_points = [max(ji-1, 0), ji+1, min(ji+3, 15)]
    for k, t_idx in enumerate(t_points):
        ax = axes[j, 3+k]
        if t_idx >= 16:
            ax.text(0.5, 0.5, 'N/A', ha='center')
            continue
        
        slots_frame = all_slots[t_idx].unsqueeze(0).cuda()
        with torch.no_grad():
            recon_img, alphas = model.decoder(slots_frame, return_alphas=True)
        
        bg = frames_vid[t_idx].permute(1, 2, 0).numpy()
        bg = np.clip((bg + 1) / 2, 0, 1)
        
        slot_alpha = alphas[0, sl, 0].cpu().numpy()
        overlay = bg.copy()
        mask = slot_alpha > 0.2
        overlay[mask] = overlay[mask] * 0.4 + np.array([1, 0.3, 0]) * 0.6
        
        ax.imshow(overlay)
        depth_val = d[t_idx]
        is_jump_frame = (t_idx == ji or t_idx == ji+1)
        title_color = 'red' if is_jump_frame else 'black'
        ax.set_title(f't={t_idx} d={depth_val:.4f}', fontsize=9, color=title_color)
        ax.axis('off')

plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/depth_erratic_v3.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: pos_depth_debug/depth_erratic_v3.png")
