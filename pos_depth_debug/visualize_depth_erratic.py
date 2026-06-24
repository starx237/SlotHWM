"""
可视化: pos平稳但depth震荡的物体
关键发现: ISA depth编码本身就有帧间不一致性
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

# 固定选择几个样本做可视化
# 选有典型depth跳变的样本
torch.manual_seed(0)
batch0 = next(iter(ds.get_dataloader(batch_size=1,shuffle=False,num_workers=0)))

cases = []
for si in range(20):
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
        if total_move < 0.05:  # 几乎静止的物体
            continue
        
        # 找depth最大跳变
        dd = np.abs(np.diff(depth))
        max_jump_idx = np.argmax(dd)
        max_jump = dd[max_jump_idx]
        
        if max_jump > 0.01:  # 有显著depth跳变
            cases.append({
                'sample': si, 'slot': s,
                'max_jump': max_jump, 'max_jump_idx': max_jump_idx,
                'total_move': total_move,
                'pos_x': pos_x, 'pos_y': pos_y, 'depth': depth,
            })

# 按 max_jump 排序
cases.sort(key=lambda c: -c['max_jump'])

# 去重：确保不同样本
seen = set()
selected = []
for c in cases:
    key = c['sample']
    if key not in seen:
        seen.add(key)
        selected.append(c)
    if len(selected) >= 5:
        break

print(f"Selected {len(selected)} cases:")
for c in selected:
    print(f"  s{c['sample']}_sl{c['slot']}: max_jump={c['max_jump']:.4f}@t={c['max_jump_idx']}→{c['max_jump_idx']+1} move={c['total_move']:.2f}")

# 可视化
n = len(selected)
fig, axes = plt.subplots(n, 4, figsize=(24, 4*n))
if n == 1:
    axes = axes.reshape(1, -1)
fig.suptitle('Objects with Linear Pos but Erratic Depth\n(Red markers = local extrema, green band = burnin, red dashed = max depth jump)', fontsize=12, y=1.03)

for j, c in enumerate(selected):
    si, sl = c['sample'], c['slot']
    sample = ds[si]
    frames_vid = sample['video']  # (16, 3, 64, 64)
    frames = frames_vid.unsqueeze(0).cuda()
    
    with torch.no_grad():
        out = model(frames)
    all_slots = torch.cat([out['slots']['corrected'], out['slots']['target']], dim=1)[0]
    
    fr = np.arange(16)
    
    # Col 0: 视频帧 (4帧拼接)
    ax = axes[j, 0]
    imgs = []
    for t in [0, 5, 10, 15]:
        img = frames_vid[t].permute(1, 2, 0).numpy()
        img = np.clip((img + 1) / 2, 0, 1)
        imgs.append(img)
    composite = np.concatenate(imgs, axis=1)
    ax.imshow(composite)
    ax.set_title(f'Sample {si}: frames 0,5,10,15', fontsize=9)
    ax.axis('off')
    
    # Col 1: 所有 slot 的 depth
    ax = axes[j, 1]
    N = all_slots.shape[1]
    for s in range(N):
        d = all_slots[:, s, app_dim+2].cpu().numpy()
        if np.mean(d) >= depth_max:
            continue
        lw = 2.5 if s == sl else 0.8
        alpha = 1.0 if s == sl else 0.4
        ax.plot(fr, d, '-o', markersize=3, linewidth=lw, alpha=alpha, label=f'sl{s}')
    ax.axvline(x=5.5, color='green', linestyle='--', alpha=0.5, label='burnin|rollout')
    ax.set_title(f'All slots depth (sl{sl} bold)', fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('depth')
    
    # Col 2: 目标 slot 的 pos + depth (双轴)
    ax = axes[j, 2]
    ln1 = ax.plot(fr, c['pos_x'], 'r-o', markersize=3, label='pos_x')
    ln2 = ax.plot(fr, c['pos_y'], 'b-o', markersize=3, label='pos_y')
    ax.axvline(x=5.5, color='green', linestyle='--', alpha=0.5)
    ax.set_ylabel('pos', fontsize=9)
    ax2 = ax.twinx()
    ln3 = ax2.plot(fr, c['depth'], 'g-s', markersize=4, label='depth', linewidth=2)
    # 标注 depth 极值点
    d = c['depth']
    for k in range(1, 15):
        if (d[k] > d[k-1] and d[k] > d[k+1]):
            ax2.plot(k, d[k], 'rv', markersize=10)
        elif (d[k] < d[k-1] and d[k] < d[k+1]):
            ax2.plot(k, d[k], 'r^', markersize=10)
    # 标注最大跳变
    ji = c['max_jump_idx']
    ax2.annotate(f'jump={c["max_jump"]:.3f}', xy=(ji+0.5, d[ji]), 
                xytext=(ji+2, d[ji]+0.01),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=8, color='red')
    ax2.set_ylabel('depth', color='green', fontsize=9)
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=7, loc='upper left')
    ax.axvline(x=5.5, color='green', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'sl{sl}: pos+depth (jump@t={ji}→{ji+1})', fontsize=9)
    
    # Col 3: Slot 重建图 (4帧: burnin末 + rollout初/中/末)
    ax = axes[j, 3]
    # 获取所有 slot 的重建
    with torch.no_grad():
        # 重建4帧的完整场景
        slots_t = [all_slots[t] for t in [4, 6, 9, 14]]
        recons = []
        for t_idx, slots_frame in enumerate([4, 6, 9, 14]):
            s_data = all_slots[slots_frame].unsqueeze(0).cuda()  # (1, N, D)
            recon_all = model.decoder(s_data)  # (1, N, 4, H, W)
            # 只取目标 slot
            slot_recon = recon_all[0, sl].cpu()  # (4, H, W)
            rgb = slot_recon[:3].numpy().transpose(1, 2, 0)
            alpha = slot_recon[3].numpy()
            img = np.clip((rgb + 1) / 2, 0, 1) * alpha[:,:,np.newaxis]
            recons.append(img)
    
    composite = np.concatenate(recons, axis=1)
    ax.imshow(composite)
    ax.set_title(f'sl{sl} reconstruction (t=4,6,9,14)', fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/depth_erratic_visual.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: pos_depth_debug/depth_erratic_visual.png")

# 额外统计: 所有物体的depth最大帧间跳变
all_jumps = [c['max_jump'] for c in cases]
if all_jumps:
    print(f"\nDepth max frame-to-frame jump stats (all moving objects):")
    print(f"  mean={np.mean(all_jumps):.4f} median={np.median(all_jumps):.4f}")
    print(f"  min={np.min(all_jumps):.4f} max={np.max(all_jumps):.4f}")
    
    # 跳变位置分布
    jump_indices = [c['max_jump_idx'] for c in cases]
    print(f"\n  Jump position distribution:")
    for pos in range(15):
        count = sum(1 for idx in jump_indices if idx == pos)
        if count > 0:
            print(f"    t={pos}→{pos+1}: {count} cases")
