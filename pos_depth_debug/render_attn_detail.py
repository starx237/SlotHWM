"""
单独渲染 Slot 0 的16帧注意力热力图，大尺寸，更清晰
加上 depth 值和帧号标注
"""
import sys, os
sys.path.insert(0, '/autodl-fs/data/SlotHWM')
import warnings; warnings.filterwarnings('ignore')
import torch
import numpy as np
import yaml, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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

SI = 9
TARGET_SLOT = 0

all_attns = []
def attn_hook(module, input, output):
    if isinstance(output, tuple) and len(output) == 2:
        all_attns.append(output[1].detach().cpu())
handle = model.slot_attention.register_forward_hook(attn_hook)

sample = ds[SI]
frames = sample['video'].unsqueeze(0).cuda()

with torch.no_grad():
    out = model(frames)

handle.remove()

all_slots = torch.cat([out['slots']['corrected'], out['slots']['target']], dim=1)[0]

# 图1: 4行 x 4列 = 16帧的注意力热力图 + 视频帧 + slot重建
fig, axes = plt.subplots(4, 8, figsize=(32, 16))
fig.suptitle(f'Sample {SI}, Slot {TARGET_SLOT}: Video | Slot Reconstruction | Attention Heatmap\n'
             f'depth (green) = sqrt(Σ attn × spread), green line = burnin|rollout boundary', 
             fontsize=13, y=1.01)

ATTN_H = ATTN_W = 16

for t in range(16):
    row = t // 4
    col_base = (t % 4) * 2
    
    # 视频帧
    ax = axes[row, col_base]
    frame = sample['video'][t].permute(1, 2, 0).numpy()
    ax.imshow(np.clip((frame+1)/2, 0, 1))
    depth_t = all_slots[t, TARGET_SLOT, app_dim+2].item()
    pos_x_t = all_slots[t, TARGET_SLOT, app_dim].item()
    pos_y_t = all_slots[t, TARGET_SLOT, app_dim+1].item()
    
    # 在帧上标注 slot 的 pos
    px = (pos_x_t + 1) / 2 * 64
    py = (pos_y_t + 1) / 2 * 64
    ax.plot(px, py, 'r+', markersize=12, markeredgewidth=2)
    
    title_color = 'red' if t >= 6 else 'black'
    ax.set_title(f't={t} d={depth_t:.4f}\npos=({pos_x_t:.2f},{pos_y_t:.2f})', fontsize=8, color=title_color)
    ax.axis('off')
    
    # 注意力热力图
    ax2 = axes[row, col_base + 1]
    if t < len(all_attns):
        attn_t = all_attns[t][0, TARGET_SLOT].numpy()
        attn_2d = attn_t.reshape(ATTN_H, ATTN_W)
        
        im = ax2.imshow(attn_2d, cmap='hot', interpolation='nearest')
        
        # 在注意力图上标注 pos
        # pos 在 [-1,1]，映射到 [0, ATTN_H-1]
        px_a = (pos_x_t + 1) / 2 * (ATTN_W - 1)
        py_a = (pos_y_t + 1) / 2 * (ATTN_H - 1)
        ax2.plot(px_a, py_a, 'c+', markersize=12, markeredgewidth=2)
        
        attn_max = attn_t.max()
        attn_entropy = -np.sum(attn_t * np.log(attn_t + 1e-10))
        ax2.set_title(f'attn max={attn_max:.3f}\nH={attn_entropy:.2f}', fontsize=8, color=title_color)
        
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    ax2.axis('off')
    
    # 在 burnin/rollout 边界加红线
    if t == 5:
        for ax_i in [ax, ax2]:
            for spine in ax_i.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)

plt.tight_layout()
out_path = '/autodl-fs/data/SlotHWM/pos_depth_debug/sample9_attn_detail.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')

# 图2: depth + attn entropy + attn max 的时间序列
fig2, axes2 = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

depth_vals = all_slots[:, TARGET_SLOT, app_dim+2].cpu().numpy()
pos_x_vals = all_slots[:, TARGET_SLOT, app_dim].cpu().numpy()
pos_y_vals = all_slots[:, TARGET_SLOT, app_dim+1].cpu().numpy()
fr = np.arange(16)

attn_max_vals = []
attn_entropy_vals = []
for t in range(min(16, len(all_attns))):
    attn_t = all_attns[t][0, TARGET_SLOT].numpy()
    attn_max_vals.append(attn_t.max())
    attn_entropy_vals.append(-np.sum(attn_t * np.log(attn_t + 1e-10)))

# pos
axes2[0].plot(fr, pos_x_vals, 'r-o', label='pos_x')
axes2[0].plot(fr, pos_y_vals, 'b-o', label='pos_y')
axes2[0].axvline(x=5.5, color='green', linestyle='--', alpha=0.5)
axes2[0].set_ylabel('Position')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)
axes2[0].set_title(f'Sample {SI}, Slot {TARGET_SLOT}')

# depth
axes2[1].plot(fr, depth_vals, 'g-o', linewidth=2)
for k in range(1, 15):
    if depth_vals[k] > depth_vals[k-1] and depth_vals[k] > depth_vals[k+1]:
        axes2[1].plot(k, depth_vals[k], 'rv', markersize=10)
    elif depth_vals[k] < depth_vals[k-1] and depth_vals[k] < depth_vals[k+1]:
        axes2[1].plot(k, depth_vals[k], 'r^', markersize=10)
axes2[1].axvline(x=5.5, color='green', linestyle='--', alpha=0.5)
axes2[1].set_ylabel('Depth')
axes2[1].grid(True, alpha=0.3)

# attn max
axes2[2].plot(fr[:len(attn_max_vals)], attn_max_vals, 'r-o')
axes2[2].axvline(x=5.5, color='green', linestyle='--', alpha=0.5)
axes2[2].set_ylabel('Attn Max')
axes2[2].grid(True, alpha=0.3)

# attn entropy
axes2[3].plot(fr[:len(attn_entropy_vals)], attn_entropy_vals, 'b-o')
axes2[3].axvline(x=5.5, color='green', linestyle='--', alpha=0.5)
axes2[3].set_ylabel('Attn Entropy')
axes2[3].set_xlabel('Frame')
axes2[3].grid(True, alpha=0.3)

plt.tight_layout()
out_path2 = '/autodl-fs/data/SlotHWM/pos_depth_debug/sample9_timeseries.png'
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path2}')
