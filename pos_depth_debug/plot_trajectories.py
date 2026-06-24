"""
绘制 pos_x, pos_y, depth 随时间步变化的折线图
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = np.load('/autodl-fs/data/SlotHWM/pos_depth_debug/selected_objects.npy', allow_pickle=True).tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
labels = ['pos_x', 'pos_y', 'depth']

colors = plt.cm.tab20(np.linspace(0, 1, len(data)))

for j, d in enumerate(data):
    frames = np.arange(16)
    pos_x = d['pos_x']
    pos_y = d['pos_y']
    depth = d['depth']
    
    direction = d.get('direction', 'static')
    depth_ch = d.get('depth_change', 'N/A')
    
    move_type = f"mv-{direction[0]}d-{depth_ch[0]}" if direction != 'static' else 'static'
    label = f"s{d['sample_idx']}_sl{d['slot_idx']}({move_type})"
    
    axes[0].plot(frames, pos_x, color=colors[j], label=label, linewidth=1.5)
    axes[1].plot(frames, pos_y, color=colors[j], label=label, linewidth=1.5)
    axes[2].plot(frames, depth, color=colors[j], label=label, linewidth=1.5)

for i, ax in enumerate(axes):
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel(labels[i], fontsize=12)
    ax.set_title(labels[i] + ' over time', fontsize=14)
    ax.axvline(x=6, color='gray', linestyle='--', alpha=0.5, label='burnin end' if i == 0 else '')
    ax.grid(True, alpha=0.3)

# Legend outside
axes[2].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=1)

plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/trajectories.png', dpi=150, bbox_inches='tight')
print("Saved trajectories.png")

# 也做逐帧 delta 分析
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
delta_labels = ['delta_pos_x', 'delta_pos_y', 'delta_depth']

for j, d in enumerate(data):
    pos_x = d['pos_x']
    pos_y = d['pos_y']
    depth = d['depth']
    
    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    dd = np.diff(depth)
    
    direction = d.get('direction', 'static')
    depth_ch = d.get('depth_change', 'N/A')
    move_type = f"mv-{direction[0]}d-{depth_ch[0]}" if direction != 'static' else 'static'
    label = f"s{d['sample_idx']}_sl{d['slot_idx']}({move_type})"
    
    axes2[0].plot(np.arange(15), dx, color=colors[j], label=label, linewidth=1.5)
    axes2[1].plot(np.arange(15), dy, color=colors[j], label=label, linewidth=1.5)
    axes2[2].plot(np.arange(15), dd, color=colors[j], label=label, linewidth=1.5)

for i, ax in enumerate(axes2):
    ax.set_xlabel('Frame transition', fontsize=12)
    ax.set_ylabel(delta_labels[i], fontsize=12)
    ax.set_title(delta_labels[i] + ' per frame', fontsize=14)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

axes2[2].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=1)

plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/delta_trajectories.png', dpi=150, bbox_inches='tight')
print("Saved delta_trajectories.png")

# 1/depth 分析 (inverse linear hypothesis)
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))

for j, d in enumerate(data):
    depth = d['depth']
    inv_depth = 1.0 / (depth + 1e-6)
    
    direction = d.get('direction', 'static')
    depth_ch = d.get('depth_change', 'N/A')
    move_type = f"mv-{direction[0]}d-{depth_ch[0]}" if direction != 'static' else 'static'
    label = f"s{d['sample_idx']}_sl{d['slot_idx']}({move_type})"
    
    axes3[0].plot(np.arange(16), depth, color=colors[j], label=label, linewidth=1.5)
    axes3[1].plot(np.arange(16), inv_depth, color=colors[j], label=label, linewidth=1.5)

axes3[0].set_title('depth over time', fontsize=14)
axes3[0].set_xlabel('Frame')
axes3[0].set_ylabel('depth')
axes3[0].axvline(x=6, color='gray', linestyle='--', alpha=0.5)
axes3[0].grid(True, alpha=0.3)

axes3[1].set_title('1/depth over time', fontsize=14)
axes3[1].set_xlabel('Frame')
axes3[1].set_ylabel('1/depth')
axes3[1].axvline(x=6, color='gray', linestyle='--', alpha=0.5)
axes3[1].grid(True, alpha=0.3)

axes3[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=1)

plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/depth_vs_invdepth.png', dpi=150, bbox_inches='tight')
print("Saved depth_vs_invdepth.png")

# 量化分析: depth 的线性度 vs 1/depth 的线性度
print("\n=== Linearity analysis: depth vs 1/depth ===")
from numpy.polynomial import polynomial as P
for j, d in enumerate(data[:10]):
    depth = d['depth'][6:]  # 只看 rollout 部分 (frame 6-15)
    inv_depth = 1.0 / (depth + 1e-6)
    frames = np.arange(len(depth))
    
    # 线性拟合
    coef_d = np.polyfit(frames, depth, 1)
    coef_id = np.polyfit(frames, inv_depth, 1)
    
    fit_d = np.polyval(coef_d, frames)
    fit_id = np.polyval(coef_id, frames)
    
    r2_d = 1 - np.sum((depth - fit_d)**2) / (np.sum((depth - depth.mean())**2) + 1e-8)
    r2_id = 1 - np.sum((inv_depth - fit_id)**2) / (np.sum((inv_depth - inv_depth.mean())**2) + 1e-8)
    
    direction = d.get('direction', 'static')
    depth_ch = d.get('depth_change', 'N/A')
    print(f"  [{j}] s{d['sample_idx']}_sl{d['slot_idx']}: R²(depth)={r2_d:.4f} R²(1/depth)={r2_id:.4f} better={'inv' if r2_id > r2_d else 'linear'} depth_ch={depth_ch}")
