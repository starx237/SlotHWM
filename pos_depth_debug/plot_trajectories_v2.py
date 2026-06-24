"""
绘制 v2: 排除 fg↔bg 突变后的轨迹图
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = np.load('/autodl-fs/data/SlotHWM/pos_depth_debug/selected_objects_v2.npy', allow_pickle=True).tolist()

# 运动物体用鲜明的颜色，静止物体用灰色
n_moving = sum(1 for d in data if d['direction'] != 'static')
n_static = len(data) - n_moving

cmap_moving = plt.cm.tab10(np.linspace(0, 1, max(n_moving, 1)))
cmap_static = plt.cm.Greys(np.linspace(0.4, 0.7, max(n_static, 1)))
colors = []
mi, si = 0, 0
for d in data:
    if d['direction'] != 'static':
        colors.append(cmap_moving[mi % len(cmap_moving)])
        mi += 1
    else:
        colors.append(cmap_static[si % len(cmap_static)])
        si += 1

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
labels = ['pos_x', 'pos_y', 'depth']

for j, d in enumerate(data):
    frames = np.arange(16)
    direction = d['direction']
    depth_ch = d['depth_change']
    if direction != 'static':
        label = f"s{d['sample_idx']}_sl{d['slot_idx']}(mv-{direction[:3]}-{depth_ch[:3]})"
        lw = 2.0
        alpha = 1.0
    else:
        label = f"s{d['sample_idx']}_sl{d['slot_idx']}(static-{depth_ch[:3]})"
        lw = 1.0
        alpha = 0.6
    
    axes[0].plot(frames, d['pos_x'], color=colors[j], label=label, linewidth=lw, alpha=alpha)
    axes[1].plot(frames, d['pos_y'], color=colors[j], label=label, linewidth=lw, alpha=alpha)
    axes[2].plot(frames, d['depth'], color=colors[j], label=label, linewidth=lw, alpha=alpha)

for i, ax in enumerate(axes):
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel(labels[i], fontsize=12)
    ax.set_title(labels[i] + ' over time (stable fg only)', fontsize=13)
    ax.axvline(x=6, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.grid(True, alpha=0.3)

axes[0].annotate('burnin→rollout', xy=(6, axes[0].get_ylim()[0]), fontsize=8, color='red', alpha=0.6)
axes[2].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6.5, ncol=1)

plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/trajectories_v2.png', dpi=150, bbox_inches='tight')
print("Saved trajectories_v2.png")

# Delta 图
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
delta_labels = ['delta_pos_x', 'delta_pos_y', 'delta_depth']

for j, d in enumerate(data):
    direction = d['direction']
    depth_ch = d['depth_change']
    if direction != 'static':
        label = f"s{d['sample_idx']}_sl{d['slot_idx']}(mv-{direction[:3]}-{depth_ch[:3]})"
        lw = 2.0; alpha = 1.0
    else:
        label = f"s{d['sample_idx']}_sl{d['slot_idx']}(static-{depth_ch[:3]})"
        lw = 1.0; alpha = 0.6
    
    axes2[0].plot(np.arange(15), np.diff(d['pos_x']), color=colors[j], label=label, linewidth=lw, alpha=alpha)
    axes2[1].plot(np.arange(15), np.diff(d['pos_y']), color=colors[j], label=label, linewidth=lw, alpha=alpha)
    axes2[2].plot(np.arange(15), np.diff(d['depth']), color=colors[j], label=label, linewidth=lw, alpha=alpha)

for i, ax in enumerate(axes2):
    ax.set_xlabel('Frame transition', fontsize=12)
    ax.set_ylabel(delta_labels[i], fontsize=12)
    ax.set_title(delta_labels[i] + ' per frame (stable fg only)', fontsize=13)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=5, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.grid(True, alpha=0.3)

axes2[2].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6.5, ncol=1)

plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/delta_trajectories_v2.png', dpi=150, bbox_inches='tight')
print("Saved delta_trajectories_v2.png")

# Depth vs 1/depth
fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))

for j, d in enumerate(data):
    direction = d['direction']
    depth_ch = d['depth_change']
    if direction != 'static':
        label = f"s{d['sample_idx']}_sl{d['slot_idx']}(mv-{direction[:3]}-{depth_ch[:3]})"
        lw = 2.0; alpha = 1.0
    else:
        label = f"s{d['sample_idx']}_sl{d['slot_idx']}(static-{depth_ch[:3]})"
        lw = 1.0; alpha = 0.6
    
    depth = d['depth']
    inv_depth = 1.0 / (depth + 1e-6)
    
    axes3[0].plot(np.arange(16), depth, color=colors[j], label=label, linewidth=lw, alpha=alpha)
    axes3[1].plot(np.arange(16), inv_depth, color=colors[j], label=label, linewidth=lw, alpha=alpha)

axes3[0].set_title('depth over time', fontsize=13)
axes3[0].set_xlabel('Frame'); axes3[0].set_ylabel('depth')
axes3[0].axvline(x=6, color='red', linestyle='--', alpha=0.4)
axes3[0].grid(True, alpha=0.3)

axes3[1].set_title('1/depth over time', fontsize=13)
axes3[1].set_xlabel('Frame'); axes3[1].set_ylabel('1/depth')
axes3[1].axvline(x=6, color='red', linestyle='--', alpha=0.4)
axes3[1].grid(True, alpha=0.3)

axes3[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6.5, ncol=1)

plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/depth_vs_invdepth_v2.png', dpi=150, bbox_inches='tight')
print("Saved depth_vs_invdepth_v2.png")

# 量化分析
print("\n=== Linearity: depth vs 1/depth (rollout frames 6-15) ===")
r2_results = []
for j, d in enumerate(data):
    depth = d['depth'][6:]
    inv_depth = 1.0 / (depth + 1e-6)
    frames = np.arange(len(depth))
    
    coef_d = np.polyfit(frames, depth, 1)
    coef_id = np.polyfit(frames, inv_depth, 1)
    fit_d = np.polyval(coef_d, frames)
    fit_id = np.polyval(coef_id, frames)
    r2_d = 1 - np.sum((depth - fit_d)**2) / (np.sum((depth - depth.mean())**2) + 1e-8)
    r2_id = 1 - np.sum((inv_depth - fit_id)**2) / (np.sum((inv_depth - inv_depth.mean())**2) + 1e-8)
    
    direction = d['direction']
    depth_ch = d['depth_change']
    r2_results.append((j, r2_d, r2_id, direction, depth_ch))
    better = 'inv' if r2_id > r2_d else 'linear'
    print(f"  [{j:2d}] s{d['sample_idx']}_sl{d['slot_idx']}: R²(d)={r2_d:.4f} R²(1/d)={r2_id:.4f} better={better:6s} {direction[:4]} depth_{depth_ch[:3]}")

# 统计
linear_better = sum(1 for _, r2d, r2id, _, _ in r2_results if r2d > r2id)
inv_better = len(r2_results) - linear_better
print(f"\nLinear better: {linear_better}, Inverse-linear better: {inv_better}")

# pos 的线性度
print("\n=== pos linearity (rollout frames 6-15) ===")
for j, d in enumerate(data):
    pos_x = d['pos_x'][6:]
    pos_y = d['pos_y'][6:]
    frames = np.arange(len(pos_x))
    
    r2_px = 1 - np.sum((pos_x - np.polyval(np.polyfit(frames, pos_x, 1), frames))**2) / (np.sum((pos_x - pos_x.mean())**2) + 1e-8)
    r2_py = 1 - np.sum((pos_y - np.polyval(np.polyfit(frames, pos_y, 1), frames))**2) / (np.sum((pos_y - pos_y.mean())**2) + 1e-8)
    
    direction = d['direction']
    depth_ch = d['depth_change']
    print(f"  [{j:2d}] R²(px)={r2_px:.4f} R²(py)={r2_py:.4f} {direction[:4]} depth_{depth_ch[:3]}")
