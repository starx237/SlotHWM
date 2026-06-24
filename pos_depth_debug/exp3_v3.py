"""
实验3v3: 长时程跟踪 - 逐帧独立编码
用 ISA encoder + slot_attention 逐帧提取slots，再跨帧匹配物体
这样可以获取任意长度的轨迹
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

# 逐帧独立编码
def encode_single_frame(model, frame):
    """对单帧做 encoder + slot_attention，返回 slots. frame: (B, C, H, W)"""
    with torch.no_grad():
        frame_5d = frame.unsqueeze(1)  # (B, 1, C, H, W)
        encoder_out = model.encoder(frame_5d)  # (B, 1, N, D)
        encoder_out = encoder_out.squeeze(1)  # (B, N, D)
        slots = model.slot_attention(encoder_out)  # (B, num_slots, D)
    return slots

# 用 stride=1 获取50帧，逐帧编码
ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=50, stride=1, subsample=1)

# 匈牙利匹配: 用 appearance 距离跨帧关联 slot
from scipy.optimize import linear_sum_assignment

def match_slots(slots_t, slots_t1, app_dim):
    """用 appearance cosine distance 匹配两帧的 slots"""
    app_t = slots_t[:, :app_dim]
    app_t1 = slots_t1[:, :app_dim]
    
    # cosine similarity
    app_t_norm = app_t / (app_t.norm(dim=-1, keepdim=True) + 1e-8)
    app_t1_norm = app_t1 / (app_t1.norm(dim=-1, keepdim=True) + 1e-8)
    sim = app_t_norm @ app_t1_norm.T  # (N, N)
    cost = 1 - sim.cpu().numpy()
    
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind

def build_tracks(all_frame_slots, app_dim, depth_max):
    """从逐帧 slots 构建轨迹"""
    T = len(all_frame_slots)
    N = all_frame_slots[0].shape[0]
    
    # 初始化: 第一帧每个 slot 一条轨迹
    tracks = {s: [{'frame': 0, 'slot_idx': s, 'data': all_frame_slots[0][s]}] for s in range(N)}
    active = set(range(N))
    
    for t in range(1, T):
        if not active:
            break
        slots_t = all_frame_slots[t]
        prev_slots = torch.stack([tracks[s][-1]['data'] for s in sorted(active)])
        active_list = sorted(active)
        
        row_ind, col_ind = match_slots(prev_slots, slots_t, app_dim)
        
        matched_prev = set()
        matched_curr = set()
        for ri, ci in zip(row_ind, col_ind):
            s = active_list[ri]
            tracks[s].append({'frame': t, 'slot_idx': ci, 'data': slots_t[ci]})
            matched_prev.add(ri)
            matched_curr.add(ci)
        
        # 未匹配的轨迹标记结束
        for ri in range(len(active_list)):
            if ri not in matched_prev:
                active.discard(active_list[ri])
    
    return tracks

# 处理几个样本
boundary_threshold = 0.8
selected_tracks = []

for sample_idx in range(10):
    batch = ds[sample_idx]
    frames = batch['video'].unsqueeze(0).cuda()  # (1, 50, 3, 64, 64)
    
    # 逐帧编码
    all_frame_slots = []
    for t in range(50):
        frame_t = frames[:, t]  # (1, 3, 64, 64)
        slots_t = encode_single_frame(model, frame_t)  # (1, N, D)
        all_frame_slots.append(slots_t[0].cpu())  # (N, D)
    
    # 构建轨迹
    tracks = build_tracks(all_frame_slots, app_dim, depth_max)
    
    for s, track in tracks.items():
        if len(track) < 20:
            continue
        
        # 提取轨迹数据
        track_frames = [pt['frame'] for pt in track]
        track_data = torch.stack([pt['data'] for pt in track])
        
        depth = track_data[:, app_dim+2].numpy()
        pos_x = track_data[:, app_dim].numpy()
        pos_y = track_data[:, app_dim+1].numpy()
        
        # 过滤背景
        if np.mean(depth[:5]) >= depth_max:
            continue
        
        # 严格边界筛选
        in_frame = (np.abs(pos_x) < boundary_threshold) & (np.abs(pos_y) < boundary_threshold) & (depth < depth_max)
        
        # 找最长的在画面内连续段
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
            if seg_len < 20: continue
            move = np.sqrt((pos_x[seg_end]-pos_x[seg_start])**2 + (pos_y[seg_end]-pos_y[seg_start])**2)
            if move < 0.1: continue
            
            selected_tracks.append({
                'sample': sample_idx,
                'slot': s,
                'track_len': len(track),
                'seg_start': seg_start,
                'seg_end': seg_end,
                'seg_len': seg_len,
                'pos_x': pos_x[seg_start:seg_end+1],
                'pos_y': pos_y[seg_start:seg_end+1],
                'depth': depth[seg_start:seg_end+1],
                'total_move': move,
            })
    
    print(f"Sample {sample_idx}: found {len([t for t in selected_tracks if t['sample']==sample_idx])} tracks", flush=True)

print(f"\nTotal tracks (>=20 frames, in-frame, moving): {len(selected_tracks)}")
selected_tracks.sort(key=lambda t: t['total_move'], reverse=True)

# 选5个不同样本的代表
final = []
used = set()
for t in selected_tracks:
    if t['sample'] not in used and len(final) < 5:
        final.append(t)
        used.add(t['sample'])

# 如果不够5个，放宽
if len(final) < 5:
    for t in selected_tracks:
        if t not in final:
            final.append(t)
        if len(final) >= 5:
            break

print(f"Selected {len(final)} representative tracks:")
for j, t in enumerate(final):
    print(f"  [{j}] s{t['sample']}_sl{t['slot']} seg={t['seg_len']}f move={t['total_move']:.3f}")

# 绘制折线图
fig, axes = plt.subplots(len(final), 3, figsize=(18, 4*len(final)))
fig.suptitle('Long-term Tracking (50 frames, per-frame encoding, strict boundary)', fontsize=14)

for j, t in enumerate(final):
    frames_range = np.arange(t['seg_len'])
    depth = t['depth']
    pos_x = t['pos_x']
    pos_y = t['pos_y']
    
    # Depth
    ax = axes[j, 0]
    ax.plot(frames_range, depth, 'b-o', markersize=2, label='depth')
    
    if depth.std() > 1e-6:
        c1 = np.polyfit(frames_range, depth, 1)
        f1 = np.polyval(c1, frames_range)
        r2_1 = 1 - np.sum((depth-f1)**2)/(np.sum((depth-depth.mean())**2)+1e-8)
        ax.plot(frames_range, f1, 'r--', label=f'linear R²={r2_1:.3f}')
        
        c2 = np.polyfit(frames_range, depth, 2)
        f2 = np.polyval(c2, frames_range)
        r2_2 = 1 - np.sum((depth-f2)**2)/(np.sum((depth-depth.mean())**2)+1e-8)
        ax.plot(frames_range, f2, 'g--', label=f'quad R²={r2_2:.3f}')
        
        inv_d = 1.0/(depth+1e-6)
        ci = np.polyfit(frames_range, inv_d, 1)
        fi = 1.0/(np.polyval(ci, frames_range)+1e-6)
        r2_i = 1 - np.sum((depth-fi)**2)/(np.sum((depth-depth.mean())**2)+1e-8)
        ax.plot(frames_range, fi, 'm--', label=f'inv-d R²={r2_i:.3f}')
    
    ax.set_title(f's{t["sample"]}_sl{t["slot"]} ({t["seg_len"]}f, move={t["total_move"]:.2f})')
    ax.set_ylabel('Depth')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Pos X
    axes[j, 1].plot(frames_range, pos_x, 'r-o', markersize=2)
    axes[j, 1].set_title(f'pos_x')
    axes[j, 1].set_ylabel('pos_x')
    axes[j, 1].grid(True, alpha=0.3)
    
    # Pos Y
    axes[j, 2].plot(frames_range, pos_y, 'g-o', markersize=2)
    axes[j, 2].set_title(f'pos_y')
    axes[j, 2].set_ylabel('pos_y')
    axes[j, 2].grid(True, alpha=0.3)

axes[-1, 0].set_xlabel('Frame')
axes[-1, 1].set_xlabel('Frame')
axes[-1, 2].set_xlabel('Frame')
plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/long_term_depth_v3.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: pos_depth_debug/long_term_depth_v3.png")

# Depth delta 图
fig2, axes2 = plt.subplots(len(final), 2, figsize=(14, 4*len(final)))
fig2.suptitle('Depth Delta & Depth vs pos_x', fontsize=14)

for j, t in enumerate(final):
    depth = t['depth']
    pos_x = t['pos_x']
    fr = np.arange(t['seg_len'])
    
    d_delta = np.diff(depth)
    axes2[j, 0].plot(fr[1:], d_delta, 'b-o', markersize=2)
    axes2[j, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes2[j, 0].set_title(f'Δdepth (s{t["sample"]}_sl{t["slot"]})')
    axes2[j, 0].set_ylabel('Δdepth')
    axes2[j, 0].grid(True, alpha=0.3)
    
    sc = axes2[j, 1].scatter(pos_x, depth, c=fr, cmap='viridis', s=15)
    axes2[j, 1].set_title(f'depth vs pos_x')
    axes2[j, 1].set_xlabel('pos_x')
    axes2[j, 1].set_ylabel('depth')
    plt.colorbar(sc, ax=axes2[j, 1], label='frame')
    axes2[j, 1].grid(True, alpha=0.3)

axes2[-1, 0].set_xlabel('Frame')
plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/depth_delta_long_v3.png', dpi=150, bbox_inches='tight')
print("Saved: pos_depth_debug/depth_delta_long_v3.png")

# 统计
print("\n=== Depth linearity (all tracks) ===")
lin_r2s, quad_r2s = [], []
for t in selected_tracks[:30]:
    depth = t['depth']
    if depth.std() < 1e-6: continue
    fr = np.arange(len(depth))
    c1 = np.polyfit(fr, depth, 1)
    r2_1 = 1 - np.sum((depth-np.polyval(c1,fr))**2)/(np.sum((depth-depth.mean())**2)+1e-8)
    c2 = np.polyfit(fr, depth, 2)
    r2_2 = 1 - np.sum((depth-np.polyval(c2,fr))**2)/(np.sum((depth-depth.mean())**2)+1e-8)
    lin_r2s.append(r2_1)
    quad_r2s.append(r2_2)

if lin_r2s:
    print(f"R²(lin):  mean={np.mean(lin_r2s):.4f} median={np.median(lin_r2s):.4f} min={np.min(lin_r2s):.4f}")
    print(f"R²(quad): mean={np.mean(quad_r2s):.4f} median={np.median(quad_r2s):.4f} min={np.min(quad_r2s):.4f}")
