"""
实验2+3: 
1. 严格边界筛选: 用 alpha mask 判断物体是否完全在画面内
   ISA decoder 输出 alpha mask，可以用来判断物体的像素覆盖范围
   更简单: 用 pos 坐标判断 - 如果 |pos_x| 或 |pos_y| > threshold，认为靠近边界
   
2. 长时程 depth 跟踪: 用更多帧的视频 (20-50帧)
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')

import torch
import numpy as np
import yaml
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset

with open('config/obj3d.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)

app_dim = cfg.appearance_dim
depth_max = getattr(cfg, 'depth_max', 0.30)

model = SlotDynamicsModel(cfg).cuda()
ckpt = torch.load(cfg.pretrained_path, map_location='cpu')
model_state = model.state_dict()
loaded = {}
for mk in model_state:
    mk_c = mk.replace('_orig_mod.', '')
    for ck in ckpt['model']:
        ck_c = ck.replace('_orig_mod.', '')
        if ck_c == mk_c and ckpt['model'][ck].shape == model_state[mk].shape:
            loaded[mk] = ckpt['model'][ck]
            break
model.load_state_dict(loaded, strict=False)
model.eval()

# ========== 实验2: 严格边界筛选 ==========
# 思路: 用 slot 的 pos 坐标判断物体是否在画面内
# ISA 的 pos 坐标范围大约是 [-1, 1]，当 |pos| > 0.8 时物体可能已靠近边缘
# 更精确: 用 decoder 的 alpha mask 判断物体像素覆盖

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=1, shuffle=True, num_workers=0)

# 先收集带边界信息的轨迹
boundary_threshold = 0.7  # pos 超过此值认为靠近边界

all_segments = []  # 每个: {sample, slot, frames_start, frames_end, pos_x, pos_y, depth}

for i in range(50):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = model(frames)
    
    corrected = out['slots']['corrected']
    target = out['slots']['target']
    all_slots = torch.cat([corrected, target], dim=1)[0]  # (16, N, D)
    
    N = all_slots.shape[1]
    for s in range(N):
        depth_vals = all_slots[:, s, app_dim+2].cpu().numpy()
        pos_x = all_slots[:, s, app_dim].cpu().numpy()
        pos_y = all_slots[:, s, app_dim+1].cpu().numpy()
        
        # 跳过背景 slot
        if np.mean(depth_vals) >= depth_max:
            continue
        
        # 找到完全在画面内的连续段
        in_frame = (np.abs(pos_x) < boundary_threshold) & (np.abs(pos_y) < boundary_threshold) & (depth_vals < depth_max)
        
        # 找最长连续段
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
            if seg_end - seg_start + 1 >= 8:  # 至少8帧
                all_segments.append({
                    'sample': i,
                    'slot': s,
                    'seg_start': seg_start,
                    'seg_end': seg_end,
                    'seg_len': seg_end - seg_start + 1,
                    'pos_x': pos_x[seg_start:seg_end+1],
                    'pos_y': pos_y[seg_start:seg_end+1],
                    'depth': depth_vals[seg_start:seg_end+1],
                    'total_move': np.sqrt((pos_x[seg_end]-pos_x[seg_start])**2 + (pos_y[seg_end]-pos_y[seg_start])**2),
                })

print(f"Total in-frame segments (>=8 frames): {len(all_segments)}")

# 选运动幅度最大的
moving_segs = [s for s in all_segments if s['total_move'] > 0.1]
moving_segs.sort(key=lambda s: s['total_move'], reverse=True)
print(f"Moving segments: {len(moving_segs)}")

# 取 top 15 做深度分析
top_segs = moving_segs[:15]
print("\nTop 15 moving in-frame segments:")
for j, s in enumerate(top_segs):
    print(f"  [{j}] s{s['sample']}_sl{s['slot']} frames={s['seg_start']}-{s['seg_end']}({s['seg_len']}) move={s['total_move']:.3f}")

# 严格筛选后的 depth 线性度分析
print("\n=== Depth linearity (strictly in-frame only) ===")
for j, s in enumerate(top_segs):
    depth = s['depth']
    inv_depth = 1.0 / (depth + 1e-6)
    frames = np.arange(len(depth))
    
    if depth.std() < 1e-6:
        print(f"  [{j}] s{s['sample']}_sl{s['slot']}: depth nearly constant, skip")
        continue
    
    coef_d = np.polyfit(frames, depth, 1)
    coef_id = np.polyfit(frames, inv_depth, 1)
    fit_d = np.polyval(coef_d, frames)
    fit_id = np.polyval(coef_id, frames)
    r2_d = 1 - np.sum((depth - fit_d)**2) / (np.sum((depth - depth.mean())**2) + 1e-8)
    r2_id = 1 - np.sum((inv_depth - fit_id)**2) / (np.sum((inv_depth - inv_depth.mean())**2) + 1e-8)
    
    # 二次拟合
    coef_d2 = np.polyfit(frames, depth, 2)
    fit_d2 = np.polyval(coef_d2, frames)
    r2_d2 = 1 - np.sum((depth - fit_d2)**2) / (np.sum((depth - depth.mean())**2) + 1e-8)
    
    better = 'inv' if r2_id > r2_d else 'linear'
    print(f"  [{j}] s{s['sample']}_sl{s['slot']}: R²(lin)={r2_d:.4f} R²(inv)={r2_id:.4f} R²(quad)={r2_d2:.4f} better={better:6s} move={s['total_move']:.2f}")

# ========== 实验3: 长时程 depth 跟踪 ==========
# OBJ3D 默认 16 帧 (stride=4), 用 stride=1 可以得到 16*4=64 帧
print("\n\n========== 长时程跟踪 ==========")

# 创建 stride=1 的数据集获取更多帧
ds_long = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=1, subsample=2)
loader_long = ds_long.get_dataloader(batch_size=1, shuffle=True, num_workers=0)

long_tracks = []
for i in range(20):
    batch = next(iter(loader_long))
    frames_long = batch['video'].cuda()
    with torch.no_grad():
        out = model(frames_long)
    
    corrected = out['slots']['corrected']
    target = out['slots']['target']
    all_slots = torch.cat([corrected, target], dim=1)[0]
    
    N = all_slots.shape[1]
    for s in range(N):
        depth_vals = all_slots[:, s, app_dim+2].cpu().numpy()
        pos_x = all_slots[:, s, app_dim].cpu().numpy()
        pos_y = all_slots[:, s, app_dim+1].cpu().numpy()
        
        if np.mean(depth_vals) >= depth_max:
            continue
        
        # 找严格在画面内的长段
        in_frame = (np.abs(pos_x) < boundary_threshold) & (np.abs(pos_y) < boundary_threshold) & (depth_vals < depth_max)
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
            move = np.sqrt((pos_x[seg_end]-pos_x[seg_start])**2 + (pos_y[seg_end]-pos_y[seg_start])**2)
            if seg_len >= 10 and move > 0.2:
                long_tracks.append({
                    'sample': i,
                    'slot': s,
                    'seg_start': seg_start,
                    'seg_end': seg_end,
                    'seg_len': seg_len,
                    'pos_x': pos_x[seg_start:seg_end+1],
                    'pos_y': pos_y[seg_start:seg_end+1],
                    'depth': depth_vals[seg_start:seg_end+1],
                    'total_move': move,
                })

print(f"Long tracks (>=10 frames, moving): {len(long_tracks)}")
long_tracks.sort(key=lambda t: t['total_move'], reverse=True)

# 选5个有代表性的
selected_long = []
# 确保不同样本
used = set()
for t in long_tracks:
    if t['sample'] not in used and len(selected_long) < 5:
        selected_long.append(t)
        used.add(t['sample'])

print(f"Selected {len(selected_long)} long tracks:")
for j, t in enumerate(selected_long):
    print(f"  [{j}] s{t['sample']}_sl{t['slot']} frames={t['seg_start']}-{t['seg_end']}({t['seg_len']}) move={t['total_move']:.3f}")

np.save('/autodl-fs/data/SlotHWM/pos_depth_debug/long_tracks.npy', selected_long, allow_pickle=True)
np.save('/autodl-fs/data/SlotHWM/pos_depth_debug/strict_segments.npy', top_segs, allow_pickle=True)
