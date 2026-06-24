"""
重新收集轨迹数据，排除 fg→bg / bg→fg 的突变 slot
只保留在整个 16 帧中一直是前景（depth < depth_max）的物体
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')

import torch
import numpy as np
import yaml
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset

with open('/autodl-fs/data/SlotHWM/config/obj3d.yaml') as f:
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

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=1, shuffle=True, num_workers=0)

all_data = []

for i in range(50):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = model(frames)
    
    corrected = out['slots']['corrected']  # (1, burnin, N, D)
    target = out['slots']['target']  # (1, rollout, N, D)
    all_slots = torch.cat([corrected, target], dim=1)[0]  # (16, N, D)
    
    N = all_slots.shape[1]
    for s in range(N):
        depth_vals = all_slots[:, s, app_dim+2].cpu().numpy()
        
        # 核心过滤: 整个 16 帧都必须是前景 (depth < depth_max)
        if not np.all(depth_vals < depth_max):
            continue
        
        pos_x = all_slots[:, s, app_dim].cpu().numpy()
        pos_y = all_slots[:, s, app_dim+1].cpu().numpy()
        total_move = np.sqrt((pos_x[-1]-pos_x[0])**2 + (pos_y[-1]-pos_y[0])**2)
        
        all_data.append({
            'sample_idx': i,
            'slot_idx': s,
            'pos_x': pos_x,
            'pos_y': pos_y,
            'depth': depth_vals,
            'total_move': total_move,
        })

print(f"Stable foreground objects (all 16 frames fg): {len(all_data)}")

moving = [d for d in all_data if d['total_move'] > 0.1]
static_fg = [d for d in all_data if d['total_move'] <= 0.1]
print(f"Moving: {len(moving)}, Static: {len(static_fg)}")

# 分类
for d in all_data:
    dx = d['pos_x'][-1] - d['pos_x'][0]
    dy = d['pos_y'][-1] - d['pos_y'][0]
    if d['total_move'] > 0.1:
        d['direction'] = 'horizontal' if abs(dx) > abs(dy) else 'vertical'
    else:
        d['direction'] = 'static'
    dd = d['depth'][-1] - d['depth'][0]
    d['depth_change'] = 'increasing' if dd > 0.01 else ('decreasing' if dd < -0.01 else 'stable')

# 选 10-20 个: 运动的覆盖不同模式 + 少量静止
selected = []
used_samples = set()

# 按模式选运动物体
for direction in ['horizontal', 'vertical']:
    for depth_change in ['increasing', 'decreasing', 'stable']:
        candidates = [d for d in moving 
                      if d['direction'] == direction and d['depth_change'] == depth_change
                      and d['sample_idx'] not in used_samples]
        if candidates:
            candidates.sort(key=lambda d: d['total_move'], reverse=True)
            selected.append(candidates[0])
            used_samples.add(candidates[0]['sample_idx'])

# 补充运动物体到 12 个
remaining = [d for d in moving if d not in selected]
remaining.sort(key=lambda d: d['total_move'], reverse=True)
while len(selected) < 12 and remaining:
    selected.append(remaining.pop(0))

# 静止前景
for d in static_fg[:4]:
    if d['sample_idx'] not in used_samples or len(selected) < 16:
        selected.append(d)
        used_samples.add(d['sample_idx'])
    if len(selected) >= 16:
        break

print(f"\nSelected {len(selected)} objects:")
for j, d in enumerate(selected):
    print(f"  [{j}] s{d['sample_idx']}_sl{d['slot_idx']} move={d['total_move']:.3f} "
          f"dir={d['direction']} depth_ch={d['depth_change']} "
          f"depth=[{d['depth'].min():.3f},{d['depth'].max():.3f}]")

np.save('/autodl-fs/data/SlotHWM/pos_depth_debug/selected_objects_v2.npy', selected, allow_pickle=True)
