"""
数据分析: 收集多个样本中前景物体的 pos_x, pos_y, depth 在 16 帧中的变化
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
burnin = cfg.burnin_frames

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

# 收集所有 16 帧的 slots
# 需要: burnin(6帧) + rollout(10帧) = 16帧
# 但 rollout 的 target 是用 ISA 重新编码的

all_data = []  # 每个 entry: {sample_idx, slot_idx, pos_x[16], pos_y[16], depth[16], is_fg}

for i in range(30):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = model(frames)
    
    corrected = out['slots']['corrected']  # (1, burnin, N, D)
    target = out['slots']['target']  # (1, rollout, N, D)
    
    # 合并 16 帧
    all_slots = torch.cat([corrected, target], dim=1)  # (1, 16, N, D)
    all_slots = all_slots[0]  # (16, N, D)
    
    N = all_slots.shape[1]
    for s in range(N):
        depth_vals = all_slots[:, s, app_dim+2].cpu().numpy()
        # 判断是否为前景: depth 在大部分帧 < 0.3
        if np.mean(depth_vals) < 0.3:
            # 检查是否真的在运动: pos 变化 > 阈值
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
                'is_moving': total_move > 0.1,
            })

print(f"Total foreground objects: {len(all_data)}")
moving = [d for d in all_data if d['is_moving']]
static_fg = [d for d in all_data if not d['is_moving']]
print(f"Moving foreground: {len(moving)}")
print(f"Static foreground: {len(static_fg)}")

# 选择 10-20 个典型物体:
# - 8-10 个运动的，覆盖不同运动模式
# - 3-5 个静止前景
# - 排除同一 sample 中重复的运动模式

# 按运动模式分类
for d in moving:
    # 运动方向
    dx = d['pos_x'][-1] - d['pos_x'][0]
    dy = d['pos_y'][-1] - d['pos_y'][0]
    if abs(dx) > abs(dy):
        d['direction'] = 'horizontal'
    else:
        d['direction'] = 'vertical'
    # depth 变化
    dd = d['depth'][-1] - d['depth'][0]
    d['depth_change'] = 'increasing' if dd > 0.01 else ('decreasing' if dd < -0.01 else 'stable')

print(f"\nMoving objects by direction: horizontal={sum(1 for d in moving if d['direction']=='horizontal')}, vertical={sum(1 for d in moving if d['direction']=='vertical')}")
print(f"Depth changes: increasing={sum(1 for d in moving if d['depth_change']=='increasing')}, decreasing={sum(1 for d in moving if d['depth_change']=='decreasing')}, stable={sum(1 for d in moving if d['depth_change']=='stable')}")

# 选择代表性物体
selected = []

# 运动物体: 选不同运动方向和深度变化的
for direction in ['horizontal', 'vertical']:
    for depth_change in ['increasing', 'decreasing', 'stable']:
        candidates = [d for d in moving if d['direction'] == direction and d['depth_change'] == depth_change]
        if candidates:
            # 选运动幅度最大的
            candidates.sort(key=lambda d: d['total_move'], reverse=True)
            selected.append(candidates[0])

# 确保选了足够的运动物体
remaining_moving = [d for d in moving if d not in selected]
remaining_moving.sort(key=lambda d: d['total_move'], reverse=True)
while len(selected) < 12 and remaining_moving:
    selected.append(remaining_moving.pop(0))

# 静止前景
static_fg.sort(key=lambda d: d['total_move'])
for d in static_fg[:4]:
    selected.append(d)

print(f"\nSelected {len(selected)} objects:")
for j, d in enumerate(selected):
    print(f"  [{j}] sample={d['sample_idx']} slot={d['slot_idx']} move={d['total_move']:.3f} dir={d.get('direction','static')} depth_ch={d.get('depth_change','N/A')}")

# 保存数据
np.save('/autodl-fs/data/SlotHWM/pos_depth_debug/selected_objects.npy', selected, allow_pickle=True)
