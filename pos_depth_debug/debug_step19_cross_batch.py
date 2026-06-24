"""
实验19: 分析跨 batch 的 target pos/depth 分布差异
如果不同 batch 的 target 差异太大，模型会不断遗忘之前学到的模式
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

# 收集 50 个 batch 的 target delta (burnin_last → target_frame0)
deltas_per_batch = []
for i in range(50):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = model(frames)
    
    burnin_last = out['slots']['corrected'][0, -1, :, app_dim:]  # (N, 3)
    target_0 = out['slots']['target'][0, 0, :, app_dim:]  # (N, 3)
    delta = target_0 - burnin_last  # (N, 3)
    deltas_per_batch.append(delta.cpu().numpy())

deltas = np.array(deltas_per_batch)  # (50, N, 3)

print("=== Target delta (burnin→frame0) per slot across batches ===")
N = deltas.shape[1]
for s in range(N):
    d = deltas[:, s, :]  # (50, 3)
    print(f"  Slot {s}: delta_px mean={d[:,0].mean():.4f} std={d[:,0].std():.4f}, "
          f"delta_py mean={d[:,1].mean():.4f} std={d[:,1].std():.4f}, "
          f"delta_d mean={d[:,2].mean():.4f} std={d[:,2].std():.4f}")

# 关键: 前景 slot 和背景 slot 的 delta 方差
# 前景: depth < 0.3 → delta 大
# 背景: depth >= 0.3 → delta 小
print("\n=== Foreground vs Background delta variance ===")
# 用 burnin_last 的 depth 判断前景/背景
fg_deltas = []
bg_deltas = []
for i in range(50):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = model(frames)
    burnin_last = out['slots']['corrected'][0, -1]
    depth = burnin_last[:, app_dim+2].cpu().numpy()
    target_0 = out['slots']['target'][0, 0, :, app_dim:]
    delta = (target_0 - burnin_last[:, app_dim:]).cpu().numpy()
    
    fg_mask = depth < 0.3
    bg_mask = ~fg_mask
    if fg_mask.any():
        fg_deltas.append(delta[fg_mask])
    if bg_mask.any():
        bg_deltas.append(delta[bg_mask])

fg_all = np.concatenate(fg_deltas, axis=0)
bg_all = np.concatenate(bg_deltas, axis=0)

print(f"  Foreground (n={len(fg_all)}): delta_px std={fg_all[:,0].std():.4f}, delta_py std={fg_all[:,1].std():.4f}, delta_d std={fg_all[:,2].std():.4f}")
print(f"  Background (n={len(bg_all)}): delta_px std={bg_all[:,0].std():.4f}, delta_py std={bg_all[:,1].std():.4f}, delta_d std={bg_all[:,2].std():.4f}")

# 核心问题: 给定 burnin_last 的 (pos_x, pos_y, depth)，
# 不同 batch 的 delta 方差有多大？
# 如果很大，说明同一位置可能有不同的运动方向 → 无法仅从位置预测
print("\n=== Conditional delta variance: same position → different delta? ===")
# 按 depth 分组
depth_bins = [(0, 0.15), (0.15, 0.3), (0.3, 0.6), (0.6, 1.0)]
for lo, hi in depth_bins:
    mask = (deltas[:, :, 2] > lo) & (deltas[:, :, 2] <= hi)  # 用 burnin depth
    # 不对，deltas 是 delta 不是 depth... 需要用 burnin_last 的 depth
    pass

# 用第一个 batch 的数据看
print("\n=== Actual issue: burnin Z^d buffer content ===")
# predictor 的输入是 (cur_Z^d, Z^d_buffer, C)
# cur_Z^d = burnin 最后一帧的 (pos_x, pos_y, depth) — 3个数
# Z^d_buffer = 历史 6 帧的 (pos_x, pos_y, depth) — 18个数
# C = global appearance — 64维 (但 freeze_C 时不参与 spatiotemporal 模块)

# 关键: spatiotemporal 模块的输入只有 Z^d (3维 per slot)
# 但它可以做空间注意力（slot 之间交互）和时间注意力（从 buffer 聚合）

# 从 buffer 可以计算速度: delta_pos = pos[t] - pos[t-1]
# 让我检查: 从 buffer 计算出的速度是否和 target 的 delta 一致

batch = next(iter(loader))
frames = batch['video'].cuda()
with torch.no_grad():
    out = model(frames)

# burnin 的 Z^d 序列
burnin_dyn = out['slots']['corrected'][0, :, :, app_dim:]  # (burnin, N, 3)
print(f"Burnin Z^d shape: {burnin_dyn.shape}")

# 从 burnin 计算速度 (最后两帧的差)
velocity = burnin_dyn[-1] - burnin_dyn[-2]  # (N, 3)
target_delta = out['slots']['target'][0, 0, :, app_dim:] - burnin_dyn[-1]  # (N, 3)

print(f"\nVelocity (last 2 burnin frames) vs target delta:")
for s in range(burnin_dyn.shape[1]):
    v = velocity[s].cpu().numpy()
    td = target_delta[s].cpu().numpy()
    # 相对误差
    rel_err = np.abs(v - td) / (np.abs(td) + 1e-6)
    print(f"  slot {s}: vel=({v[0]:.4f},{v[1]:.4f},{v[2]:.4f}), "
          f"target=({td[0]:.4f},{td[1]:.4f},{td[2]:.4f}), "
          f"rel_err=({rel_err[0]:.2f},{rel_err[1]:.2f},{rel_err[2]:.2f})")

# 如果速度 ≈ target delta，那模型只需学会"复制速度"
# 如果不等，模型需要更复杂的预测
