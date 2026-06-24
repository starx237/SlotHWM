"""
实验9: 最根本的分析 - target 的 pos/depth 变化模式
假设: 每个物体的运动模式（速度、方向）完全由 appearance 决定。
而 freeze_C=True 时 predictor 看不到 appearance（只有 Z^d = 3维 pos/depth），
所以无法区分不同物体，无法预测各自不同的运动方向。

验证:
1. 不同 slot 的 pos delta 是否有显著差异？
2. 同一 slot 在不同 batch 的 pos delta 是否一致？
3. 如果只用 Z^d 输入，能否从 3 维信息推断出正确的 delta？
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
loader = ds.get_dataloader(batch_size=4, shuffle=True, num_workers=0)

app_dim = cfg.appearance_dim

# 收集多个 batch 的 pos delta 数据
all_deltas = []  # (pos_x_delta, pos_y_delta, depth_delta, pos_x, pos_y, depth, slot_idx)
all_burnin_end = []

for i in range(20):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = model(frames)
    
    target = out['slots']['target']  # (B, rollout, N, D)
    corrected = out['slots']['corrected']  # (B, burnin, N, D)
    burnin_last = corrected[:, -1]  # (B, N, D)
    
    for b in range(target.shape[0]):
        for s in range(target.shape[2]):
            # burnin 末帧的 Z^d
            zd = burnin_last[b, s, app_dim:].cpu().numpy()
            # 第一帧 target 的 Z^d
            td = target[b, 0, s, app_dim:].cpu().numpy()
            delta = td - zd
            all_deltas.append({
                'pos_x': zd[0], 'pos_y': zd[1], 'depth': zd[2],
                'delta_px': delta[0], 'delta_py': delta[1], 'delta_d': delta[2],
                'slot': s, 'batch': i
            })
            all_burnin_end.append(zd)

deltas = np.array([[d['pos_x'], d['pos_y'], d['depth'],
                     d['delta_px'], d['delta_py'], d['delta_d']] for d in all_deltas])

print(f"=== Delta statistics ({len(deltas)} samples) ===")
print(f"  pos_x delta: mean={deltas[:,3].mean():.6f} std={deltas[:,3].std():.6f} range=[{deltas[:,3].min():.4f}, {deltas[:,3].max():.4f}]")
print(f"  pos_y delta: mean={deltas[:,4].mean():.6f} std={deltas[:,4].std():.6f} range=[{deltas[:,4].min():.4f}, {deltas[:,4].max():.4f}]")
print(f"  depth delta: mean={deltas[:,5].mean():.6f} std={deltas[:,5].std():.6f} range=[{deltas[:,5].min():.4f}, {deltas[:,5].max():.4f}]")

# 核心问题: 给定 (pos_x, pos_y, depth)，能否预测 (delta_px, delta_py, delta_d)?
# 如果 delta 和 position 之间没有相关性，那纯 Z^d 输入无法学到东西
print("\n=== Correlation: position → delta ===")
from numpy import corrcoef
for i, name in enumerate(['pos_x', 'pos_y', 'depth']):
    for j, dname in enumerate(['delta_px', 'delta_py', 'delta_d']):
        c = corrcoef(deltas[:, i], deltas[:, j+3])[0, 1]
        if abs(c) > 0.05:
            print(f"  {name} → {dname}: r={c:.4f}")

# 按 slot 分组看
print("\n=== Per-slot delta statistics ===")
for s in range(6):
    mask = np.array([d['slot'] == s for d in all_deltas])
    if mask.sum() == 0: continue
    sd = deltas[mask]
    print(f"  Slot {s} (n={mask.sum()}): "
          f"delta_px mean={sd[:,3].mean():.4f} std={sd[:,3].std():.4f}, "
          f"delta_py mean={sd[:,4].mean():.4f} std={sd[:,4].std():.4f}, "
          f"delta_d mean={sd[:,5].mean():.4f} std={sd[:,5].std():.4f}")

# 关键: 按 (pos_x, pos_y, depth) 的相似位置聚类，看 delta 是否一致
# 如果同一位置的 delta 方差很大 → 无法仅从位置预测运动
print("\n=== Can position predict motion? ===")
# 按 depth 范围分组 (前景 vs 背景)
fg_mask = deltas[:, 2] < 0.3  # depth < 0.3 是前景
bg_mask = ~fg_mask
print(f"  Foreground (depth<0.3): n={fg_mask.sum()}, "
      f"delta_px std={deltas[fg_mask,3].std():.4f}, "
      f"delta_py std={deltas[fg_mask,4].std():.4f}, "
      f"delta_d std={deltas[fg_mask,5].std():.4f}")
print(f"  Background (depth>=0.3): n={bg_mask.sum()}, "
      f"delta_px std={deltas[bg_mask,3].std():.4f}, "
      f"delta_py std={deltas[bg_mask,4].std():.4f}, "
      f"delta_d std={deltas[bg_mask,5].std():.4f}")

# 最关键: 同一 slot index 在不同场景中的 delta 是否一致？
# 如果不一致，说明 slot index 不提供物体身份信息
print("\n=== Same slot index, different scenes: delta consistency ===")
for s in range(6):
    mask = np.array([d['slot'] == s for d in all_deltas])
    sd = deltas[mask]
    if len(sd) < 10: continue
    # 按前 10 个样本看
    print(f"  Slot {s} first 5 deltas (px, py, d):")
    for k in range(min(5, len(sd))):
        print(f"    [{sd[k,3]:.4f}, {sd[k,4]:.4f}, {sd[k,5]:.4f}] (pos=[{sd[k,0]:.2f},{sd[k,1]:.2f},{sd[k,2]:.2f}])")
