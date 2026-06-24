"""
实验2+4: 分析 target slots 的 pos/depth 数值特征，以及 rollout 过程中的传播情况
关键问题: pred_S 和 target_S 的 pos/depth 之间到底差什么？
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')

import torch
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
torch.manual_seed(42)
batch = next(iter(ds.get_dataloader(batch_size=4, shuffle=True, num_workers=0)))
frames = batch['video'].cuda()

with torch.no_grad():
    out = model(frames)

app_dim = cfg.appearance_dim
rollout = cfg.rollout_frames
burnin = cfg.burnin_frames

# 分析 target slots 的 Z^d（pos, depth）
target_S = out['slots']['target']  # (B, rollout, N, D)
corrected_S = out['slots']['corrected']  # (B, burnin, N, D)
pred_S = out['slots']['predicted']

print("=== Target slots Z^d (pos_x, pos_y, depth) statistics ===")
print(f"Target shape: {target_S.shape}")
print(f"Corrected (burnin) shape: {corrected_S.shape}")

# Burnin 最后一帧的 Z^d
burnin_last = corrected_S[:, -1]  # (B, N, D)
print(f"\n--- Burnin last frame Z^d ---")
for i in range(burnin_last.shape[1]):
    px = burnin_last[0, i, app_dim].item()
    py = burnin_last[0, i, app_dim+1].item()
    d = burnin_last[0, i, app_dim+2].item()
    print(f"  slot {i}: pos_x={px:.4f}, pos_y={py:.4f}, depth={d:.4f}")

# Target rollout 每帧的 Z^d
print(f"\n--- Target rollout Z^d per frame (batch 0) ---")
for t in range(min(5, rollout)):
    s = target_S[0, t]
    print(f"  frame {t}:")
    for i in range(s.shape[0]):
        px = s[i, app_dim].item()
        py = s[i, app_dim+1].item()
        d = s[i, app_dim+2].item()
        print(f"    slot {i}: pos_x={px:.4f}, pos_y={py:.4f}, depth={d:.4f}")

# 预测 rollout 每帧的 Z^d (初始化后，zero_init → pred=copy of current)
print(f"\n--- Predicted rollout Z^d per frame (batch 0) ---")
for t in range(min(5, rollout)):
    s = pred_S[0, t]
    print(f"  frame {t}:")
    for i in range(s.shape[0]):
        px = s[i, app_dim].item()
        py = s[i, app_dim+1].item()
        d = s[i, app_dim+2].item()
        print(f"    slot {i}: pos_x={px:.4f}, pos_y={py:.4f}, depth={d:.4f}")

# Target 的逐帧 delta
print(f"\n--- Target Z^d per-frame delta (batch 0, slot 0-5) ---")
for i in range(min(6, target_S.shape[2])):
    print(f"  slot {i}:")
    for t in range(min(5, rollout)):
        if t == 0:
            prev = corrected_S[0, -1, i, app_dim:]
        else:
            prev = target_S[0, t-1, i, app_dim:]
        cur = target_S[0, t, i, app_dim:]
        delta = cur - prev
        print(f"    frame {t}: delta_pos=({delta[0]:.4f}, {delta[1]:.4f}), delta_depth={delta[2]:.4f}")

# 关键问题: depth_mask 有多少 slot 被遮住了？
print(f"\n--- Depth mask analysis ---")
dm = out['depth_mask']  # (B, rollout, N)
print(f"Total slots: {dm.numel()}")
print(f"Masked (valid) slots: {dm.sum().item()}")
print(f"Mask ratio: {dm.float().mean().item():.4f}")
print(f"Per-frame mask ratio:")
for t in range(rollout):
    print(f"  frame {t}: {dm[:, t].float().mean().item():.4f}")

# 哪些 slot 被 mask 了？(per slot index)
print(f"\nPer-slot mask ratio across all frames:")
for i in range(dm.shape[2]):
    ratio = dm[:, :, i].float().mean().item()
    print(f"  slot {i}: {ratio:.4f}")
