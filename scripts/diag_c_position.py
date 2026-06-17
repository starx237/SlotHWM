#!/usr/bin/env python3
"""诊断：C (Z^c) 中是否含有位置编码信息。
方法：
  1. 加载 checkpoint
  2. 在 burnin 后提取 C = mean(Z^c)
  3. 比较两个 slot 的 C 差异 vs 位置差异
  4. 如果位置相同但 C 差异大 → C 不含位置
     如果位置不同时 C 差异也大 → C 可能含有位置
"""
import os, sys, yaml, torch
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

cfg = SimpleNamespace(**yaml.safe_load(open('config/interpret_obj3d.yaml')))
model = SlotDynamicsModel(cfg).to(device)
model.eval()

# 找 checkpoint
import glob
ckpts = glob.glob('experiments/obj3d/checkpoints/*.pt')
if not ckpts:
    print("No checkpoint found. Using random init model.")
else:
    ckpt_path = ckpts[0]
    sd = torch.load(ckpt_path, map_location=device)
    ckpt = sd.get('model', sd)
    missing = model.load_state_dict(ckpt, strict=False)
    print(f"Loaded: {ckpt_path}")
    if missing.missing_keys:
        print(f"  Missing keys: {len(missing.missing_keys)}")
    if missing.unexpected_keys:
        print(f"  Unexpected keys: {len(missing.unexpected_keys)}")

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=cfg.burnin_frames + cfg.rollout_frames,
                  stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=1, shuffle=True, num_workers=0)

with torch.no_grad():
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        frames = batch["video"].to(device)
        B, T, C, H, W = frames.shape
        burnin = cfg.burnin_frames

        enc_features = model.encoder(frames)
        _, _, N_feat, _ = enc_features.shape
        grid_sz = int(N_feat ** 0.5)

        # 手动执行 burnin，模拟 _forward_finetune
        slots_list = []
        centroid_list = []
        Z_core_list = []
        p_list = []
        slots = None
        for t in range(burnin):
            feat_t = enc_features[:, t]
            slots, attn = model._sa(feat_t, slots, t)
            centroid = model._compute_slot_centroid(attn, grid_sz)
            Z_core = model.f_z(slots)
            p = model._encode_pos_to_zd(centroid, cfg.pos_enc_dim)
            slots_list.append(slots)
            centroid_list.append(centroid)
            Z_core_list.append(Z_core)
            p_list.append(p)

        slots = torch.stack(slots_list, dim=1)        # (1, Bc, K, 128)
        centroids = torch.stack(centroid_list, dim=1)  # (1, Bc, K, 2)
        Z_core = torch.stack(Z_core_list, dim=1)       # (1, Bc, K, 128)
        p = torch.stack(p_list, dim=1)                 # (1, Bc, K, 8)

        Z_c = Z_core[:, :, :, :cfg.static_dim]         # (1, Bc, K, 120)
        C = Z_c.mean(dim=1)                            # (1, K, 120)

        print(f"\n{'='*60}")
        print(f"Sample {i}:")
        print(f"{'='*60}")
        pos = centroids[0]  # (Bc, K, 2)
        print(f"\nPositions (last burnin frame, slot y,x):")
        for k in range(cfg.num_slots):
            py, px = pos[-1, k, 0].item(), pos[-1, k, 1].item()
            print(f"  Slot {k}: ({py:.3f}, {px:.3f})")

        print(f"\nC norms (||C_k||₂):")
        for k in range(cfg.num_slots):
            cn = C[0, k].norm().item()
            print(f"  Slot {k}: {cn:.4f}")

        print(f"\nC pair differences (mutual MSE):")
        for a in range(cfg.num_slots):
            for b in range(a+1, cfg.num_slots):
                c_diff = (C[0, a] - C[0, b]).square().mean().item()
                pos_a = pos[-1, a]
                pos_b = pos[-1, b]
                pos_dist = ((pos_a - pos_b)**2).sum().sqrt().item()
                print(f"  C diff S{a}-S{b}: {c_diff:.6f}  |  "
                      f"position dist: {pos_dist:.4f}")

        print(f"\nIs C correlated with position? "
              f"Check if large pos_dist → large C_diff:")
        pos_dists = []
        c_diffs = []
        for a in range(cfg.num_slots):
            for b in range(a+1, cfg.num_slots):
                c_diff = (C[0, a] - C[0, b]).square().mean().item()
                pd = ((pos[-1, a] - pos[-1, b])**2).sum().sqrt().item()
                pos_dists.append(pd)
                c_diffs.append(c_diff)
        if max(pos_dists) > 0.1:
            high_pos = [(c, p) for c, p in zip(c_diffs, pos_dists) if p > 0.5]
            low_pos = [(c, p) for c, p in zip(c_diffs, pos_dists) if p < 0.2]
            if high_pos and low_pos:
                avg_high = sum(c for c, _ in high_pos) / len(high_pos)
                avg_low = sum(c for c, _ in low_pos) / len(low_pos)
                print(f"  Avg C diff when positions far apart: {avg_high:.6f}")
                print(f"  Avg C diff when positions close:    {avg_low:.6f}")
                if avg_high > avg_low * 1.5:
                    print(f"  => C IS correlated with position!")
                else:
                    print(f"  => C is NOT strongly correlated with position")
        else:
            print(f"  => All slots have similar positions, can't assess correlation")

print("\nDone")
