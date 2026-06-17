#!/usr/bin/env python3
"""诊断：decoder 是否使用后 32 维的正弦位置编码来控制渲染位置。
方法：
  1. 对样本做 burnin，取最后一帧的 slots_raw 和 PE_32
  2. 正常 decode → 对照图
  3. 将 PE_32 在 slot 维度随机打乱（不改变 slots_raw），再 decode
  4. 对比前后图像差异 → 如果 decoder 使用 PE 做位置，打乱后位置应交换
"""
import os, sys, yaml, torch
import torch.nn.functional as F
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

cfg = SimpleNamespace(**yaml.safe_load(open('config/interpret_obj3d.yaml')))
model = SlotDynamicsModel(cfg).to(device)
model.eval()

import glob
ckpts = glob.glob('experiments/obj3d/checkpoints/*.pt')
if ckpts:
    ckpt_path = ckpts[0]
    sd = torch.load(ckpt_path, map_location=device)
    ckpt = sd.get('model', sd)
    # 处理 _orig_mod 前缀（来自 torch.compile）
    ckpt_clean = {}
    for k, v in ckpt.items():
        ckpt_clean[k.replace('_orig_mod.', '')] = v
    model_state = model.state_dict()
    matched = {}
    for mk in model_state:
        mk_clean = mk.replace('_orig_mod.', '')
        if mk_clean in ckpt_clean:
            matched[mk] = ckpt_clean[mk_clean]
        elif mk in ckpt_clean:
            matched[mk] = ckpt_clean[mk]
    missing_keys = set(model_state.keys()) - set(matched.keys())
    if missing_keys:
        print(f"  Missing keys: {len(missing_keys)} (non-encoder/decoder components)")
    model.load_state_dict(matched, strict=False)
    print(f"Loaded: {ckpt_path} ({len(matched)} keys matched)")
else:
    print("No checkpoint, using random init.")

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=cfg.burnin_frames + 1,
                  stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=1, shuffle=True, num_workers=0)

out_dir = 'eval_images/diag_pe_shuffle'
os.makedirs(out_dir, exist_ok=True)

with torch.no_grad():
    for i, batch in enumerate(loader):
        if i >= 3:
            break
        frames = batch["video"].to(device)
        B, T, C, H, W = frames.shape
        burnin = cfg.burnin_frames

        enc_features = model.encoder(frames)
        _, _, N_feat, _ = enc_features.shape
        grid_sz = int(N_feat ** 0.5)

        # Burnin 取最后一帧
        slots = None
        last_slots, last_attn = None, None
        for t in range(burnin):
            feat_t = enc_features[:, t]
            slots, attn = model._sa(feat_t, slots, t)
            last_slots, last_attn = slots, attn

        # 计算 PE_32
        centroid = model._compute_slot_centroid(last_attn, grid_sz)  # (1, K, 2)
        pe_32 = model._reconstruct_pe(centroid)                     # (1, K, 32)

        # 正常 decode
        slots_pe = last_slots.clone()
        slots_pe[:, :, -32:] += pe_32
        dec_normal, alpha_normal = model.decoder(slots_pe, return_alpha=True)

        # 打乱 PE（在 slot 维度随机置换）
        K = pe_32.shape[1]
        perm = torch.randperm(K, device=device)
        pe_shuffled = pe_32[:, perm, :]
        slots_pe_shuffled = last_slots.clone()
        slots_pe_shuffled[:, :, -32:] += pe_shuffled
        dec_shuffled, alpha_shuffled = model.decoder(slots_pe_shuffled, return_alpha=True)

        # GT 帧
        gt_frame = frames[:, burnin - 1]

        # 保存对比图
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, rows = plt.subplots(4, K + 1, figsize=((K + 1) * 3, 12))
        for r in range(4):
            for c in range(K + 1):
                rows[r, c].axis('off')

        # 第 0 行: RGB GT + 各 slot alpha (normal)
        rows[0, 0].imshow(gt_frame[0].permute(1, 2, 0).cpu().clamp(0, 1))
        rows[0, 0].set_title("GT")
        for k in range(K):
            alpha_k = alpha_normal[0, k, 0].cpu()
            alpha_k = (alpha_k / alpha_k.max()).clamp(0, 1)
            rows[0, k + 1].imshow(alpha_k, cmap='viridis')
            rows[0, k + 1].set_title(f"S{k} α")

        # 第 1 行: Normal decode + slot alpha (normal)
        rows[1, 0].imshow(dec_normal[0].permute(1, 2, 0).cpu().clamp(0, 1))
        rows[1, 0].set_title("Normal Recon")

        # 第 2 行: PE shuffled decode + slot alpha (shuffled)
        rows[2, 0].imshow(dec_shuffled[0].permute(1, 2, 0).cpu().clamp(0, 1))
        rows[2, 0].set_title(f"PE Shuffled (perm={perm.cpu().tolist()})")
        for k in range(K):
            alpha_k = alpha_shuffled[0, k, 0].cpu()
            alpha_k = (alpha_k / alpha_k.max()).clamp(0, 1)
            rows[2, k + 1].imshow(alpha_k, cmap='viridis')
            rows[2, k + 1].set_title(f"S{perm[k].item()} α")

        # 第 3 行: Normal-Shuffled diff ×10
        diff = (dec_normal - dec_shuffled).abs().mean(dim=1, keepdim=True)
        diff_amp = (diff * 10).clamp(0, 1)
        rows[3, 0].imshow(diff_amp[0].permute(1, 2, 0).cpu().squeeze(), cmap='hot')
        rows[3, 0].set_title("|Norm-Shuf|×10")
        mse = F.mse_loss(dec_normal, dec_shuffled).item()
        rows[3, 0].set_xlabel(f"MSE={mse:.6f}")

        # 标注位置信息
        pos = centroid[0]
        for k in range(K):
            py, px = pos[k, 0].item(), pos[k, 1].item()
            perm_k = perm[k].item()
            rows[0, k + 1].set_xlabel(f"pos=({py:.2f},{px:.2f})")
            rows[2, k + 1].set_xlabel(f"pos from S{perm_k}({pos[perm_k,0].item():.2f},{pos[perm_k,1].item():.2f})")

        plt.suptitle(f"Sample {i}: PE Shuffle Diagnostic\n"
                     f"Top: Normal decode | Bottom: PE permuted across slots")
        plt.tight_layout()
        save_path = os.path.join(out_dir, f'sample_{i}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

        # 命令行输出简要统计数据
        print(f"\nSample {i}:")
        print(f"  Positions (y, x):")
        for k in range(K):
            print(f"    Slot {k}: ({pos[k,0].item():.4f}, {pos[k,1].item():.4f})")
        print(f"  Normal vs Shuffled MSE: {mse:.6f}")
        print(f"  Permutation: {perm.cpu().tolist()}")
        if mse > 0.01:
            print(f"  => DECODER IS USING PE FOR POSITION (large MSE)")
        else:
            print(f"  => Decoder may NOT use PE (small MSE)")

print(f"\nDone. Results in {out_dir}/")
