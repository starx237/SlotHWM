#!/usr/bin/env python3
# 可视化：原帧 vs 重建 vs rollout 预测，逐帧 MSE
# Usage: python scripts/visualize_rollout.py --checkpoint experiments/obj3d/checkpoints/best.pt

import os, sys, argparse, yaml
import torch
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.slotpi import SlotPi
from data.obj3d_dataset import OBJ3DDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/obj3d.yaml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output', default='rollout_viz.png')
    parser.add_argument('--num_samples', type=int, default=2)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SlotPi(cfg).to(device)
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(sd.get('model', sd))
        print(f"Loaded: {args.checkpoint}")
    model.eval()

    ds = OBJ3DDataset(data_path=cfg_dict.get('data_root','./data/obj3d'),
                       num_frames=cfg.burnin_frames + cfg.rollout_frames,
                       stride=cfg_dict.get('slide_stride', 5))
    batch = next(iter(ds.get_dataloader(batch_size=args.num_samples, shuffle=True, num_workers=0)))
    frames = batch["video"].to(device)
    B, T, C, H, W = frames.shape
    burnin, rollout = cfg.burnin_frames, cfg.rollout_frames

    with torch.no_grad():
        out = model(frames)

    # 解码器输出 (8,8) → 上采样到 64
    def up8(t):
        B_, T_, C_, _, _ = t.shape
        flat = t.reshape(-1, 3, 8, 8)
        up = F.interpolate(flat, size=(64, 64), mode='bilinear', align_corners=False)
        return up.reshape(B_, T_, 3, 64, 64)

    dec_burnin = up8(out["outputs"]["video_burnin"])
    dec_pred   = up8(out["outputs"]["video_pred"])
    gt_rollout = frames[:, burnin:burnin + rollout]

    # 逐帧 MSE
    print("=" * 60)
    print(f"Reconstruction & Prediction MSE (decoder 8x8 → 64x64)")
    print("=" * 60)

    ts = list(range(burnin))
    print(f"\nBurnin frames {ts}:")
    for t in ts:
        mse = F.mse_loss(dec_burnin[:, t], frames[:, t]).item()
        print(f"  step {t:2d}: {mse:.6f}")

    ts = list(range(rollout))
    print(f"\nRollout frames {ts}:")
    for t in ts:
        mse = F.mse_loss(dec_pred[:, t], gt_rollout[:, t]).item()
        print(f"  step {t:2d}: {mse:.6f}")

    print(f"\n  Burnin avg: {F.mse_loss(dec_burnin, frames[:, :burnin]).item():.6f}")
    print(f"  Rollout avg: {F.mse_loss(dec_pred, gt_rollout).item():.6f}")

    # 保存可视化
    try:
        from PIL import Image, ImageDraw, ImageFont
        S = 70
        rows = 4 + rollout + 1  # GT burnin, recon burnin, GT rollout (×1), pred rollout (×rollout)
        cols = max(burnin, rollout)
        font = ImageFont.load_default()

        for s in range(B):
            canvas = Image.new('RGB', (cols * S, rows * S), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)

            def pt(t, row, label=None, mse_val=None):
                arr = t.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                im = Image.fromarray((arr * 255).astype(np.uint8))
                canvas.paste(im, (0, row * S))
                if label:
                    draw.text((2, row * S + 2), label, fill=(0,0,0), font=font)
                if mse_val is not None:
                    draw.text((2, row * S + S - 12), f"{mse_val:.4f}", fill=(0,0,0), font=font)

            for t in range(burnin):
                pt(frames[s, t], 0, "GT Burnin")
            for t in range(burnin):
                pt(dec_burnin[s, t], 1, "Recon Burnin",
                   F.mse_loss(dec_burnin[s, t], frames[s, t]).item())
            for t in range(rollout):
                pt(gt_rollout[s, t], 2, "GT Rollout")
            for t in range(rollout):
                # 复制每列到多行
                pt(dec_pred[s, t], 3 + t, f"Pred step {t}",
                   F.mse_loss(dec_pred[s, t], gt_rollout[s, t]).item())

            out_path = args.output.replace('.png', f'_s{s}.png')
            canvas.save(out_path)
            print(f"\nSaved {out_path}")
    except Exception as e:
        print(f"\nImage save skipped (no PIL?): {e}")


if __name__ == '__main__':
    main()
