#!/usr/bin/env python3
# Baseline 可视化：原帧 vs 重建 vs rollout 预测，逐帧 MSE
# Usage: python scripts/visualize_baseline.py --checkpoint experiments/baseline_clevrer/checkpoints/best.pt

import os, sys, argparse, yaml
import torch
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
from dotenv import load_dotenv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.baseline_model import BaselineModel
from data import get_dataset
from train.trainer import WandBLogger


def load_wandb_config():
    load_dotenv()
    enabled = os.getenv('WANDB_ENABLED', 'false').lower() == 'true'
    return {
        'enabled': enabled,
        'api_key': os.getenv('WANDB_API_KEY', ''),
        'project': os.getenv('WANDB_PROJECT', 'slotpi'),
        'entity': os.getenv('WANDB_ENTITY', ''),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/baseline_clevrer.yaml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--workdir', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=2)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg_dict)
    workdir = args.workdir or getattr(cfg, 'workdir', './experiments/baseline_default')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wb_cfg = load_wandb_config()
    wandb_logger = WandBLogger(enabled=wb_cfg['enabled'])
    if wb_cfg['enabled']:
        if wb_cfg['api_key']:
            os.environ['WANDB_API_KEY'] = wb_cfg['api_key']
        import wandb
        wandb_logger.init(
            project=wb_cfg['project'],
            entity=wb_cfg['entity'] or None,
            name=f"viz_baseline_{os.path.basename(args.checkpoint or 'latest')}",
            config=cfg_dict,
        )

    model = BaselineModel(cfg).to(device)
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(sd.get('model', sd))
        print(f"Loaded: {args.checkpoint}")
    model.eval()

    num_frames = getattr(cfg, 'num_frames', None) or (cfg.burnin_frames + cfg.rollout_frames)
    slide_stride = getattr(cfg, 'slide_stride', 1)
    ds = get_dataset(cfg.dataset, data_path=cfg.data_root,
                     num_frames=num_frames, stride=slide_stride)
    loader = ds.get_dataloader(batch_size=1, shuffle=True, num_workers=0)
    burnin, rollout = cfg.burnin_frames, cfg.rollout_frames

    viz_dir = os.path.join(workdir, 'eval_images', 'step_viz')
    os.makedirs(viz_dir, exist_ok=True)

    samples = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_samples:
                break
            frames = batch["video"].to(device)
            out = model(frames)
            samples.append((out, frames))

    for s, (out, frames) in enumerate(samples):
        B, T, C, H, W = frames.shape
        dec_b = out["outputs"]["video_burnin"]  # [1, burnin, 3, H, W]
        dec_p = out["outputs"]["video_pred"]    # [1, rollout, 3, H, W]
        gt_r = frames[:, burnin:burnin + rollout]

        # 逐帧 MSE
        for t in range(burnin):
            mse = F.mse_loss(dec_b[:, t], frames[:, t]).item()
            print(f"  sample {s} burnin step {t:2d}: {mse:.6f}")
        for t in range(rollout):
            mse = F.mse_loss(dec_p[:, t], gt_r[:, t]).item()
            print(f"  sample {s} rollout step {t:2d}: {mse:.6f}")

        try:
            from PIL import Image, ImageDraw, ImageFont
            S = 70
            rows = 4 + rollout + 1
            cols = max(burnin, rollout)
            font = ImageFont.load_default()
            canvas = Image.new('RGB', (cols * S, rows * S), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)

            def pt(t, row, label=None, mse_val=None):
                arr = t.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                im = Image.fromarray((arr * 255).astype(np.uint8))
                canvas.paste(im, (0, row * S))
                if label:
                    draw.text((2, row * S + 2), label, fill=(0, 0, 0), font=font)
                if mse_val is not None:
                    draw.text((2, row * S + S - 12), f"{mse_val:.4f}", fill=(0, 0, 0), font=font)

            for t in range(burnin):
                pt(frames[0, t], 0, "GT Burnin")
                pt(dec_b[0, t], 1, "Recon Burnin",
                   F.mse_loss(dec_b[0, t], frames[0, t]).item())
            for t in range(rollout):
                pt(gt_r[0, t], 2, "GT Rollout")
                pt(dec_p[0, t], 3 + t, f"Pred step {t}",
                   F.mse_loss(dec_p[0, t], gt_r[0, t]).item())

            out_path = os.path.join(viz_dir, f'sample_{s}.png')
            canvas.save(out_path)
            print(f"\nSaved {out_path}")

            if wb_cfg['enabled']:
                wandb_logger.log({f"eval/recon_{s}": wandb.Image(out_path)})
        except Exception as e:
            print(f"\nImage save skipped (no PIL?): {e}")

    wandb_logger.finish()


if __name__ == '__main__':
    main()
