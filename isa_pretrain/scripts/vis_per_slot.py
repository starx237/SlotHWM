#!/usr/bin/env python3
import os, sys, argparse, yaml
import torch
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from isa_pretrain.models.isa_dynamics import SlotDynamicsModelISA
from data.obj3d_dataset import OBJ3DDataset
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description='ISA Per-Slot Visualization')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default='isa_pretrain/config/isa_obj3d.yaml')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg_dict)
    workdir = getattr(cfg, 'workdir', './experiments/isa_obj3d')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SlotDynamicsModelISA(cfg).to(device)
    sd = torch.load(args.checkpoint, map_location=device)
    ckpt = sd.get('model', sd)
    model_state = model.state_dict()
    matched = {}
    for mk in model_state:
        mk_clean = mk.replace('_orig_mod.', '')
        if mk_clean in ckpt:
            matched[mk] = ckpt[mk_clean]
        elif mk in ckpt:
            matched[mk] = ckpt[mk]
    model.load_state_dict(matched, strict=False)
    model.eval()
    print(f"Loaded: {args.checkpoint}  Device: {device}")

    burnin = getattr(cfg, 'burnin_frames', 6)
    ds = OBJ3DDataset(data_path=cfg_dict.get('data_root', './data/obj3d'),
                       num_frames=burnin, stride=cfg_dict.get('slide_stride', 4),
                       subsample=getattr(cfg, 'subsample', 1))
    loader = ds.get_dataloader(batch_size=1, shuffle=True, num_workers=0)

    out_dir = args.out or os.path.join(workdir, 'eval_images', 'per_slot')
    os.makedirs(out_dir, exist_ok=True)

    font = ImageFont.load_default()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_samples:
                break
            frames = batch["video"].to(device)
            feat = model._encode_features(frames)

            slots = None
            for t in range(feat.shape[1]):
                slots, attn = model.slot_attention(feat[:, t], slots)

            N = slots.shape[1]
            S = 64
            rows = N + 2
            W = S * 2 + 8
            H = rows * S + 14 * N
            canvas = Image.new('RGB', (W, H), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)

            blended, alphas = model.decoder(slots, return_alphas=True)

            def to_pil(t):
                a = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                return Image.fromarray((a * 255).astype('uint8'))

            def put_rgb(t, y, x):
                canvas.paste(to_pil(t), (x, y))

            put_rgb(frames[0, 0], 0, 0)
            draw.text((2, 2), 'GT', fill=(0, 0, 0), font=font)

            put_rgb(blended[0], 0, S + 8)
            draw.text((S + 10, 2), 'Blended', fill=(0, 0, 0), font=font)

            for j in range(N):
                y = (j + 2) * S + 14 * j

                slot_only = torch.cat([
                    slots[:, j:j+1, :-4],
                    slots[:, j:j+1, -4:-2],
                    slots[:, j:j+1, -2:],
                ], dim=-1)
                rgb_j = model.decoder(slot_only, return_alphas=False)[0]

                alpha_j = alphas[0, j]
                alpha_rgb = alpha_j.expand(3, -1, -1)

                put_rgb(rgb_j, y, 0)
                put_rgb(alpha_rgb, y, S + 8)

                pos = slots[0, j, -4:-2]
                sc = slots[0, j, -2:]
                draw.text((2, y + S + 2),
                          f'Slot {j}  pos({pos[0]:+.3f},{pos[1]:+.3f})  sc({sc[0]:.3f},{sc[1]:.3f})',
                          fill=(0, 0, 0), font=font)

            out_path = os.path.join(out_dir, f'sample_{i}.png')
            canvas.save(out_path)
            print(f"Saved: {out_path}")

    print(f"Done. Results in {out_dir}")


if __name__ == '__main__':
    main()
