#!/usr/bin/env python3
# C 交换可解释性测试
# 在 rollout 阶段交换两个前景 slot 的静态特征 C，观察模型预测的变化
# Usage: python scripts/interpret_swap_c.py --config config/interpret_obj3d.yaml --checkpoint experiments/obj3d/checkpoints/best.pt

import os, sys, argparse, yaml
import torch
import torch.nn.functional as F
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset
from scripts.interpret_utils import run_interpret_swap, visualize_swap_result


def main():
    parser = argparse.ArgumentParser(description='C-Swap Interpretability Test')
    parser.add_argument('--config', default='config/interpret_obj3d.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--workdir', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--swap_pairs', type=int, nargs=2, action='append', default=None,
                        help='Manual swap pair (e.g. --swap_pairs 1 2). Auto-selected if not given.')
    parser.add_argument('--debug', action='store_true', help='Print debug info')
    parser.add_argument('--mode', choices=['swap', 'ablate'], default='swap',
                        help="'swap'=交换 C, 'ablate'=将指定 slot 的 C 置零")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg_dict)
    workdir = args.workdir or getattr(cfg, 'workdir', './experiments/interpret_obj3d')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SlotDynamicsModel(cfg).to(device)
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
    missing = set(model_state.keys()) - set(matched.keys())
    if missing:
        print(f"  Missing keys: {len(missing)} (non-encoder/decoder components)")
    model.load_state_dict(matched, strict=False)
    model.eval()
    burnin = getattr(cfg, 'burnin_frames', 6)
    rollout = getattr(cfg, 'rollout_frames', 10)
    subsample = getattr(cfg, 'subsample', 1)
    print(f"Loaded: {args.checkpoint}")
    print(f"Device: {device}, burnin={burnin}, rollout={rollout}")

    ds = OBJ3DDataset(data_path=cfg_dict.get('data_root', './data/obj3d'),
                       num_frames=burnin + rollout,
                       stride=cfg_dict.get('slide_stride', 5),
                       subsample=subsample)
    loader = ds.get_dataloader(batch_size=1, shuffle=True, num_workers=0)

    viz_dir = os.path.join(workdir, 'eval_images', 'interpret_swap')
    os.makedirs(viz_dir, exist_ok=True)

    results = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_samples:
                break
            frames = batch["video"].to(device)
            if frames.shape[1] < burnin + rollout:
                continue
            frames = frames[:, :burnin + rollout]

            swap_pairs = args.swap_pairs
            result = run_interpret_swap(model, frames, swap_pairs, burnin, rollout,
                                        device, debug=args.debug, mode=args.mode)
            results.append(result)

    for s, res in enumerate(results):
        target = res["video_target"]
        normal = res["video_normal"]
        modified = res["video_modified"]
        bg = res["bg_idx"]
        pairs = res["swap_pairs"]
        mode = res.get("mode", "swap")

        print(f"\n{'='*60}")
        print(f"Sample {s}: BG slot={bg}, mode={mode}, target_slots={pairs}")
        print(f"{'='*60}")
        header = f"{'Step':>6} {'N_GT_MSE':>12} {'M_GT_MSE':>12} {'D_GT':>10}   {'N_M_MSE':>12}"
        print(header)
        print("-" * 55)
        total_n, total_m, total_nm = 0.0, 0.0, 0.0
        for t in range(rollout):
            mn = F.mse_loss(normal[0, t], target[0, t]).item()
            mm = F.mse_loss(modified[0, t], target[0, t]).item()
            mnm = F.mse_loss(normal[0, t], modified[0, t]).item()
            total_n += mn
            total_m += mm
            total_nm += mnm
            print(f"{t:>6d} {mn:>12.6f} {mm:>12.6f} {mm-mn:>+10.6f}   {mnm:>12.8f}")
        avg_n = total_n / rollout
        avg_m = total_m / rollout
        avg_nm = total_nm / rollout
        print("-" * 55)
        print(f"{'Avg':>6} {avg_n:>12.6f} {avg_m:>12.6f} {avg_m-avg_n:>+10.6f}   {avg_nm:>12.8f}")

        out_path = os.path.join(viz_dir, f'sample_{s}.png')
        visualize_swap_result(res, burnin, rollout, out_path)
        print(f"Saved: {out_path}")

    print(f"\nDone. Results in {viz_dir}")


if __name__ == '__main__':
    main()
