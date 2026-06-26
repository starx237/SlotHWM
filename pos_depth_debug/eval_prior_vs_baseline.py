#!/usr/bin/env python3
"""
公平对比 Baseline (MLP predictor) vs Prior Phase2 的 depth-spread-cov 关系。
3000 样本，每样本取1帧，FG filter 无 a_mean。
两个模型都用 polyfit 线性拟合算 R²，Prior 同时显示 prior 参数预测线。
"""

import torch, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace
import yaml
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset
from torch.utils.data import DataLoader
from train import Trainer, create_optimizer
from train.trainer import WandBLogger


def load_model(cfg_dict, ckpt_path):
    cfg = SimpleNamespace(**cfg_dict)
    model = SlotDynamicsModel(cfg)
    opt, sch = create_optimizer((p for p in model.parameters() if p.requires_grad), cfg)
    wb = WandBLogger(enabled=False)
    trainer = Trainer(model, opt, sch, cfg, wandb_logger=wb)
    trainer.load_checkpoint(ckpt_path)
    return model


def collect_data(model, ds, n_samples=3000, max_frames=1):
    model.eval().cuda()
    app_dim = model.appearance_dim
    all_depths, all_spreads, all_covs = [], [], []

    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    samples_done = 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if samples_done >= n_samples: break
            frames = batch["video"].cuda()
            out = model(frames)
            slots_c = out['slots']['corrected'] if isinstance(out['slots'], dict) else out['slots']
            alpha = out['alpha']
            T = min(slots_c.shape[1], max_frames)

            for t in range(T):
                slots_t = slots_c[:, t]
                alpha_t = alpha[:, :, t]
                if alpha_t.dim() == 5:
                    alpha_2d = alpha_t.squeeze(2)
                else:
                    alpha_2d = alpha_t
                B, N, H, W = alpha_2d.shape
                gy, gx = torch.meshgrid(
                    torch.linspace(-1, 1, H, device='cuda'),
                    torch.linspace(-1, 1, W, device='cuda'), indexing='ij')
                gx_b = gx.unsqueeze(0).unsqueeze(0).expand(B, N, H, W)
                gy_b = gy.unsqueeze(0).unsqueeze(0).expand(B, N, H, W)
                a_sum = alpha_2d.sum(dim=[-2, -1], keepdim=True) + 1e-8
                a_norm = alpha_2d / a_sum
                cx = (a_norm * gx_b).sum(dim=[-2, -1])
                cy = (a_norm * gy_b).sum(dim=[-2, -1])
                spread = torch.sqrt((a_norm * ((gx_b - cx.unsqueeze(-1).unsqueeze(-1))**2 +
                                               (gy_b - cy.unsqueeze(-1).unsqueeze(-1))**2)).sum(dim=[-2, -1]))
                depth = slots_t[:, :, app_dim + 2]
                a_max = alpha_2d.amax(dim=[-2, -1])
                cov = alpha_2d.sum(dim=[-2, -1])
                fg = (cov > 20) & (cov < 1500) & (spread > 0.01) & (a_max > 0.7) & (depth < 0.3)

                for b_idx in range(B):
                    for s_idx in range(N):
                        if fg[b_idx, s_idx]:
                            all_depths.append(depth[b_idx, s_idx].item())
                            all_spreads.append(spread[b_idx, s_idx].item())
                            all_covs.append(cov[b_idx, s_idx].item() / (H * W))

            samples_done += B

    return np.array(all_depths), np.array(all_spreads), np.array(all_covs)


def huber_r2(y_true, y_pred, delta_scale=1.5):
    res = np.abs(y_true - y_pred)
    delta = np.median(res) * delta_scale + 1e-12
    huber_res = np.where(res <= delta, 0.5 * res**2, delta * (res - 0.5 * delta))
    y_centered = np.abs(y_true - y_true.mean())
    huber_var = np.where(y_centered <= delta, 0.5 * y_centered**2, delta * (y_centered - 0.5 * delta))
    return 1.0 - huber_res.sum() / max(huber_var.sum(), 1e-12)


def huber_fit(x, y, delta_scale=1.5):
    from scipy.optimize import minimize
    def huber_loss(params):
        k, b = params
        pred = k * x + b
        res = np.abs(y - pred)
        delta = max(np.median(res) * delta_scale, 1e-12)
        h = np.where(res <= delta, 0.5 * res**2, delta * (res - 0.5 * delta))
        return h.sum()
    init = np.polyfit(x, y, 1)
    result = minimize(huber_loss, init, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 10000})
    return result.x


def main():
    with open('config/pretrain_phase2.yaml', 'r') as f:
        cfg_dict = yaml.safe_load(f)

    ds = OBJ3DDataset(data_path='data/obj3d', num_frames=1, subsample=2, stride=4)
    n_samples = 10000

    # === Baseline (MLP predictor, depth_spread_prior=False) ===
    print("Loading baseline model...")
    cfg_bl = dict(cfg_dict)
    cfg_bl['depth_spread_prior'] = False
    cfg_bl['continue_pretrain'] = True
    model_bl = load_model(cfg_bl, 'good_checkpoints/coveragedepth_pred_best_single.pt')
    print("Collecting baseline data...")
    d_bl, s_bl, c_bl = collect_data(model_bl, ds, n_samples=n_samples, max_frames=1)
    del model_bl; torch.cuda.empty_cache()

    # === Prior (ax+c, bx²+d, depth_spread_prior=True) ===
    print("Loading prior model...")
    cfg_pr = dict(cfg_dict)
    cfg_pr['depth_spread_prior'] = True
    cfg_pr['continue_pretrain'] = True
    model_pr = load_model(cfg_pr, 'experiments/phase2_depth_spread/checkpoints/best.pt')
    print("Collecting prior data...")
    d_pr, s_pr, c_pr = collect_data(model_pr, ds, n_samples=n_samples, max_frames=1)

    a_val = model_pr.depth_spread_a.item()
    b_val = model_pr.depth_spread_b.item()
    c_val = model_pr.depth_spread_c.item()
    d_val = model_pr.depth_spread_d.item()
    del model_pr; torch.cuda.empty_cache()

    # Filter depth > 0.04
    mask_bl = d_bl > 0.04; mask_pr = d_pr > 0.04
    d_bl, s_bl, c_bl = d_bl[mask_bl], s_bl[mask_bl], c_bl[mask_bl]
    d_pr, s_pr, c_pr = d_pr[mask_pr], s_pr[mask_pr], c_pr[mask_pr]
    d2_bl, d2_pr = d_bl**2, d_pr**2

    print(f"Baseline: {len(d_bl)} FG points, Prior: {len(d_pr)} FG points")

    # === Huber-robust fit ===
    coef_s_bl = huber_fit(d_bl, s_bl)
    coef_c_bl = huber_fit(d2_bl, c_bl)
    coef_s_pr = huber_fit(d_pr, s_pr)
    coef_c_pr = huber_fit(d2_pr, c_pr)

    r2_s_bl = huber_r2(s_bl, coef_s_bl[0] * d_bl + coef_s_bl[1])
    r2_c_bl = huber_r2(c_bl, coef_c_bl[0] * d2_bl + coef_c_bl[1])
    r2_s_pr = huber_r2(s_pr, coef_s_pr[0] * d_pr + coef_s_pr[1])
    r2_c_pr = huber_r2(c_pr, coef_c_pr[0] * d2_pr + coef_c_pr[1])

    r2_s_prior = huber_r2(s_pr, a_val * d_pr + c_val)
    r2_c_prior = huber_r2(c_pr, b_val * d2_pr + d_val)

    print(f"\n=== Results (Huber fit + Huber R²) ===")
    print(f"Baseline: R²(spread)={r2_s_bl:.4f}, R²(cov)={r2_c_bl:.4f}")
    print(f"Prior fit: R²(spread)={r2_s_pr:.4f}, R²(cov)={r2_c_pr:.4f}")
    print(f"Prior params: R²(spread)={r2_s_prior:.4f}, R²(cov)={r2_c_prior:.4f}")
    print(f"Prior params: a={a_val:.4f}, c={c_val:.6f}, b={b_val:.4f}, d={d_val:.6f}")
    print(f"Huber fit: spread k_bl={coef_s_bl[0]:.4f}, k_pr={coef_s_pr[0]:.4f}")
    print(f"Huber fit: cov k_bl={coef_c_bl[0]:.4f}, k_pr={coef_c_pr[0]:.4f}")

    # === Plot ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Row 1: Baseline
    ax = axes[0, 0]
    ax.scatter(d_bl, s_bl, s=2, alpha=0.15, c='steelblue')
    x_line = np.linspace(0, d_bl.max() * 1.05, 100)
    ax.plot(x_line, coef_s_bl[0] * x_line + coef_s_bl[1], 'r-', linewidth=2,
            label=f'y={coef_s_bl[0]:.3f}x+{coef_s_bl[1]:.3f}  R²={r2_s_bl:.4f}')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_xlabel('Depth'); ax.set_ylabel('Alpha Spread')
    ax.set_title(f'Baseline: Depth vs Spread ({len(d_bl)} pts)'); ax.legend()

    ax = axes[0, 1]
    ax.scatter(d2_bl, c_bl, s=2, alpha=0.15, c='steelblue')
    x_line2 = np.linspace(0, d2_bl.max() * 1.05, 100)
    ax.plot(x_line2, coef_c_bl[0] * x_line2 + coef_c_bl[1], 'r-', linewidth=2,
            label=f'y={coef_c_bl[0]:.3f}x+{coef_c_bl[1]:.3f}  R²={r2_c_bl:.4f}')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_xlabel('Depth²'); ax.set_ylabel('Alpha Coverage (norm)')
    ax.set_title(f'Baseline: Depth² vs Coverage ({len(d_bl)} pts)'); ax.legend()

    # Row 2: Prior
    ax = axes[1, 0]
    ax.scatter(d_pr, s_pr, s=2, alpha=0.15, c='darkorange')
    ax.plot(x_line, coef_s_pr[0] * x_line + coef_s_pr[1], 'b-', linewidth=2,
            label=f'Huber y={coef_s_pr[0]:.3f}x+{coef_s_pr[1]:.3f}  R²={r2_s_pr:.4f}')
    ax.plot(x_line, a_val * x_line + c_val, 'r--', linewidth=2,
            label=f'Prior y={a_val:.3f}x+{c_val:.4f}  R²={r2_s_prior:.4f}')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_xlabel('Depth'); ax.set_ylabel('Alpha Spread')
    ax.set_title(f'Prior: Depth vs Spread ({len(d_pr)} pts)'); ax.legend()

    ax = axes[1, 1]
    ax.scatter(d2_pr, c_pr, s=2, alpha=0.15, c='darkorange')
    ax.plot(x_line2, coef_c_pr[0] * x_line2 + coef_c_pr[1], 'b-', linewidth=2,
            label=f'Huber y={coef_c_pr[0]:.3f}x+{coef_c_pr[1]:.3f}  R²={r2_c_pr:.4f}')
    ax.plot(x_line2, b_val * x_line2 + d_val, 'r--', linewidth=2,
            label=f'Prior y={b_val:.3f}x+{d_val:.4f}  R²={r2_c_prior:.4f}')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_xlabel('Depth²'); ax.set_ylabel('Alpha Coverage (norm)')
    ax.set_title(f'Prior: Depth² vs Coverage ({len(d_pr)} pts)'); ax.legend()

    plt.suptitle(f'Baseline vs Prior Phase2 (3000 samples, 1 frame)\n'
                 f'Baseline R²: s={r2_s_bl:.4f} c={r2_c_bl:.4f}  |  '
                 f'Prior R²: s={r2_s_pr:.4f} c={r2_c_pr:.4f}',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    save_path = 'pos_depth_debug/baseline_vs_prior_phase2.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
