#!/usr/bin/env python3
"""
v11: 1. 统计 sharpened acov vs pixcov 相关性
     2. 用 sharpened acov 做 d_t 优化 + 可视化
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn
import numpy as np, yaml, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from train import Trainer, create_optimizer
from train.trainer import WandBLogger

OUT = os.path.dirname(os.path.abspath(__file__))


def load_model(cfg_dict, ckpt_path):
    cfg = SimpleNamespace(**cfg_dict)
    model = SlotDynamicsModel(cfg)
    opt, sch = create_optimizer((p for p in model.parameters() if p.requires_grad), cfg)
    wb = WandBLogger(enabled=False)
    trainer = Trainer(model, opt, sch, cfg, wandb_logger=wb)
    trainer.load_checkpoint(ckpt_path)
    return model


def sharpen_alpha(alpha, tau=0.05):
    """sigmoid sharpening: push alpha toward 0 and 1"""
    return torch.sigmoid((alpha - 0.5) / tau)


def solve_depth_sharpened_acov(decoder, best_app, pos_t, d_init, bg_slots_t,
                               target_sharpened_acov, tau=0.05,
                               n_steps=100, lr=0.01):
    """d_t 优化: 匹配 sharpened acov"""
    d_t = torch.tensor([d_init], device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([d_t], lr=lr)
    for step in range(n_steps):
        target_slot = torch.cat([best_app.detach(), pos_t.detach(), d_t.reshape(1)])
        slots = torch.cat([
            target_slot.reshape(1, 1, -1),
            bg_slots_t.detach().reshape(1, -1, bg_slots_t.shape[-1])
        ], dim=1)
        _, alpha, _ = decoder(slots, return_rgb=True)
        alpha_sharp = sharpen_alpha(alpha[0, 0, 0], tau)
        pred_acov = alpha_sharp.sum() / (64 * 64)
        loss = (pred_acov - target_sharpened_acov) ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            d_t.data.clamp_(0.01, 0.5)
    return d_t.item()


def solve_appearance_full(decoder, a_init, pos_t, d_t_val, all_slots_t, target_idx,
                          target_img, n_steps=200, lr=0.01):
    a_t = a_init.clone().detach().requires_grad_(True)
    d_t_tensor = torch.tensor([d_t_val], device='cuda')
    optimizer = torch.optim.Adam([a_t], lr=lr)
    for step in range(n_steps):
        target_slot = torch.cat([a_t, pos_t.detach(), d_t_tensor])
        slots = all_slots_t.clone()
        slots[0, target_idx] = target_slot
        recon, _, _ = decoder(slots, return_rgb=True)
        loss = nn.functional.mse_loss(recon, target_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return a_t.detach()


def main():
    with open('config/pretrain_phase3.yaml') as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict['burnin_frames'] = 16
    ckpt_path = 'experiments/phase3_gru2_full/checkpoints/best.pt'

    print("Loading model...")
    model = load_model(cfg_dict, ckpt_path)
    model.eval().cuda()
    app_dim = model.appearance_dim

    sample = torch.load('data/sample9/sample9.pt')
    frames = sample['video'].unsqueeze(0).cuda()

    with torch.no_grad():
        out = model(frames)
    slots_c = out['slots']['corrected'] if isinstance(out['slots'], dict) else out['slots']
    alpha_out = out['alpha']
    T = slots_c.shape[1]
    N = slots_c.shape[2]

    fg_slots, bg_slots = [], []
    for s in range(N):
        mean_amax = np.mean([alpha_out[0, s, t].amax().item() for t in range(T)])
        max_depth = max(slots_c[0, t, s, app_dim + 2].item() for t in range(T))
        if mean_amax > 0.3 and max_depth < 0.4:
            fg_slots.append(s)
        else:
            bg_slots.append(s)
    print(f"FG={fg_slots}, BG={bg_slots}")

    # ============================================================
    # Part 1: 统计 sharpened acov vs pixcov 相关性
    # ============================================================
    print("\n=== Part 1: Sharpened acov vs pixcov correlation ===")

    # 收集所有 FG slot 的 NoFG alpha 数据
    all_acovs = []       # original acov
    all_pixcovs = []     # pixel coverage (argmax)
    all_sharp_acovs = {} # sharpened acov for different tau

    tau_values = [0.01, 0.02, 0.05, 0.1, 0.2]
    for tau in tau_values:
        all_sharp_acovs[tau] = []

    # 也测试 alpha^gamma
    gamma_values = [0.5, 2.0, 4.0]
    all_gamma_acovs = {}
    for gamma in gamma_values:
        all_gamma_acovs[gamma] = []

    # 以及硬阈值
    threshold_values = [0.3, 0.5]
    all_thresh_acovs = {}
    for thr in threshold_values:
        all_thresh_acovs[thr] = []

    for s in fg_slots:
        for t in range(T):
            keep = [s] + bg_slots
            slots_nofg = slots_c[0, t][keep].unsqueeze(0)
            with torch.no_grad():
                _, alpha_nofg, _ = model.decoder(slots_nofg, return_rgb=True)
            alpha = alpha_nofg[0, 0, 0]  # (H, W)

            acov = alpha.sum().item() / (64*64)
            dominant = alpha_nofg[0, :, 0].argmax(dim=0)
            pixcov = (dominant == 0).sum().item() / (64*64)

            all_acovs.append(acov)
            all_pixcovs.append(pixcov)

            for tau in tau_values:
                alpha_s = sharpen_alpha(alpha, tau)
                all_sharp_acovs[tau].append(alpha_s.sum().item() / (64*64))

            for gamma in gamma_values:
                alpha_g = alpha ** gamma
                all_gamma_acovs[gamma].append(alpha_g.sum().item() / (64*64))

            for thr in threshold_values:
                alpha_t = (alpha > thr).float()
                all_thresh_acovs[thr].append(alpha_t.sum().item() / (64*64))

    all_acovs = np.array(all_acovs)
    all_pixcovs = np.array(all_pixcovs)

    # Compute R² for each method
    def compute_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Linear fit for R²
    def fit_r2(x, y):
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        return compute_r2(y, y_pred), coeffs

    r2_orig, coeff_orig = fit_r2(all_acovs, all_pixcovs)
    print(f"Original acov vs pixcov:  R²={r2_orig:.4f}")

    results_table = [("original acov", r2_orig)]

    for tau in tau_values:
        vals = np.array(all_sharp_acovs[tau])
        r2, _ = fit_r2(vals, all_pixcovs)
        results_table.append((f"sigmoid(tau={tau})", r2))
        print(f"Sigmoid tau={tau}:  R²={r2:.4f}")

    for gamma in gamma_values:
        vals = np.array(all_gamma_acovs[gamma])
        r2, _ = fit_r2(vals, all_pixcovs)
        results_table.append((f"alpha^{gamma}", r2))
        print(f"Alpha^{gamma}:  R²={r2:.4f}")

    for thr in threshold_values:
        vals = np.array(all_thresh_acovs[thr])
        r2, _ = fit_r2(vals, all_pixcovs)
        results_table.append((f"thresh>{thr}", r2))
        print(f"Threshold>{thr}:  R²={r2:.4f}")

    # Plot scatter: original vs best sharpened
    best_method = max(results_table[1:], key=lambda x: x[1])
    best_tau = float(best_method[0].split("=")[1].rstrip(")")) if "tau" in best_method[0] else 0.05

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].scatter(all_acovs, all_pixcovs, s=10, alpha=0.5)
    x_line = np.linspace(0, max(all_acovs), 100)
    axes[0].plot(x_line, np.polyval(coeff_orig, x_line), 'r-', linewidth=2)
    axes[0].set_xlabel('Original acov'); axes[0].set_ylabel('Pixel coverage')
    axes[0].set_title(f'Original acov vs pixcov\nR²={r2_orig:.4f}')
    axes[0].grid(True, alpha=0.3)

    # Sigmoid sharpened
    best_sharp_vals = np.array(all_sharp_acovs[best_tau])
    r2_best, coeff_best = fit_r2(best_sharp_vals, all_pixcovs)
    axes[1].scatter(best_sharp_vals, all_pixcovs, s=10, alpha=0.5, color='orange')
    x_line2 = np.linspace(0, max(best_sharp_vals), 100)
    axes[1].plot(x_line2, np.polyval(coeff_best, x_line2), 'r-', linewidth=2)
    axes[1].set_xlabel(f'Sharpened acov (tau={best_tau})'); axes[1].set_ylabel('Pixel coverage')
    axes[1].set_title(f'Sharpened acov vs pixcov\nR²={r2_best:.4f}')
    axes[1].grid(True, alpha=0.3)

    # Threshold
    thresh_vals = np.array(all_thresh_acovs[0.5])
    r2_thresh, coeff_thresh = fit_r2(thresh_vals, all_pixcovs)
    axes[2].scatter(thresh_vals, all_pixcovs, s=10, alpha=0.5, color='green')
    x_line3 = np.linspace(0, max(thresh_vals), 100)
    axes[2].plot(x_line3, np.polyval(coeff_thresh, x_line3), 'r-', linewidth=2)
    axes[2].set_xlabel('Threshold>0.5 acov'); axes[2].set_ylabel('Pixel coverage')
    axes[2].set_title(f'Threshold acov vs pixcov\nR²={r2_thresh:.4f}')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('acov vs pixcov: Original / Sigmoid Sharpened / Threshold', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUT}/v11_acov_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: v11_acov_correlation.png")

    # ============================================================
    # Part 2: 用 sharpened acov 做 d_t 优化 + 可视化
    # ============================================================
    print(f"\n=== Part 2: d_t optimization with sharpened acov (tau={best_tau}) ===")

    # Pre-compute
    origin_imgs = []
    nofg_imgs = {s: [] for s in fg_slots}
    nofg_sharp_acovs = {s: [] for s in fg_slots}
    per_frame_cov = {s: [] for s in fg_slots}

    for t in range(T):
        slots_t = slots_c[0, t].unsqueeze(0)
        with torch.no_grad():
            dec_full, _, _ = model.decoder(slots_t, return_rgb=True)
        origin_imgs.append(dec_full[0].detach())

        for s in fg_slots:
            keep = [s] + bg_slots
            slots_nofg = slots_c[0, t][keep].unsqueeze(0)
            with torch.no_grad():
                dec_nofg, alpha_nofg, _ = model.decoder(slots_nofg, return_rgb=True)
            nofg_imgs[s].append(dec_nofg[0].detach())
            alpha_sharp = sharpen_alpha(alpha_nofg[0, 0, 0], best_tau)
            nofg_sharp_acovs[s].append(alpha_sharp.sum().item() / (64*64))
            alpha_t = alpha_out[0, s, t]
            a2 = alpha_t.squeeze(0) if alpha_t.dim() == 3 else alpha_t
            per_frame_cov[s].append(a2.sum().item() / (64 * 64))

    # Solve
    solved = {}
    t0 = time.time()

    for s_target in fg_slots:
        best_t = max(range(T), key=lambda t: per_frame_cov[s_target][t])
        best_app = slots_c[0, best_t, s_target, :app_dim].clone().detach()
        print(f"\nSlot {s_target}: best_t={best_t}")
        solved[s_target] = {}

        for t in range(T):
            slots_t = slots_c[0, t]
            pos_t = slots_t[s_target, app_dim:app_dim+2].detach()
            d_orig = slots_t[s_target, app_dim+2].item()
            a_orig = slots_t[s_target, :app_dim].detach()
            bg_slots_t = slots_t[bg_slots].detach()

            d_t = solve_depth_sharpened_acov(
                model.decoder, best_app, pos_t, d_orig, bg_slots_t,
                nofg_sharp_acovs[s_target][t], tau=best_tau,
                n_steps=100, lr=0.01)

            target_origin = origin_imgs[t].unsqueeze(0)
            all_slots_t = slots_t.unsqueeze(0).clone().detach()
            a_t = solve_appearance_full(
                model.decoder, a_orig, pos_t, d_t, all_slots_t, s_target,
                target_origin, n_steps=200, lr=0.01)

            solved[s_target][t] = {'d_t': d_t, 'a_t': a_t, 'd_orig': d_orig, 'best_t': best_t}
            elapsed = time.time() - t0
            print(f"  t={t:>2}: d={d_orig:.4f}→{d_t:.4f} ({d_t-d_orig:+.4f})  "
                  f"sharp_acov={nofg_sharp_acovs[s_target][t]:.6f}  [{elapsed:.0f}s]")

    # === Per-slot 5-row images ===
    print("\nGenerating per-slot images...")
    for s_target in fg_slots:
        best_t = solved[s_target][0]['best_t']
        best_app = slots_c[0, best_t, s_target, :app_dim].clone().detach()

        fig, axes = plt.subplots(5, T, figsize=(2.8 * T, 13))

        for t in range(T):
            gt = sample['video'][t].permute(1,2,0).cpu().numpy()
            gt = np.clip((gt + 1) / 2, 0, 1)
            d_orig = solved[s_target][t]['d_orig']
            d_t = solved[s_target][t]['d_t']
            a_t = solved[s_target][t]['a_t']

            axes[0][t].imshow(gt); axes[0][t].set_title(f't={t}', fontsize=8); axes[0][t].axis('off')

            axes[1][t].imshow(np.clip(origin_imgs[t].permute(1,2,0).cpu().numpy(), 0, 1))
            axes[1][t].set_title(f'd={d_orig:.4f}', fontsize=7); axes[1][t].axis('off')

            # Row 3: orig app + orig depth NoFG decode (target)
            slots_t = slots_c[0, t]
            keep = [s_target] + bg_slots
            slots_nofg = slots_t[keep].unsqueeze(0)
            with torch.no_grad():
                dec_nofg, alpha_nofg, _ = model.decoder(slots_nofg, return_rgb=True)
            acov_nofg = alpha_nofg[0, 0, 0].sum().item() / (64*64)
            pixcov_nofg = (alpha_nofg[0, :, 0].argmax(dim=0) == 0).sum().item() / (64*64)
            axes[2][t].imshow(np.clip(dec_nofg[0].permute(1,2,0).cpu().numpy(), 0, 1))
            axes[2][t].set_title(f'd={d_orig:.4f} acov={acov_nofg:.5f}\npixcov={pixcov_nofg:.5f}', fontsize=5)
            axes[2][t].axis('off')

            # Row 4: BestApp + solved d_t NoFG decode
            slot_sd = slots_t[s_target].clone()
            slot_sd[:app_dim] = best_app
            slot_sd[app_dim+2] = d_t
            slots_nofg_sd = slots_nofg.clone()
            slots_nofg_sd[0, 0] = slot_sd
            with torch.no_grad():
                dec_sd, alpha_sd, _ = model.decoder(slots_nofg_sd, return_rgb=True)
            acov_sd = alpha_sd[0, 0, 0].sum().item() / (64*64)
            pixcov_sd = (alpha_sd[0, :, 0].argmax(dim=0) == 0).sum().item() / (64*64)
            axes[3][t].imshow(np.clip(dec_sd[0].permute(1,2,0).cpu().numpy(), 0, 1))
            d_delta = d_t - d_orig
            d_color = 'red' if d_delta < -0.01 else ('blue' if d_delta > 0.01 else 'black')
            axes[3][t].set_title(f'd={d_t:.4f} acov={acov_sd:.5f}\npixcov={pixcov_sd:.5f}', fontsize=5, color=d_color)
            axes[3][t].axis('off')

            # Row 5: solved a_t + solved d_t NoFG decode
            slot_sa = torch.cat([a_t, slots_t[s_target, app_dim:].detach()])
            slot_sa[app_dim+2] = d_t
            slots_nofg_sa = slots_nofg.clone()
            slots_nofg_sa[0, 0] = slot_sa
            with torch.no_grad():
                dec_sa, alpha_sa, _ = model.decoder(slots_nofg_sa, return_rgb=True)
            acov_sa = alpha_sa[0, 0, 0].sum().item() / (64*64)
            axes[4][t].imshow(np.clip(dec_sa[0].permute(1,2,0).cpu().numpy(), 0, 1))
            axes[4][t].set_title(f'd={d_t:.4f} acov={acov_sa:.5f}', fontsize=5)
            axes[4][t].axis('off')

        for row, label in enumerate(['GT', 'Normal ISA', 'origApp+origD\n(NoFG, target)',
                                      'BestApp+solvedD\n(NoFG, sharp_acov)', 'solvedA+solvedD\n(NoFG)']):
            axes[row][0].set_ylabel(label, fontsize=7, rotation=0, labelpad=60, ha='right', va='center')

        plt.suptitle(f'Slot {s_target}  [BestApp t={best_t}, tau={best_tau}]', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{OUT}/v11_solved_slot{s_target}.png', dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Saved: v11_solved_slot{s_target}.png")

    # === Line plots ===
    print("\nGenerating line plots...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    fr = np.arange(T)
    colors = plt.cm.tab10(np.linspace(0, 1, len(fg_slots)))

    for idx, s in enumerate(fg_slots):
        d_origs = [solved[s][t]['d_orig'] for t in range(T)]
        d_ts = [solved[s][t]['d_t'] for t in range(T)]
        acovs_orig, acovs_solved = [], []
        pixcovs_orig, pixcovs_solved = [], []

        for t in range(T):
            slots_t = slots_c[0, t]
            keep = [s] + bg_slots
            slots_nofg = slots_t[keep].unsqueeze(0)
            with torch.no_grad():
                _, alpha_o, _ = model.decoder(slots_nofg, return_rgb=True)
            acovs_orig.append(alpha_o[0, 0, 0].sum().item() / (64*64))
            pixcovs_orig.append((alpha_o[0, :, 0].argmax(dim=0) == 0).sum().item() / (64*64))

            a_t = solved[s][t]['a_t']
            d_t = solved[s][t]['d_t']
            slot_sa = torch.cat([a_t, slots_t[s, app_dim:].detach()])
            slot_sa[app_dim+2] = d_t
            slots_nofg_sa = slots_nofg.clone()
            slots_nofg_sa[0, 0] = slot_sa
            with torch.no_grad():
                _, alpha_sa, _ = model.decoder(slots_nofg_sa, return_rgb=True)
            acovs_solved.append(alpha_sa[0, 0, 0].sum().item() / (64*64))
            pixcovs_solved.append((alpha_sa[0, :, 0].argmax(dim=0) == 0).sum().item() / (64*64))

        axes[0].plot(fr, d_origs, '--', color=colors[idx], alpha=0.4, label=f'Slot {s} orig')
        axes[0].plot(fr, d_ts, '-o', markersize=3, color=colors[idx], label=f'Slot {s} solved')
        axes[1].plot(fr, pixcovs_orig, '--', color=colors[idx], alpha=0.4)
        axes[1].plot(fr, pixcovs_solved, '-o', markersize=3, color=colors[idx])
        axes[2].plot(fr, acovs_orig, '--', color=colors[idx], alpha=0.4)
        axes[2].plot(fr, acovs_solved, '-o', markersize=3, color=colors[idx])

    axes[0].set_xlabel('Frame'); axes[0].set_ylabel('Depth (=Scale)')
    axes[0].set_title('d_t: orig vs solved'); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Frame'); axes[1].set_ylabel('Pixel Coverage')
    axes[1].set_title('pixcov (NoFG): orig vs solved'); axes[1].grid(True, alpha=0.3)
    axes[2].set_xlabel('Frame'); axes[2].set_ylabel('Alpha Coverage')
    axes[2].set_title('acov (NoFG): orig vs solved'); axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'v11: d_t via sharpened acov (tau={best_tau})', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUT}/v11_solved_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # === Summary ===
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Best tau: {best_tau}")
    print(f"R² improvement: original acov={r2_orig:.4f} → sharpened={r2_best:.4f}")
    for s in fg_slots:
        best_t_s = solved[s][0]['best_t']
        print(f"\nSlot {s} [best_t={best_t_s}]:")
        for t in range(T):
            d_o = solved[s][t]['d_orig']; d_s = solved[s][t]['d_t']
            print(f"  t={t:>2}: d={d_o:.4f}→{d_s:.4f} ({d_s-d_o:+.4f})")


if __name__ == '__main__':
    main()
