#!/usr/bin/env python3
"""
v12: 二分法 d_t 优化 + 精简 a_t 优化 (BestApp init, 25步)
- d_t: bisection on sharpened acov, ~12次 forward-only
- a_t: BestApp init + 25步 gradient descent on full decode RGB
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
TAU = 0.05


def load_model(cfg_dict, ckpt_path):
    cfg = SimpleNamespace(**cfg_dict)
    model = SlotDynamicsModel(cfg)
    opt, sch = create_optimizer((p for p in model.parameters() if p.requires_grad), cfg)
    wb = WandBLogger(enabled=False)
    trainer = Trainer(model, opt, sch, cfg, wandb_logger=wb)
    trainer.load_checkpoint(ckpt_path)
    return model


def sharpen_alpha(alpha, tau=TAU):
    return torch.sigmoid((alpha - 0.5) / tau)


def compute_sharp_acov(decoder, slot_vec, bg_slots, tau=TAU):
    """NoFG decode → sharpened acov (forward only, no grad)"""
    with torch.no_grad():
        slots = torch.cat([slot_vec.reshape(1, 1, -1),
                           bg_slots.reshape(1, -1, bg_slots.shape[-1])], dim=1)
        _, alpha, _ = decoder(slots, return_rgb=True)
        alpha_sharp = sharpen_alpha(alpha[0, 0, 0], tau)
        return alpha_sharp.sum().item() / (64 * 64)


def solve_depth_gd(decoder, best_app, pos_t, d_init, bg_slots_t,
                   target_sharp_acov, n_steps=20, lr=0.01, tau=TAU):
    """梯度下降 d_t: 匹配 sharpened acov, 从 d_init 开始"""
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
        loss = (pred_acov - target_sharp_acov) ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            d_t.data.clamp_(0.01, 0.5)
    return d_t.item()


def solve_appearance_fast(decoder, best_app, pos_t, d_t_val, all_slots_t, target_idx,
                          target_img, n_steps=25, lr=0.02):
    """a_t 优化: BestApp init + 少量步数"""
    a_t = best_app.clone().detach().requires_grad_(True)
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

    # Pre-compute targets
    origin_imgs = []
    nofg_sharp_acovs = {s: [] for s in fg_slots}
    per_frame_cov = {s: [] for s in fg_slots}

    print("Pre-computing NoFG targets...")
    for t in range(T):
        slots_t = slots_c[0, t].unsqueeze(0)
        with torch.no_grad():
            dec_full, _, _ = model.decoder(slots_t, return_rgb=True)
        origin_imgs.append(dec_full[0].detach())

        for s in fg_slots:
            keep = [s] + bg_slots
            slots_nofg = slots_c[0, t][keep].unsqueeze(0)
            with torch.no_grad():
                _, alpha_nofg, _ = model.decoder(slots_nofg, return_rgb=True)
            alpha_sharp = sharpen_alpha(alpha_nofg[0, 0, 0], TAU)
            nofg_sharp_acovs[s].append(alpha_sharp.sum().item() / (64 * 64))
            alpha_t = alpha_out[0, s, t]
            a2 = alpha_t.squeeze(0) if alpha_t.dim() == 3 else alpha_t
            per_frame_cov[s].append(a2.sum().item() / (64 * 64))

    # === Solve: bisection d_t + fast a_t ===
    solved = {}
    t0_total = time.time()

    for s_target in fg_slots:
        best_t = max(range(T), key=lambda t: per_frame_cov[s_target][t])
        best_app = slots_c[0, best_t, s_target, :app_dim].clone().detach()
        print(f"\nSlot {s_target}: best_t={best_t}")
        solved[s_target] = {}

        for t in range(T):
            slots_t = slots_c[0, t]
            pos_t = slots_t[s_target, app_dim:app_dim+2].detach()
            d_orig = slots_t[s_target, app_dim+2].item()
            bg_slots_t = slots_t[bg_slots].detach()

            t0 = time.time()
            d_t = solve_depth_gd(
                model.decoder, best_app, pos_t, d_orig, bg_slots_t,
                nofg_sharp_acovs[s_target][t], n_steps=20, lr=0.01, tau=TAU)
            t_bisect = time.time() - t0

            t0 = time.time()
            target_origin = origin_imgs[t].unsqueeze(0)
            all_slots_t = slots_t.unsqueeze(0).clone().detach()
            a_t = solve_appearance_fast(
                model.decoder, best_app, pos_t, d_t, all_slots_t, s_target,
                target_origin, n_steps=25, lr=0.02)
            t_at = time.time() - t0

            solved[s_target][t] = {'d_t': d_t, 'a_t': a_t, 'd_orig': d_orig, 'best_t': best_t}
            print(f"  t={t:>2}: d={d_orig:.4f}→{d_t:.4f} ({d_t-d_orig:+.4f})  "
                  f"[bisect={t_bisect:.2f}s, a_t={t_at:.2f}s]")

    total_time = time.time() - t0_total
    print(f"\nTotal solve time: {total_time:.1f}s")

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
                                      'BestApp+solvedD\n(NoFG, bisect+sharp)', 'solvedA+solvedD\n(NoFG)']):
            axes[row][0].set_ylabel(label, fontsize=7, rotation=0, labelpad=60, ha='right', va='center')

        plt.suptitle(f'Slot {s_target}  [BestApp t={best_t}, tau={TAU}, bisect+fastA]', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{OUT}/v12_solved_slot{s_target}.png', dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Saved: v12_solved_slot{s_target}.png")

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

    plt.suptitle(f'v12: bisect d_t + fast a_t (tau={TAU})', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUT}/v12_solved_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # === v11 vs v12 d_t comparison ===
    print("\nGenerating v11 vs v12 comparison...")
    try:
        v11_data = {}
        # Re-run v11 gradient method for comparison on a subset
        for s_target in fg_slots:
            best_t = max(range(T), key=lambda t: per_frame_cov[s_target][t])
            best_app = slots_c[0, best_t, s_target, :app_dim].clone().detach()
            v11_data[s_target] = []
            for t in range(T):
                slots_t = slots_c[0, t]
                pos_t = slots_t[s_target, app_dim:app_dim+2].detach()
                d_orig = slots_t[s_target, app_dim+2].item()
                bg_slots_t = slots_t[bg_slots].detach()
                # v11 gradient descent
                d_t = torch.tensor([d_orig], device='cuda', requires_grad=True)
                optimizer = torch.optim.Adam([d_t], lr=0.01)
                for step in range(100):
                    target_slot = torch.cat([best_app.detach(), pos_t.detach(), d_t.reshape(1)])
                    slots_batch = torch.cat([
                        target_slot.reshape(1, 1, -1),
                        bg_slots_t.detach().reshape(1, -1, bg_slots_t.shape[-1])
                    ], dim=1)
                    _, alpha, _ = model.decoder(slots_batch, return_rgb=True)
                    alpha_sharp = sharpen_alpha(alpha[0, 0, 0], TAU)
                    pred_acov = alpha_sharp.sum() / (64 * 64)
                    loss = (pred_acov - nofg_sharp_acovs[s_target][t]) ** 2
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        d_t.data.clamp_(0.01, 0.5)
                v11_data[s_target].append(d_t.item())

        fig, axes = plt.subplots(1, len(fg_slots), figsize=(5 * len(fg_slots), 4))
        if len(fg_slots) == 1:
            axes = [axes]
        for idx, s in enumerate(fg_slots):
            d_origs = [solved[s][t]['d_orig'] for t in range(T)]
            d_v11 = v11_data[s]
            d_v12 = [solved[s][t]['d_t'] for t in range(T)]
            axes[idx].plot(fr, d_origs, 'k--', alpha=0.3, label='orig')
            axes[idx].plot(fr, d_v11, 'o-', markersize=3, color='blue', label='v11 gradient')
            axes[idx].plot(fr, d_v12, 's-', markersize=3, color='red', label='v12 bisect')
            axes[idx].set_title(f'Slot {s}'); axes[idx].legend(fontsize=7)
            axes[idx].set_xlabel('Frame'); axes[idx].set_ylabel('Depth')
            axes[idx].grid(True, alpha=0.3)
        plt.suptitle('v11 (gradient 100步) vs v12 (bisect 12次) d_t comparison', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{OUT}/v12_vs_v11.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: v12_vs_v11.png")
    except Exception as e:
        print(f"  v11 comparison skipped: {e}")

    # === Summary ===
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Method: GD d_t (20 steps) + BestApp-init a_t (25 steps)")
    print(f"tau={TAU}, total_time={total_time:.1f}s")
    for s in fg_slots:
        best_t_s = solved[s][0]['best_t']
        print(f"\nSlot {s} [best_t={best_t_s}]:")
        for t in range(T):
            d_o = solved[s][t]['d_orig']; d_s = solved[s][t]['d_t']
            print(f"  t={t:>2}: d={d_o:.4f}→{d_s:.4f} ({d_s-d_o:+.4f})")


if __name__ == '__main__':
    main()
