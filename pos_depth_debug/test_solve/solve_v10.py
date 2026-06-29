#!/usr/bin/env python3
"""
Solved Depth v10: d_t 只优化 alpha (无 RGB loss), 两个版本对比
A: 匹配整个 alpha mask (MSE on alpha map)
B: 只匹配 alpha coverage (scalar)
Target: 每帧各自的 NoFG decode alpha
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


def solve_depth_alpha_mask(decoder, best_app, pos_t, d_init, bg_slots_t, target_alpha,
                           n_steps=100, lr=0.01):
    """A: 匹配整个 alpha mask"""
    d_t = torch.tensor([d_init], device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([d_t], lr=lr)
    for step in range(n_steps):
        target_slot = torch.cat([best_app.detach(), pos_t.detach(), d_t.reshape(1)])
        slots = torch.cat([
            target_slot.reshape(1, 1, -1),
            bg_slots_t.detach().reshape(1, -1, bg_slots_t.shape[-1])
        ], dim=1)
        _, alpha, _ = decoder(slots, return_rgb=True)
        pred_alpha = alpha[0, 0, 0]  # (H, W)
        loss = nn.functional.mse_loss(pred_alpha, target_alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            d_t.data.clamp_(0.01, 0.5)
    return d_t.item()


def solve_depth_acov(decoder, best_app, pos_t, d_init, bg_slots_t, target_acov,
                     n_steps=100, lr=0.01):
    """B: 只匹配 alpha coverage (scalar)"""
    d_t = torch.tensor([d_init], device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([d_t], lr=lr)
    for step in range(n_steps):
        target_slot = torch.cat([best_app.detach(), pos_t.detach(), d_t.reshape(1)])
        slots = torch.cat([
            target_slot.reshape(1, 1, -1),
            bg_slots_t.detach().reshape(1, -1, bg_slots_t.shape[-1])
        ], dim=1)
        _, alpha, _ = decoder(slots, return_rgb=True)
        pred_acov = alpha[0, 0, 0].sum() / (64 * 64)
        loss = (pred_acov - target_acov) ** 2
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

    # Pre-compute
    origin_imgs = []
    nofg_imgs = {s: [] for s in fg_slots}
    nofg_alphas = {s: [] for s in fg_slots}
    nofg_acovs = {s: [] for s in fg_slots}
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
            nofg_alphas[s].append(alpha_nofg[0, 0, 0].detach())  # (H, W)
            nofg_acovs[s].append(alpha_nofg[0, 0, 0].sum().item() / (64*64))
            alpha_t = alpha_out[0, s, t]
            a2 = alpha_t.squeeze(0) if alpha_t.dim() == 3 else alpha_t
            per_frame_cov[s].append(a2.sum().item() / (64 * 64))

    # === Solve: both methods ===
    results = {}
    for method, solve_fn in [('A_mask', solve_depth_alpha_mask), ('B_acov', solve_depth_acov)]:
        print(f"\n{'='*50}")
        print(f"Method {method}")
        print(f"{'='*50}")
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

                # d_t
                if method == 'A_mask':
                    d_t = solve_fn(model.decoder, best_app, pos_t, d_orig, bg_slots_t,
                                   nofg_alphas[s_target][t], n_steps=100, lr=0.01)
                else:
                    d_t = solve_fn(model.decoder, best_app, pos_t, d_orig, bg_slots_t,
                                   nofg_acovs[s_target][t], n_steps=100, lr=0.01)

                # a_t
                target_origin = origin_imgs[t].unsqueeze(0)
                all_slots_t = slots_t.unsqueeze(0).clone().detach()
                a_t = solve_appearance_full(
                    model.decoder, a_orig, pos_t, d_t, all_slots_t, s_target,
                    target_origin, n_steps=200, lr=0.01)

                solved[s_target][t] = {'d_t': d_t, 'a_t': a_t, 'd_orig': d_orig, 'best_t': best_t}
                elapsed = time.time() - t0
                print(f"  t={t:>2}: d={d_orig:.4f}→{d_t:.4f} ({d_t-d_orig:+.4f})  "
                      f"nofg_acov={nofg_acovs[s_target][t]:.6f}  [{elapsed:.0f}s]")

        results[method] = solved

    # === Visualization: one figure per slot, 5 rows x 2 methods side by side ===
    print("\nGenerating comparison images...")
    for s_target in fg_slots:
        best_t = results['A_mask'][s_target][0]['best_t']
        best_app = slots_c[0, best_t, s_target, :app_dim].clone().detach()

        # 5 rows x 2*16 cols (method A | method B)
        fig, axes = plt.subplots(5, T * 2, figsize=(1.8 * T * 2, 13))

        for t in range(T):
            gt = sample['video'][t].permute(1,2,0).cpu().numpy()
            gt = np.clip((gt + 1) / 2, 0, 1)
            d_orig = results['A_mask'][s_target][t]['d_orig']

            # Row 1: GT (span both methods)
            axes[0][t*2].imshow(gt); axes[0][t*2].set_title(f't={t}', fontsize=7); axes[0][t*2].axis('off')
            axes[0][t*2+1].imshow(gt); axes[0][t*2+1].axis('off')

            # Row 2: Normal ISA
            axes[1][t*2].imshow(np.clip(origin_imgs[t].permute(1,2,0).cpu().numpy(), 0, 1))
            axes[1][t*2].set_title(f'd={d_orig:.4f}', fontsize=6); axes[1][t*2].axis('off')
            axes[1][t*2+1].imshow(np.clip(origin_imgs[t].permute(1,2,0).cpu().numpy(), 0, 1))
            axes[1][t*2+1].axis('off')

            # Row 3: orig app NoFG decode (target)
            slots_t = slots_c[0, t]
            keep = [s_target] + bg_slots
            slots_nofg = slots_t[keep].unsqueeze(0)
            with torch.no_grad():
                dec_nofg, alpha_nofg, _ = model.decoder(slots_nofg, return_rgb=True)
            acov_nofg = alpha_nofg[0, 0, 0].sum().item() / (64*64)
            nofg_img = np.clip(dec_nofg[0].permute(1,2,0).cpu().numpy(), 0, 1)
            axes[2][t*2].imshow(nofg_img)
            axes[2][t*2].set_title(f'acov={acov_nofg:.5f}', fontsize=5); axes[2][t*2].axis('off')
            axes[2][t*2+1].imshow(nofg_img); axes[2][t*2+1].axis('off')

            for mi, method in enumerate(['A_mask', 'B_acov']):
                d_t = results[method][s_target][t]['d_t']
                a_t = results[method][s_target][t]['a_t']
                col = t * 2 + mi

                # Row 4: BestApp + solved d_t NoFG
                slot_sd = slots_t[s_target].clone()
                slot_sd[:app_dim] = best_app
                slot_sd[app_dim+2] = d_t
                slots_nofg_sd = slots_nofg.clone()
                slots_nofg_sd[0, 0] = slot_sd
                with torch.no_grad():
                    dec_sd, alpha_sd, _ = model.decoder(slots_nofg_sd, return_rgb=True)
                acov_sd = alpha_sd[0, 0, 0].sum().item() / (64*64)
                axes[3][col].imshow(np.clip(dec_sd[0].permute(1,2,0).cpu().numpy(), 0, 1))
                axes[3][col].set_title(f'd={d_t:.4f} acov={acov_sd:.5f}', fontsize=5)
                axes[3][col].axis('off')

                # Row 5: solved a_t + solved d_t NoFG
                slot_sa = torch.cat([a_t, slots_t[s_target, app_dim:].detach()])
                slot_sa[app_dim+2] = d_t
                slots_nofg_sa = slots_nofg.clone()
                slots_nofg_sa[0, 0] = slot_sa
                with torch.no_grad():
                    dec_sa, alpha_sa, _ = model.decoder(slots_nofg_sa, return_rgb=True)
                acov_sa = alpha_sa[0, 0, 0].sum().item() / (64*64)
                axes[4][col].imshow(np.clip(dec_sa[0].permute(1,2,0).cpu().numpy(), 0, 1))
                axes[4][col].set_title(f'd={d_t:.4f} acov={acov_sa:.5f}', fontsize=5)
                axes[4][col].axis('off')

        for row, label in enumerate(['GT', 'Normal', 'origApp NoFG\n(target alpha)',
                                      'BestApp+solvedD\nNoFG', 'solvedA+solvedD\nNoFG']):
            axes[row][0].set_ylabel(label, fontsize=7, rotation=0, labelpad=55, ha='right', va='center')

        # Add method labels
        for t in range(T):
            axes[3][t*2].set_facecolor('#ffe0e0')
            axes[3][t*2+1].set_facecolor('#e0e0ff')
        fig.text(0.25, 0.98, 'A: alpha mask', fontsize=10, ha='center', fontweight='bold', color='red')
        fig.text(0.75, 0.98, 'B: alpha coverage', fontsize=10, ha='center', fontweight='bold', color='blue')

        plt.suptitle(f'Slot {s_target}  [BestApp t={best_t}]', fontsize=12, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(f'{OUT}/v10_solved_slot{s_target}.png', dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Saved: v10_solved_slot{s_target}.png")

    # === Line plots: d_t comparison ===
    print("\nGenerating line plots...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fr = np.arange(T)
    colors = plt.cm.tab10(np.linspace(0, 1, len(fg_slots)))

    for idx, s in enumerate(fg_slots):
        d_origs = [results['A_mask'][s][t]['d_orig'] for t in range(T)]
        d_a = [results['A_mask'][s][t]['d_t'] for t in range(T)]
        d_b = [results['B_acov'][s][t]['d_t'] for t in range(T)]

        axes[0][0].plot(fr, d_origs, '--', color=colors[idx], alpha=0.4, label=f'Slot {s} orig')
        axes[0][0].plot(fr, d_a, '-o', markersize=3, color=colors[idx], label=f'Slot {s} A_mask')
        axes[0][1].plot(fr, d_origs, '--', color=colors[idx], alpha=0.4)
        axes[0][1].plot(fr, d_b, '-o', markersize=3, color=colors[idx], label=f'Slot {s} B_acov')

        # acov for both methods
        for mi, method in enumerate(['A_mask', 'B_acov']):
            acovs = []
            for t in range(T):
                a_t = results[method][s][t]['a_t']
                d_t = results[method][s][t]['d_t']
                slots_t = slots_c[0, t]
                keep = [s] + bg_slots
                slot_sa = torch.cat([a_t, slots_t[s, app_dim:].detach()])
                slot_sa[app_dim+2] = d_t
                slots_nofg_sa = slots_t[keep].unsqueeze(0).clone()
                slots_nofg_sa[0, 0] = slot_sa
                with torch.no_grad():
                    _, alpha_sa, _ = model.decoder(slots_nofg_sa, return_rgb=True)
                acovs.append(alpha_sa[0, 0, 0].sum().item() / (64*64))
            axes[1][mi].plot(fr, nofg_acovs[s], '--', color=colors[idx], alpha=0.4)
            axes[1][mi].plot(fr, acovs, '-o', markersize=3, color=colors[idx])

    # orig acov
    for idx, s in enumerate(fg_slots):
        axes[1][2].plot(fr, nofg_acovs[s], '--', color=colors[idx], alpha=0.4, label=f'Slot {s}')

    axes[0][0].set_title('d_t: A (alpha mask)'); axes[0][0].legend(fontsize=6); axes[0][0].grid(True, alpha=0.3)
    axes[0][1].set_title('d_t: B (acov scalar)'); axes[0][1].legend(fontsize=6); axes[0][1].grid(True, alpha=0.3)
    axes[0][2].set_title('d_t orig'); axes[0][2].grid(True, alpha=0.3)
    for idx, s in enumerate(fg_slots):
        d_origs = [results['A_mask'][s][t]['d_orig'] for t in range(T)]
        axes[0][2].plot(fr, d_origs, '-o', markersize=3, color=colors[idx])

    axes[1][0].set_title('acov: A (alpha mask)'); axes[1][0].grid(True, alpha=0.3)
    axes[1][1].set_title('acov: B (acov scalar)'); axes[1][1].grid(True, alpha=0.3)
    axes[1][2].set_title('orig acov (NoFG)'); axes[1][2].legend(fontsize=6); axes[1][2].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('Frame')

    plt.suptitle('v10: A (alpha mask MSE) vs B (acov scalar MSE)', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUT}/v10_solved_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # === Timing analysis ===
    print("\n=== Timing analysis ===")
    # Time one forward pass of decoder vs one inner loop step
    slots_t = slots_c[0, 0].unsqueeze(0)
    
    # Decoder forward only
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            model.decoder(slots_t, return_rgb=True)
    torch.cuda.synchronize()
    t_decode = (time.perf_counter() - t_start) / 100

    # One inner loop step (d_t optimization)
    d_param = torch.tensor([0.15], device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([d_param], lr=0.01)
    target_alpha = nofg_alphas[0][0]
    best_app_test = slots_c[0, 0, 0, :app_dim].detach()
    pos_test = slots_c[0, 0, 0, app_dim:app_dim+2].detach()
    bg_test = slots_c[0, 0,bg_slots].detach()

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for _ in range(100):
        target_slot = torch.cat([best_app_test, pos_test, d_param.reshape(1)])
        s = torch.cat([target_slot.reshape(1,1,-1), bg_test.reshape(1,-1,bg_test.shape[-1])], dim=1)
        _, alpha, _ = model.decoder(s, return_rgb=True)
        pred_acov = alpha[0,0,0].sum() / (64*64)
        loss = (pred_acov - 0.05) ** 2
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad(): d_param.data.clamp_(0.01, 0.5)
    torch.cuda.synchronize()
    t_inner = (time.perf_counter() - t_start) / 100

    print(f"Decoder forward: {t_decode*1000:.2f} ms")
    print(f"Inner loop step (d_t): {t_inner*1000:.2f} ms")
    print(f"Inner loop / Decoder: {t_inner/t_decode:.1f}x")

    # Estimate for full video
    n_fg = len(fg_slots)
    n_steps_d = 100
    n_steps_a = 200
    n_frames = 16
    total_inner = n_fg * n_frames * (n_steps_d + n_steps_a) * t_inner
    total_decode = n_fg * n_frames * t_decode
    print(f"\nPer-video estimate:")
    print(f"  ISA decode only: {total_decode:.2f} s")
    print(f"  Inner loop total: {total_inner:.1f} s")
    print(f"  Slowdown factor: {total_inner/total_decode:.0f}x")

    print("\nDone!")


if __name__ == '__main__':
    main()
