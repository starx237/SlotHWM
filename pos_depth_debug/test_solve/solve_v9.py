#!/usr/bin/env python3
"""
Solved Depth v9: Row3 = orig app NoFG decode (优化target), 标注BestApp帧
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
from data.obj3d_dataset import OBJ3DDataset
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


def solve_depth_nofg(decoder, best_app, pos_t, d_init, bg_slots_t, target_img,
                     n_steps=100, lr=0.005):
    d_t = torch.tensor([d_init], device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([d_t], lr=lr)
    losses = []
    for step in range(n_steps):
        target_slot = torch.cat([best_app.detach(), pos_t.detach(), d_t.reshape(1)])
        slots = torch.cat([
            target_slot.reshape(1, 1, -1),
            bg_slots_t.detach().reshape(1, -1, bg_slots_t.shape[-1])
        ], dim=1)
        recon, _, _ = decoder(slots, return_rgb=True)
        loss = nn.functional.mse_loss(recon, target_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            d_t.data.clamp_(0.01, 0.5)
        losses.append(loss.item())
    return d_t.item(), losses


def solve_appearance_full(decoder, a_init, pos_t, d_t_val, all_slots_t, target_idx,
                          target_img, n_steps=200, lr=0.01):
    a_t = a_init.clone().detach().requires_grad_(True)
    d_t_tensor = torch.tensor([d_t_val], device='cuda')
    optimizer = torch.optim.Adam([a_t], lr=lr)
    losses = []
    for step in range(n_steps):
        target_slot = torch.cat([a_t, pos_t.detach(), d_t_tensor])
        slots = all_slots_t.clone()
        slots[0, target_idx] = target_slot
        recon, _, _ = decoder(slots, return_rgb=True)
        loss = nn.functional.mse_loss(recon, target_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return a_t.detach(), losses


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
            nofg_alphas[s].append(alpha_nofg[0].detach())
            alpha_t = alpha_out[0, s, t]
            a2 = alpha_t.squeeze(0) if alpha_t.dim() == 3 else alpha_t
            per_frame_cov[s].append(a2.sum().item() / (64 * 64))

    # === Solve ===
    print("\nSolving...")
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

            target_nofg = nofg_imgs[s_target][t].unsqueeze(0)
            d_t, d_losses = solve_depth_nofg(
                model.decoder, best_app, pos_t, d_orig, bg_slots_t,
                target_nofg, n_steps=100, lr=0.005)

            target_origin = origin_imgs[t].unsqueeze(0)
            all_slots_t = slots_t.unsqueeze(0).clone().detach()
            a_t, a_losses = solve_appearance_full(
                model.decoder, a_orig, pos_t, d_t, all_slots_t, s_target,
                target_origin, n_steps=200, lr=0.01)

            solved[s_target][t] = {'d_t': d_t, 'a_t': a_t, 'd_orig': d_orig, 'best_t': best_t}
            elapsed = time.time() - t0
            print(f"  t={t:>2}: d={d_orig:.4f}→{d_t:.4f} ({d_t-d_orig:+.4f})  "
                  f"d_loss: {d_losses[0]:.6f}→{d_losses[-1]:.6f}  "
                  f"a_loss: {a_losses[0]:.6f}→{a_losses[-1]:.6f}  [{elapsed:.0f}s]")

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

            # Row 1: GT
            axes[0][t].imshow(gt); axes[0][t].set_title(f't={t}', fontsize=8); axes[0][t].axis('off')

            # Row 2: Normal ISA reconstruction
            axes[1][t].imshow(np.clip(origin_imgs[t].permute(1,2,0).cpu().numpy(), 0, 1))
            axes[1][t].set_title(f'd={d_orig:.4f}', fontsize=7); axes[1][t].axis('off')

            # Row 3: orig app + orig depth NoFG (这就是d_t的优化target)
            slots_t = slots_c[0, t]
            keep = [s_target] + bg_slots
            slots_nofg = slots_t[keep].unsqueeze(0)
            with torch.no_grad():
                dec_nofg_orig, alpha_nofg_orig, _ = model.decoder(slots_nofg, return_rgb=True)
            acov_orig = alpha_nofg_orig[0, 0, 0].sum().item() / (64*64)
            axes[2][t].imshow(np.clip(dec_nofg_orig[0].permute(1,2,0).cpu().numpy(), 0, 1))
            axes[2][t].set_title(f'd={d_orig:.4f} acov={acov_orig:.4f}', fontsize=6)
            axes[2][t].axis('off')

            # Row 4: BestApp + solved d_t NoFG
            slot_ba_sd = slots_t[s_target].clone()
            slot_ba_sd[:app_dim] = best_app
            slot_ba_sd[app_dim+2] = d_t
            slots_nofg_sd = slots_nofg.clone()
            slots_nofg_sd[0, 0] = slot_ba_sd
            with torch.no_grad():
                dec_sd, alpha_sd, _ = model.decoder(slots_nofg_sd, return_rgb=True)
            acov_sd = alpha_sd[0, 0, 0].sum().item() / (64*64)
            axes[3][t].imshow(np.clip(dec_sd[0].permute(1,2,0).cpu().numpy(), 0, 1))
            d_delta = d_t - d_orig
            d_color = 'red' if d_delta < -0.01 else ('blue' if d_delta > 0.01 else 'black')
            axes[3][t].set_title(f'd={d_t:.4f} acov={acov_sd:.4f}', fontsize=6, color=d_color)
            axes[3][t].axis('off')

            # Row 5: solved a_t + solved d_t NoFG
            slot_sa = torch.cat([a_t, slots_t[s_target, app_dim:].detach()])
            slot_sa[app_dim+2] = d_t
            slots_nofg_sa = slots_nofg.clone()
            slots_nofg_sa[0, 0] = slot_sa
            with torch.no_grad():
                dec_sa, alpha_sa, _ = model.decoder(slots_nofg_sa, return_rgb=True)
            acov_sa = alpha_sa[0, 0, 0].sum().item() / (64*64)
            axes[4][t].imshow(np.clip(dec_sa[0].permute(1,2,0).cpu().numpy(), 0, 1))
            axes[4][t].set_title(f'd={d_t:.4f} acov={acov_sa:.4f}', fontsize=6); axes[4][t].axis('off')

        for row, label in enumerate(['GT', 'Normal ISA', 'origApp+origD\n(NoFG, d_t target)',
                                      'BestApp+solvedD\n(NoFG)', 'solvedA+solvedD\n(NoFG)']):
            axes[row][0].set_ylabel(label, fontsize=7, rotation=0, labelpad=60, ha='right', va='center')

        plt.suptitle(f'Slot {s_target}  [BestApp from t={best_t}]', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{OUT}/v9_solved_slot{s_target}.png', dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Saved: v9_solved_slot{s_target}.png")

    # === Line plots ===
    print("\nGenerating line plots...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    fr = np.arange(T)
    colors = plt.cm.tab10(np.linspace(0, 1, len(fg_slots)))

    for idx, s in enumerate(fg_slots):
        d_origs = [solved[s][t]['d_orig'] for t in range(T)]
        d_ts = [solved[s][t]['d_t'] for t in range(T)]
        acovs_nofg_orig, acovs_nofg_solved = [], []
        pixcovs_nofg_orig, pixcovs_nofg_solved = [], []

        for t in range(T):
            acovs_nofg_orig.append(nofg_alphas[s][t][0, 0].sum().item() / (64*64))
            dominant_o = nofg_alphas[s][t][:, 0].argmax(dim=0)
            pixcovs_nofg_orig.append((dominant_o == 0).sum().item() / (64*64))

            a_t = solved[s][t]['a_t']
            d_t = solved[s][t]['d_t']
            slots_t = slots_c[0, t]
            keep = [s] + bg_slots
            slots_nofg = slots_t[keep].unsqueeze(0)
            slot_sa = torch.cat([a_t, slots_t[s, app_dim:].detach()])
            slot_sa[app_dim+2] = d_t
            slots_nofg_sa = slots_nofg.clone()
            slots_nofg_sa[0, 0] = slot_sa
            with torch.no_grad():
                _, alpha_sa, _ = model.decoder(slots_nofg_sa, return_rgb=True)
            acovs_nofg_solved.append(alpha_sa[0, 0, 0].sum().item() / (64*64))
            dominant_s = alpha_sa[0, :, 0].argmax(dim=0)
            pixcovs_nofg_solved.append((dominant_s == 0).sum().item() / (64*64))

        axes[0].plot(fr, d_origs, '--', color=colors[idx], alpha=0.4, label=f'Slot {s} orig')
        axes[0].plot(fr, d_ts, '-o', markersize=3, color=colors[idx], label=f'Slot {s} solved')
        axes[1].plot(fr, acovs_nofg_orig, '--', color=colors[idx], alpha=0.4)
        axes[1].plot(fr, acovs_nofg_solved, '-o', markersize=3, color=colors[idx])
        axes[2].plot(fr, pixcovs_nofg_orig, '--', color=colors[idx], alpha=0.4)
        axes[2].plot(fr, pixcovs_nofg_solved, '-o', markersize=3, color=colors[idx])

    axes[0].set_xlabel('Frame'); axes[0].set_ylabel('Depth (=Scale)')
    axes[0].set_title('d_t: orig vs solved'); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Frame'); axes[1].set_ylabel('acov (NoFG)')
    axes[1].set_title('acov (NoFG): orig vs solved'); axes[1].grid(True, alpha=0.3)
    axes[2].set_xlabel('Frame'); axes[2].set_ylabel('pixcov (NoFG)')
    axes[2].set_title('pixcov (NoFG): orig vs solved'); axes[2].grid(True, alpha=0.3)
    plt.suptitle('Solved d_t, acov, pixcov over 16 Frames', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUT}/v9_solved_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nDone!")


if __name__ == '__main__':
    main()
