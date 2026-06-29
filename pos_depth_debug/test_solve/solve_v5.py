#!/usr/bin/env python3
"""
Solved Depth v5: 严格按用户原始设计
1. d_t: BestApp + NoFG decode target (所有帧), 带 loss 改善检查
2. a_t: solved d_t + origin full decode target (所有帧)
3. 完整可视化 + 折线图
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


def solve_depth(decoder, best_app, pos_t, d_init, bg_slots_t, target_img,
                n_steps=300, lr=0.005, loss_improve_thresh=0.3):
    """
    用 BestApp, NoFG decode target 优化 d_t
    如果 loss 改善 < loss_improve_thresh, 保持 d_orig (BestApp 不适合该帧)
    """
    d_t = torch.tensor([d_init], device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([d_t], lr=lr)
    losses = []
    best_loss = float('inf')
    best_d = d_init

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
        loss_val = loss.item()
        losses.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            best_d = d_t.item()

    # 如果 loss 改善不足, 回退到 d_orig
    if losses[0] > 1e-8:
        improve_ratio = (losses[0] - best_loss) / losses[0]
    else:
        improve_ratio = 1.0  # loss 已经接近 0, d_orig 就是最优

    if improve_ratio < loss_improve_thresh and losses[0] > 1e-7:
        return d_init, losses, False
    return best_d, losses, True


def solve_appearance(decoder, a_init, pos_t, d_t_val, all_slots_t, target_idx,
                     target_img, n_steps=300, lr=0.01):
    """用 solved d_t, 全 slot decode target 优化 a_t"""
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

    ds = OBJ3DDataset(data_path='data/obj3d', num_frames=16, stride=4, subsample=2)
    sample = ds[9]
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
    per_frame_cov = {s: [] for s in fg_slots}
    per_frame_acov_nofg = {s: [] for s in fg_slots}

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
            per_frame_acov_nofg[s].append(alpha_nofg[0, 0, 0].sum().item() / (64*64))

            alpha_t = alpha_out[0, s, t]
            a2 = alpha_t.squeeze(0) if alpha_t.dim() == 3 else alpha_t
            per_frame_cov[s].append(a2.sum().item() / (64 * 64))

    # === Solve ===
    print("\nSolving v5 (BestApp+NoFG→d_t, fullDecode→a_t, ALL frames)...")
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

            # Step 1: d_t with BestApp + NoFG target
            target_nofg = nofg_imgs[s_target][t].unsqueeze(0)
            d_t, d_losses, d_improved = solve_depth(
                model.decoder, best_app, pos_t, d_orig, bg_slots_t,
                target_nofg, n_steps=300, lr=0.005, loss_improve_thresh=0.3)

            # Step 2: a_t with solved d_t + origin full decode target
            target_origin = origin_imgs[t].unsqueeze(0)
            all_slots_t = slots_t.unsqueeze(0).clone().detach()
            a_t, a_losses = solve_appearance(
                model.decoder, a_orig, pos_t, d_t, all_slots_t, s_target,
                target_origin, n_steps=300, lr=0.01)

            solved[s_target][t] = {
                'd_t': d_t, 'a_t': a_t, 'd_orig': d_orig,
                'd_improved': d_improved,
                'd_loss_start': d_losses[0], 'd_loss_end': d_losses[-1],
                'a_loss_start': a_losses[0], 'a_loss_end': a_losses[-1],
            }
            elapsed = time.time() - t0
            status = 'SOLVED' if d_improved else 'KEPT'
            print(f"  t={t:>2}: d={d_orig:.4f}→{d_t:.4f} ({d_t-d_orig:+.4f}) [{status}]  "
                  f"d_loss: {d_losses[0]:.6f}→{d_losses[-1]:.6f}  "
                  f"a_loss: {a_losses[0]:.6f}→{a_losses[-1]:.6f}  [{elapsed:.0f}s]")

    # === Per-slot 5-row images ===
    print("\nGenerating per-slot images...")
    for s_target in fg_slots:
        best_cov_t = max(range(T), key=lambda t: per_frame_cov[s_target][t])
        best_app = slots_c[0, best_cov_t, s_target, :app_dim].clone().detach()

        fig, axes = plt.subplots(5, T, figsize=(2.8 * T, 13))
        for t in range(T):
            gt = sample['video'][t].permute(1,2,0).cpu().numpy()
            gt = np.clip((gt + 1) / 2, 0, 1)
            d_orig = solved[s_target][t]['d_orig']
            d_t = solved[s_target][t]['d_t']
            a_t = solved[s_target][t]['a_t']
            d_improved = solved[s_target][t]['d_improved']

            axes[0][t].imshow(gt); axes[0][t].set_title(f't={t}', fontsize=8); axes[0][t].axis('off')

            axes[1][t].imshow(np.clip(origin_imgs[t].permute(1,2,0).cpu().numpy(), 0, 1))
            axes[1][t].set_title(f'd={d_orig:.4f}', fontsize=7); axes[1][t].axis('off')

            # Row 3: BestApp + origin depth NoFG
            slots_t = slots_c[0, t]
            keep = [s_target] + bg_slots
            slots_nofg = slots_t[keep].unsqueeze(0)
            slot_ba = slots_t[s_target].clone()
            slot_ba[:app_dim] = best_app
            slots_nofg_ba = slots_nofg.clone()
            slots_nofg_ba[0, 0] = slot_ba
            with torch.no_grad():
                dec_ba, alpha_ba, _ = model.decoder(slots_nofg_ba, return_rgb=True)
            acov_ba = alpha_ba[0, 0, 0].sum().item() / (64*64)
            axes[2][t].imshow(np.clip(dec_ba[0].permute(1,2,0).cpu().numpy(), 0, 1))
            axes[2][t].set_title(f'd={d_orig:.4f} acov={acov_ba:.4f}', fontsize=6); axes[2][t].axis('off')

            # Row 4: BestApp + solved d_t NoFG
            slot_sd = slot_ba.clone()
            slot_sd[app_dim+2] = d_t
            slots_nofg_sd = slots_nofg.clone()
            slots_nofg_sd[0, 0] = slot_sd
            with torch.no_grad():
                dec_sd, alpha_sd, _ = model.decoder(slots_nofg_sd, return_rgb=True)
            acov_sd = alpha_sd[0, 0, 0].sum().item() / (64*64)
            axes[3][t].imshow(np.clip(dec_sd[0].permute(1,2,0).cpu().numpy(), 0, 1))
            status_color = 'green' if d_improved else 'gray'
            axes[3][t].set_title(f'd={d_t:.4f} acov={acov_sd:.4f}', fontsize=6,
                                 color=status_color); axes[3][t].axis('off')

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

        for row, label in enumerate(['GT', 'Normal ISA', 'BestApp+origD\n(NoFG)', 'BestApp+solvedD\n(NoFG)', 'solvedA+solvedD\n(NoFG)']):
            axes[row][0].set_ylabel(label, fontsize=8, rotation=0, labelpad=55, ha='right', va='center')
        plt.suptitle(f'Slot {s_target} Solved Depth v5', fontsize=13)
        plt.tight_layout()
        plt.savefig(f'{OUT}/v5_solved_slot{s_target}.png', dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Saved: v5_solved_slot{s_target}.png")

    # === Overall comparison ===
    print("\nGenerating overall comparison...")
    mse_normal, mse_solved = [], []
    fig, axes = plt.subplots(3, T, figsize=(2.8 * T, 8))
    for t in range(T):
        gt = sample['video'][t].cuda()
        gt_np = np.clip((sample['video'][t].permute(1,2,0).numpy() + 1) / 2, 0, 1)
        axes[0][t].imshow(gt_np); axes[0][t].set_title(f't={t}', fontsize=8); axes[0][t].axis('off')

        recon_normal = origin_imgs[t]
        axes[1][t].imshow(np.clip(recon_normal.permute(1,2,0).cpu().numpy(), 0, 1))
        mse_n = nn.functional.mse_loss(recon_normal.cuda(), gt).item()
        mse_normal.append(mse_n)
        axes[1][t].set_title(f'MSE={mse_n:.6f}', fontsize=7); axes[1][t].axis('off')

        slots_t = slots_c[0, t].clone()
        for s in fg_slots:
            slots_t[s, :app_dim] = solved[s][t]['a_t']
            slots_t[s, app_dim+2] = solved[s][t]['d_t']
        with torch.no_grad():
            dec_solved, _, _ = model.decoder(slots_t.unsqueeze(0), return_rgb=True)
        mse_s = nn.functional.mse_loss(dec_solved[0].cuda(), gt).item()
        mse_solved.append(mse_s)
        axes[2][t].imshow(np.clip(dec_solved[0].permute(1,2,0).cpu().numpy(), 0, 1))
        axes[2][t].set_title(f'MSE={mse_s:.6f}', fontsize=7, color='green' if mse_s < mse_n else 'red')
        axes[2][t].axis('off')

    for row, label in enumerate(['GT', f'Normal (mean={np.mean(mse_normal):.6f})',
                                  f'Solved (mean={np.mean(mse_solved):.6f})']):
        axes[row][0].set_ylabel(label, fontsize=8, rotation=0, labelpad=65, ha='right', va='center')
    plt.suptitle('v5: Normal vs Solved Full Decode', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUT}/v5_solved_overall.png', dpi=130, bbox_inches='tight')
    plt.close()

    # === Line plots: d_t, acov, pixcov ===
    print("\nGenerating line plots...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    fr = np.arange(T)
    colors = plt.cm.tab10(np.linspace(0, 1, len(fg_slots)))

    for idx, s in enumerate(fg_slots):
        d_origs = [solved[s][t]['d_orig'] for t in range(T)]
        d_ts = [solved[s][t]['d_t'] for t in range(T)]
        acovs_nofg_orig, acovs_nofg_solved = [], []
        pixcovs_orig, pixcovs_solved = [], []

        for t in range(T):
            slots_t = slots_c[0, t]
            keep = [s] + bg_slots
            slots_nofg = slots_t[keep].unsqueeze(0)

            # orig NoFG decode
            with torch.no_grad():
                _, alpha_orig, _ = model.decoder(slots_nofg, return_rgb=True)
            acovs_nofg_orig.append(alpha_orig[0, 0, 0].sum().item() / (64*64))
            dominant_o = alpha_orig[0, :, 0].argmax(dim=0)
            pixcovs_orig.append((dominant_o == 0).sum().item() / (64*64))

            # solved NoFG decode
            a_t = solved[s][t]['a_t']
            d_t = solved[s][t]['d_t']
            slot_sa = torch.cat([a_t, slots_t[s, app_dim:].detach()])
            slot_sa[app_dim+2] = d_t
            slots_nofg_sa = slots_nofg.clone()
            slots_nofg_sa[0, 0] = slot_sa
            with torch.no_grad():
                _, alpha_sa, _ = model.decoder(slots_nofg_sa, return_rgb=True)
            acovs_nofg_solved.append(alpha_sa[0, 0, 0].sum().item() / (64*64))
            dominant_s = alpha_sa[0, :, 0].argmax(dim=0)
            pixcovs_solved.append((dominant_s == 0).sum().item() / (64*64))

        axes[0].plot(fr, d_origs, '--', color=colors[idx], alpha=0.4, label=f'Slot {s} orig')
        axes[0].plot(fr, d_ts, '-o', markersize=3, color=colors[idx], label=f'Slot {s} solved')
        axes[1].plot(fr, acovs_nofg_orig, '--', color=colors[idx], alpha=0.4)
        axes[1].plot(fr, acovs_nofg_solved, '-o', markersize=3, color=colors[idx])
        axes[2].plot(fr, pixcovs_orig, '--', color=colors[idx], alpha=0.4)
        axes[2].plot(fr, pixcovs_solved, '-o', markersize=3, color=colors[idx])

    axes[0].set_xlabel('Frame'); axes[0].set_ylabel('Depth (=Scale)')
    axes[0].set_title('d_t: orig vs solved'); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Frame'); axes[1].set_ylabel('Alpha Coverage')
    axes[1].set_title('acov (NoFG): orig vs solved'); axes[1].grid(True, alpha=0.3)
    axes[2].set_xlabel('Frame'); axes[2].set_ylabel('Pixel Coverage')
    axes[2].set_title('pixcov (NoFG): orig vs solved'); axes[2].grid(True, alpha=0.3)

    plt.suptitle('v5: Solved d_t, acov, pixcov over 16 Frames', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUT}/v5_solved_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # === Summary ===
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for s in fg_slots:
        n_solved = sum(1 for t in range(T) if solved[s][t]['d_improved'])
        print(f"Slot {s}: {n_solved}/{T} frames d_t solved, {T-n_solved} kept d_orig")
        for t in range(T):
            d_o = solved[s][t]['d_orig']; d_s = solved[s][t]['d_t']
            st = 'SOLVED' if solved[s][t]['d_improved'] else 'KEPT'
            print(f"  t={t:>2}: d={d_o:.4f}→{d_s:.4f} ({d_s-d_o:+.4f}) [{st}]")

    print(f"\nMSE: Normal={np.mean(mse_normal):.6f}, Solved={np.mean(mse_solved):.6f}")
    print(f"Improvement: {(1 - np.mean(mse_solved)/np.mean(mse_normal))*100:.2f}%")
    print(f"\nPer-frame MSE comparison:")
    for t in range(T):
        better = 'S' if mse_solved[t] < mse_normal[t] else 'N'
        print(f"  t={t:>2}: Normal={mse_normal[t]:.6f} Solved={mse_solved[t]:.6f} [{better}]")


if __name__ == '__main__':
    main()
