#!/usr/bin/env python3
"""
v13: 收敛速度实验
1. d_t: 比较 3 种初始化 (d_orig, d_best, d_prev_solved) 的收敛曲线
2. a_t: 比较 3 种初始化 (a_orig, best_app, a_prev_solved) 的收敛曲线
3. 找最少步数
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


def optimize_d_trajectory(decoder, best_app, pos_t, d_init, bg_slots_t,
                          target_sharp_acov, max_steps=50, lr=0.01, tau=TAU):
    """返回每步的 d 值和 loss"""
    d_t = torch.tensor([d_init], device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([d_t], lr=lr)
    d_traj, loss_traj = [d_init], []
    for step in range(max_steps):
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
        d_traj.append(d_t.item())
        loss_traj.append(loss.item())
    return d_traj, loss_traj


def optimize_a_trajectory(decoder, a_init, pos_t, d_t_val, all_slots_t, target_idx,
                          target_img, max_steps=50, lr=0.02):
    """返回每步的 MSE loss"""
    a_t = a_init.clone().detach().requires_grad_(True)
    d_t_tensor = torch.tensor([d_t_val], device='cuda')
    optimizer = torch.optim.Adam([a_t], lr=lr)
    loss_traj = []
    for step in range(max_steps):
        target_slot = torch.cat([a_t, pos_t.detach(), d_t_tensor])
        slots = all_slots_t.clone()
        slots[0, target_idx] = target_slot
        recon, _, _ = decoder(slots, return_rgb=True)
        loss = nn.functional.mse_loss(recon, target_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_traj.append(loss.item())
    return loss_traj


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

    print("Pre-computing targets...")
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

    # 先跑一次完整的 v11-style 优化获取 "ground truth" solved 值
    print("Computing reference solved values (50 steps each)...")
    ref_d = {s: {} for s in fg_slots}
    ref_a = {s: {} for s in fg_slots}
    for s_target in fg_slots:
        best_t = max(range(T), key=lambda t: per_frame_cov[s_target][t])
        best_app = slots_c[0, best_t, s_target, :app_dim].clone().detach()
        for t in range(T):
            slots_t = slots_c[0, t]
            pos_t = slots_t[s_target, app_dim:app_dim+2].detach()
            d_orig = slots_t[s_target, app_dim+2].item()
            bg_slots_t = slots_t[bg_slots].detach()
            d_traj, _ = optimize_d_trajectory(
                model.decoder, best_app, pos_t, d_orig, bg_slots_t,
                nofg_sharp_acovs[s_target][t], max_steps=50, lr=0.01)
            ref_d[s_target][t] = d_traj[-1]
            a_orig = slots_t[s_target, :app_dim].detach()
            target_origin = origin_imgs[t].unsqueeze(0)
            all_slots_t = slots_t.unsqueeze(0).clone().detach()
            a_traj = optimize_a_trajectory(
                model.decoder, a_orig, pos_t, ref_d[s_target][t],
                all_slots_t, s_target, target_origin, max_steps=50, lr=0.02)
            ref_a[s_target][t] = a_traj[-1]  # final loss

    # ============================================================
    # Experiment 1: d_t 收敛速度
    # ============================================================
    print("\n=== Experiment 1: d_t convergence ===")
    MAX_STEPS = 50
    test_slots = fg_slots
    test_frames = list(range(T))

    d_inits = {}  # {s: {t: {init_name: d_init_value}}}
    for s in test_slots:
        best_t = max(range(T), key=lambda t: per_frame_cov[s][t])
        d_best_val = slots_c[0, best_t, s, app_dim+2].item()
        d_inits[s] = {}
        for t in test_frames:
            d_orig_val = slots_c[0, t, s, app_dim+2].item()
            d_inits[s][t] = {
                'd_orig': d_orig_val,
                'd_best': d_best_val,
                'd_prev': ref_d[s][max(0, t-1)] if t > 0 else d_orig_val
            }

    d_convergence = {}  # {init_name: {step: mean_abs_error}}
    for init_name in ['d_orig', 'd_best', 'd_prev']:
        step_errors = {step: [] for step in range(MAX_STEPS + 1)}
        for s in test_slots:
            best_t = max(range(T), key=lambda t: per_frame_cov[s][t])
            best_app = slots_c[0, best_t, s, :app_dim].clone().detach()
            for t in test_frames:
                slots_t = slots_c[0, t]
                pos_t = slots_t[s, app_dim:app_dim+2].detach()
                bg_slots_t = slots_t[bg_slots].detach()
                d_init = d_inits[s][t][init_name]
                d_traj, _ = optimize_d_trajectory(
                    model.decoder, best_app, pos_t, d_init, bg_slots_t,
                    nofg_sharp_acovs[s][t], max_steps=MAX_STEPS, lr=0.01)
                ref_val = ref_d[s][t]
                for step, d_val in enumerate(d_traj):
                    step_errors[step].append(abs(d_val - ref_val))
        d_convergence[init_name] = {
            step: np.mean(errs) for step, errs in step_errors.items()
        }
        print(f"  {init_name}: step1={d_convergence[init_name][1]:.6f}, "
              f"step5={d_convergence[init_name][5]:.6f}, "
              f"step10={d_convergence[init_name][10]:.6f}, "
              f"step20={d_convergence[init_name][20]:.6f}, "
              f"step50={d_convergence[init_name][50]:.6f}")

    # ============================================================
    # Experiment 2: a_t 收敛速度
    # ============================================================
    print("\n=== Experiment 2: a_t convergence ===")
    MAX_STEPS_A = 50

    a_convergence = {}  # {init_name: {step: mean_loss}}
    for init_name in ['a_orig', 'best_app', 'a_prev']:
        step_losses = {step: [] for step in range(MAX_STEPS_A)}
        for s in test_slots:
            best_t = max(range(T), key=lambda t: per_frame_cov[s][t])
            best_app = slots_c[0, best_t, s, :app_dim].clone().detach()
            for t in test_frames:
                slots_t = slots_c[0, t]
                pos_t = slots_t[s, app_dim:app_dim+2].detach()
                a_orig = slots_t[s, :app_dim].detach()
                d_solved = ref_d[s][t]
                target_origin = origin_imgs[t].unsqueeze(0)
                all_slots_t = slots_t.unsqueeze(0).clone().detach()

                if init_name == 'a_orig':
                    a_init = a_orig
                elif init_name == 'best_app':
                    a_init = best_app
                else:  # a_prev: 使用上一帧的 ref solved loss 近似
                    # 用上一帧的 a_orig 作为近似（真正的 a_prev_solved 需要存储）
                    a_init = slots_c[0, max(0, t-1), s, :app_dim].detach()

                loss_traj = optimize_a_trajectory(
                    model.decoder, a_init, pos_t, d_solved,
                    all_slots_t, s, target_origin, max_steps=MAX_STEPS_A, lr=0.02)
                for step, loss_val in enumerate(loss_traj):
                    step_losses[step].append(loss_val)
        a_convergence[init_name] = {
            step: np.mean(losses) for step, losses in step_losses.items()
        }
        print(f"  {init_name}: step1={a_convergence[init_name][0]:.6f}, "
              f"step5={a_convergence[init_name][4]:.6f}, "
              f"step10={a_convergence[init_name][9]:.6f}, "
              f"step20={a_convergence[init_name][19]:.6f}, "
              f"step50={a_convergence[init_name][49]:.6f}")

    # ============================================================
    # Plotting
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # d_t: absolute error vs step
    for init_name, color, label in [('d_orig', 'blue', 'init=d_orig'),
                                     ('d_best', 'green', 'init=d_best'),
                                     ('d_prev', 'red', 'init=d_prev')]:
        steps = sorted(d_convergence[init_name].keys())
        errors = [d_convergence[init_name][s] for s in steps]
        axes[0][0].plot(steps, errors, color=color, label=label, linewidth=1.5)
    axes[0][0].set_xlabel('Step'); axes[0][0].set_ylabel('Mean |d_t - d_ref|')
    axes[0][0].set_title('d_t convergence (error vs reference)')
    axes[0][0].legend(); axes[0][0].grid(True, alpha=0.3)
    axes[0][0].set_xlim(0, MAX_STEPS)

    # d_t: log scale
    for init_name, color, label in [('d_orig', 'blue', 'init=d_orig'),
                                     ('d_best', 'green', 'init=d_best'),
                                     ('d_prev', 'red', 'init=d_prev')]:
        steps = sorted(d_convergence[init_name].keys())
        errors = [d_convergence[init_name][s] for s in steps]
        errors_safe = [max(e, 1e-8) for e in errors]
        axes[0][1].semilogy(steps, errors_safe, color=color, label=label, linewidth=1.5)
    axes[0][1].set_xlabel('Step'); axes[0][1].set_ylabel('Mean |d_t - d_ref| (log)')
    axes[0][1].set_title('d_t convergence (log scale)')
    axes[0][1].legend(); axes[0][1].grid(True, alpha=0.3)

    # a_t: loss vs step
    for init_name, color, label in [('a_orig', 'blue', 'init=a_orig'),
                                     ('best_app', 'green', 'init=BestApp'),
                                     ('a_prev', 'red', 'init=a_prev')]:
        steps = sorted(a_convergence[init_name].keys())
        losses = [a_convergence[init_name][s] for s in steps]
        axes[1][0].plot(steps, losses, color=color, label=label, linewidth=1.5)
    axes[1][0].set_xlabel('Step'); axes[1][0].set_ylabel('Mean MSE loss')
    axes[1][0].set_title('a_t convergence (reconstruction loss)')
    axes[1][0].legend(); axes[1][0].grid(True, alpha=0.3)

    # a_t: log scale
    for init_name, color, label in [('a_orig', 'blue', 'init=a_orig'),
                                     ('best_app', 'green', 'init=BestApp'),
                                     ('a_prev', 'red', 'init=a_prev')]:
        steps = sorted(a_convergence[init_name].keys())
        losses = [max(a_convergence[init_name][s], 1e-10) for s in steps]
        axes[1][1].semilogy(steps, losses, color=color, label=label, linewidth=1.5)
    axes[1][1].set_xlabel('Step'); axes[1][1].set_ylabel('Mean MSE loss (log)')
    axes[1][1].set_title('a_t convergence (log scale)')
    axes[1][1].legend(); axes[1][1].grid(True, alpha=0.3)

    plt.suptitle('Convergence Speed: d_t and a_t with different initializations', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUT}/v13_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: v13_convergence.png")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Minimum steps for convergence")
    print(f"{'='*70}")

    def find_min_steps(convergence, threshold_ratio=0.05, max_steps=50):
        """找到 error 降到 ref 的 5% 以内的最少步数"""
        final_err = convergence[max_steps]
        if final_err < 1e-8:
            return 0
        threshold = max(final_err, convergence[max_steps] * 1.1) * threshold_ratio
        for step in range(max_steps + 1):
            if convergence[step] <= threshold:
                return step
        return max_steps

    print("\nd_t convergence (error < 5% of initial):")
    for init_name in ['d_orig', 'd_best', 'd_prev']:
        min_steps = find_min_steps(d_convergence[init_name], 0.05, MAX_STEPS)
        init_err = d_convergence[init_name][0]
        final_err = d_convergence[init_name][MAX_STEPS]
        print(f"  {init_name:10s}: min_steps={min_steps:3d}, "
              f"init_err={init_err:.6f}, final_err={final_err:.6f}")

    print("\na_t convergence (loss at 95% of total reduction):")
    for init_name in ['a_orig', 'best_app', 'a_prev']:
        init_loss = a_convergence[init_name][0]
        final_loss = a_convergence[init_name][MAX_STEPS_A - 1]
        total_reduction = init_loss - final_loss
        if total_reduction < 1e-10:
            min_steps = 0
        else:
            threshold = final_loss + 0.05 * total_reduction
            min_steps = MAX_STEPS_A
            for step in range(MAX_STEPS_A):
                if a_convergence[init_name][step] <= threshold:
                    min_steps = step + 1
                    break
        print(f"  {init_name:10s}: min_steps={min_steps:3d}, "
              f"init_loss={init_loss:.6f}, final_loss={final_loss:.6f}")


if __name__ == '__main__':
    main()
