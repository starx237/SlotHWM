#!/usr/bin/env python3
"""
v16: 二分法求depth (在 [0.01, d_orig] 上, monotonic ascending side) + App Predictor求appearance
核心insight: BestApp覆盖最大 → 同样alpha只需更小depth → d_t ≤ d_orig
因此在 [0.01, d_orig] 上 sharp_acov 单调递增, 二分法可行
无需梯度, 无需trust region, 更快更准
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np, yaml, time, math
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


class AppPredictor(nn.Module):
    def __init__(self, encoder, app_dim=64, feat_dim=64, hidden=128):
        super().__init__()
        self.encoder = encoder
        self.feat_dim = feat_dim
        self.app_dim = app_dim
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.read = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, app_dim)
        )

    def forward(self, frames, pos, depth):
        B, T = frames.shape[:2]
        with torch.no_grad():
            feats = self.encoder(frames)
        N = feats.shape[2]
        H_feat = int(math.sqrt(N))
        feats_2d = feats.reshape(B * T, H_feat, H_feat, self.feat_dim)
        feats_2d = feats_2d.permute(0, 3, 1, 2)
        grid_x = pos[..., 0:1].reshape(B * T, 1, 1, 1)
        grid_y = pos[..., 1:2].reshape(B * T, 1, 1, 1)
        grid = torch.cat([grid_x, grid_y], dim=-1)
        sampled = F.grid_sample(feats_2d, grid, align_corners=True)
        sampled = sampled.reshape(B, T, self.feat_dim)
        x = torch.cat([sampled, depth], dim=-1)
        return self.read(x)


def bisect_depth(decoder, best_app, pos_t, d_orig, bg_t, target_sa, n_iter=12):
    slot_at_hi = torch.cat([best_app, pos_t, torch.tensor([d_orig], device='cuda')])
    with torch.no_grad():
        sl = torch.cat([slot_at_hi.reshape(1, 1, -1),
                        bg_t.reshape(1, -1, bg_t.shape[-1])], dim=1)
        _, alpha, _ = decoder(sl, return_rgb=True)
        sa_at_hi = sharpen_alpha(alpha[0, 0, 0]).sum().item() / (64 * 64)
    if sa_at_hi < target_sa:
        return d_orig

    d_lo, d_hi = 0.01, d_orig
    for _ in range(n_iter):
        d_mid = (d_lo + d_hi) / 2
        slot_v = torch.cat([best_app, pos_t, torch.tensor([d_mid], device='cuda')])
        with torch.no_grad():
            sl = torch.cat([slot_v.reshape(1, 1, -1),
                            bg_t.reshape(1, -1, bg_t.shape[-1])], dim=1)
            _, alpha, _ = decoder(sl, return_rgb=True)
            sa = sharpen_alpha(alpha[0, 0, 0]).sum().item() / (64 * 64)
        if sa < target_sa:
            d_lo = d_mid
        else:
            d_hi = d_mid
    return (d_lo + d_hi) / 2


def train_app_predictor(model, slots_c, fg_slots, bg_slots_list,
                        app_dim, frames, T, ref_d):
    print("Training App Predictor...")
    encoder = model.encoder
    predictor = AppPredictor(encoder, app_dim=app_dim, feat_dim=64, hidden=128).cuda()

    app_targets = {s: [] for s in fg_slots}
    for s in fg_slots:
        for t in range(T):
            slots_t = slots_c[0, t]
            pos_t = slots_t[s, app_dim:app_dim + 2].detach()
            d_solved = ref_d[s][t]
            d_t_tensor = torch.tensor([d_solved], device='cuda')
            a_orig = slots_t[s, :app_dim].detach()
            all_slots_t = slots_t.unsqueeze(0).clone().detach()
            with torch.no_grad():
                dec_full, _, _ = model.decoder(slots_t.unsqueeze(0), return_rgb=True)
            target_img = dec_full.detach()
            a_t = a_orig.clone().detach().requires_grad_(True)
            opt_a = torch.optim.Adam([a_t], lr=0.02)
            for _ in range(50):
                target_slot = torch.cat([a_t, pos_t.detach(), d_t_tensor])
                sl = all_slots_t.clone()
                sl[0, s] = target_slot
                recon, _, _ = model.decoder(sl, return_rgb=True)
                l = F.mse_loss(recon, target_img)
                opt_a.zero_grad(); l.backward(); opt_a.step()
            app_targets[s].append(a_t.detach())

    train_pos, train_depth, train_app = [], [], []
    for s in fg_slots:
        for t in range(T):
            pos_t = slots_c[0, t, s, app_dim:app_dim + 2].cpu()
            d_t_val = torch.tensor([ref_d[s][t]])
            a_t = app_targets[s][t].cpu()
            train_pos.append(pos_t)
            train_depth.append(d_t_val)
            train_app.append(a_t)

    train_pos = torch.stack(train_pos).cuda()
    train_depth = torch.stack(train_depth).cuda()
    train_app = torch.stack(train_app).cuda()

    frame_list = []
    for s in fg_slots:
        for t in range(T):
            frame_list.append(frames[0, t:t + 1])
    frames_batch = torch.stack(frame_list)

    opt_pred = torch.optim.Adam(predictor.read.parameters(), lr=1e-3)
    for epoch in range(200):
        predictor.train()
        pred = predictor(frames_batch, train_pos.unsqueeze(1),
                         train_depth.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(pred, train_app)
        opt_pred.zero_grad(); loss.backward(); opt_pred.step()

    predictor.eval()
    return predictor


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

    fg_slots, bg_slots_list = [], []
    for s in range(N):
        mean_amax = np.mean([alpha_out[0, s, t].amax().item() for t in range(T)])
        max_depth = max(slots_c[0, t, s, app_dim + 2].item() for t in range(T))
        if mean_amax > 0.3 and max_depth < 0.4:
            fg_slots.append(s)
        else:
            bg_slots_list.append(s)
    print(f"FG={fg_slots}, BG={bg_slots_list}")

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
            keep = [s] + bg_slots_list
            slots_nofg = slots_c[0, t][keep].unsqueeze(0)
            with torch.no_grad():
                _, alpha_nofg, _ = model.decoder(slots_nofg, return_rgb=True)
            alpha_sharp = sharpen_alpha(alpha_nofg[0, 0, 0])
            nofg_sharp_acovs[s].append(alpha_sharp.sum().item() / (64 * 64))
            alpha_t = alpha_out[0, s, t]
            a2 = alpha_t.squeeze(0) if alpha_t.dim() == 3 else alpha_t
            per_frame_cov[s].append(a2.sum().item() / (64 * 64))

    # Step 1: Bisection for depth
    print("\n=== Step 1: Bisection for depth [0.01, d_orig] ===")
    solved = {}
    t0_total = time.time()

    for s_target in fg_slots:
        best_t = max(range(T), key=lambda t: per_frame_cov[s_target][t])
        best_app = slots_c[0, best_t, s_target, :app_dim].clone().detach()
        print(f"\nSlot {s_target}: best_t={best_t}")
        solved[s_target] = {}

        for t in range(T):
            slots_t = slots_c[0, t]
            pos_t = slots_t[s_target, app_dim:app_dim + 2].detach()
            d_orig = slots_t[s_target, app_dim + 2].item()
            bg_t = slots_t[bg_slots_list].detach()

            d_t = bisect_depth(model.decoder, best_app, pos_t, d_orig,
                               bg_t, nofg_sharp_acovs[s_target][t])

            solved[s_target][t] = {'d_t': d_t, 'd_orig': d_orig, 'best_t': best_t,
                                    'pos_t': pos_t, 'best_app': best_app}
            delta = d_t - d_orig
            flag = ' [fallback=d_orig]' if d_t == d_orig and delta == 0 else ''
            print(f"  t={t:>2}: d={d_orig:.4f}→{d_t:.4f} ({delta:+.4f}){flag}")

    t_depth = time.time() - t0_total
    print(f"\nDepth bisection time: {t_depth:.2f}s")

    # Step 2: Train App Predictor
    print("\n=== Step 2: Train App Predictor ===")
    t0 = time.time()
    ref_d = {s: [solved[s][t]['d_t'] for t in range(T)] for s in fg_slots}
    app_pred = train_app_predictor(model, slots_c, fg_slots, bg_slots_list,
                                   app_dim, frames, T, ref_d)
    t_app_train = time.time() - t0
    print(f"App Predictor training time: {t_app_train:.1f}s")

    # Step 3: Inference with App Predictor
    print("\n=== Step 3: App Predictor inference ===")
    t0 = time.time()
    for s_target in fg_slots:
        best_t = solved[s_target][0]['best_t']
        frame_list = [frames[0, t:t + 1] for t in range(T)]
        frames_batch = torch.stack(frame_list)
        pos_batch = torch.stack([solved[s_target][t]['pos_t'].cpu() for t in range(T)])
        depth_batch = torch.tensor([[solved[s_target][t]['d_t']] for t in range(T)])

        frames_input = frames_batch.squeeze(1).unsqueeze(0)
        with torch.no_grad():
            pred_app = app_pred(frames_input.cuda(),
                                pos_batch.unsqueeze(0).cuda(),
                                depth_batch.unsqueeze(0).cuda())
            pred_app = pred_app.squeeze(0)

        for t in range(T):
            solved[s_target][t]['a_t'] = pred_app[t].detach()

    t_app_infer = time.time() - t0
    print(f"App Predictor inference time: {t_app_infer * 1000:.1f}ms")

    print(f"\n{'=' * 60}")
    print("TIMING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Depth (bisection):     {t_depth:.2f}s")
    print(f"App Predictor train:   {t_app_train:.1f}s (one-time)")
    print(f"App Predictor infer:   {t_app_infer * 1000:.1f}ms")
    print(f"Pipeline per sample:   {t_depth + t_app_infer:.2f}s (excluding training)")

    # Quality assessment
    print(f"\n=== Quality: Reconstruction MSE ===")
    for s in fg_slots:
        mses_orig, mses_solved = [], []
        for t in range(T):
            slots_t = slots_c[0, t].unsqueeze(0)
            with torch.no_grad():
                recon_orig, _, _ = model.decoder(slots_t, return_rgb=True)
            mses_orig.append(F.mse_loss(recon_orig, origin_imgs[t].unsqueeze(0)).item())

            a_t = solved[s][t]['a_t']
            d_t = solved[s][t]['d_t']
            pos_t = solved[s][t]['pos_t']
            slot_solved = torch.cat([a_t, pos_t, torch.tensor([d_t], device='cuda')])
            slots_solved = slots_c[0, t].unsqueeze(0).clone()
            slots_solved[0, s] = slot_solved
            with torch.no_grad():
                recon_solved, _, _ = model.decoder(slots_solved, return_rgb=True)
            mses_solved.append(F.mse_loss(recon_solved, origin_imgs[t].unsqueeze(0)).item())

        print(f"  Slot {s}: orig_MSE={np.mean(mses_orig):.6f}, solved_MSE={np.mean(mses_solved):.6f}, "
              f"ratio={np.mean(mses_solved) / np.mean(mses_orig):.4f}")

    # 5-row visualization
    print("\nGenerating per-slot images...")
    for s_target in fg_slots:
        best_t = solved[s_target][0]['best_t']
        best_app = slots_c[0, best_t, s_target, :app_dim].clone().detach()

        fig, axes = plt.subplots(5, T, figsize=(2.8 * T, 13))

        for t in range(T):
            gt = sample['video'][t].permute(1, 2, 0).cpu().numpy()
            gt = np.clip((gt + 1) / 2, 0, 1)
            d_orig = solved[s_target][t]['d_orig']
            d_t = solved[s_target][t]['d_t']
            a_t = solved[s_target][t]['a_t']
            pos_t = solved[s_target][t]['pos_t']

            axes[0][t].imshow(gt)
            axes[0][t].set_title(f't={t}', fontsize=8)
            axes[0][t].axis('off')

            axes[1][t].imshow(np.clip(origin_imgs[t].permute(1, 2, 0).cpu().numpy(), 0, 1))
            axes[1][t].set_title(f'd={d_orig:.4f}', fontsize=7)
            axes[1][t].axis('off')

            # Row 3: orig app + orig depth NoFG decode (target)
            slots_t = slots_c[0, t]
            keep = [s_target] + bg_slots_list
            slots_nofg = slots_t[keep].unsqueeze(0)
            with torch.no_grad():
                dec_nofg, alpha_nofg, _ = model.decoder(slots_nofg, return_rgb=True)
            acov_nofg = alpha_nofg[0, 0, 0].sum().item() / (64 * 64)
            pixcov_nofg = (alpha_nofg[0, :, 0].argmax(dim=0) == 0).sum().item() / (64 * 64)
            axes[2][t].imshow(np.clip(dec_nofg[0].permute(1, 2, 0).cpu().numpy(), 0, 1))
            axes[2][t].set_title(f'd={d_orig:.4f} acov={acov_nofg:.5f}\npixcov={pixcov_nofg:.5f}',
                                 fontsize=5)
            axes[2][t].axis('off')

            # Row 4: BestApp + bisect d_t NoFG decode
            slot_sd = slots_t[s_target].clone()
            slot_sd[:app_dim] = best_app
            slot_sd[app_dim + 2] = d_t
            slots_nofg_sd = slots_nofg.clone()
            slots_nofg_sd[0, 0] = slot_sd
            with torch.no_grad():
                dec_sd, alpha_sd, _ = model.decoder(slots_nofg_sd, return_rgb=True)
            acov_sd = alpha_sd[0, 0, 0].sum().item() / (64 * 64)
            pixcov_sd = (alpha_sd[0, :, 0].argmax(dim=0) == 0).sum().item() / (64 * 64)
            sa_sd = sharpen_alpha(alpha_sd[0, 0, 0]).sum().item() / (64 * 64)
            axes[3][t].imshow(np.clip(dec_sd[0].permute(1, 2, 0).cpu().numpy(), 0, 1))
            d_delta = d_t - d_orig
            d_color = 'red' if d_delta < -0.01 else ('blue' if d_delta > 0.01 else 'black')
            axes[3][t].set_title(
                f'd={d_t:.4f} ({d_delta:+.4f})\nacov={acov_sd:.5f} sa={sa_sd:.5f}',
                fontsize=5, color=d_color)
            axes[3][t].axis('off')

            # Row 5: predicted a_t + bisect d_t NoFG decode
            slot_sa = torch.cat([a_t, slots_t[s_target, app_dim:].detach()])
            slot_sa[app_dim + 2] = d_t
            slots_nofg_sa = slots_nofg.clone()
            slots_nofg_sa[0, 0] = slot_sa
            with torch.no_grad():
                dec_sa, alpha_sa, _ = model.decoder(slots_nofg_sa, return_rgb=True)
            acov_sa = alpha_sa[0, 0, 0].sum().item() / (64 * 64)
            sa_sa = sharpen_alpha(alpha_sa[0, 0, 0]).sum().item() / (64 * 64)
            axes[4][t].imshow(np.clip(dec_sa[0].permute(1, 2, 0).cpu().numpy(), 0, 1))
            axes[4][t].set_title(f'd={d_t:.4f} acov={acov_sa:.5f}\nsa={sa_sa:.5f}', fontsize=5)
            axes[4][t].axis('off')

        for row, label in enumerate(['GT', 'Normal ISA', 'origApp+origD\n(NoFG, target)',
                                      'BestApp+bisectD\n(NoFG, bisection)',
                                      'predApp+bisectD\n(NoFG, AppPred)']):
            axes[row][0].set_ylabel(label, fontsize=7, rotation=0, labelpad=60,
                                    ha='right', va='center')

        plt.suptitle(f'Slot {s_target}  [BestApp t={best_t}]', fontsize=12,
                     fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{OUT}/v16_solved_slot{s_target}.png', dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Saved: v16_solved_slot{s_target}.png")

    # Line plots
    print("\nGenerating line plots...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    fr = np.arange(T)
    colors = plt.cm.tab10(np.linspace(0, 1, len(fg_slots)))

    for idx, s in enumerate(fg_slots):
        d_origs = [solved[s][t]['d_orig'] for t in range(T)]
        d_ts = [solved[s][t]['d_t'] for t in range(T)]
        acovs_orig, acovs_solved, pixcovs_orig, pixcovs_solved = [], [], [], []

        for t in range(T):
            slots_t = slots_c[0, t]
            keep = [s] + bg_slots_list
            slots_nofg = slots_t[keep].unsqueeze(0)
            with torch.no_grad():
                _, alpha_o, _ = model.decoder(slots_nofg, return_rgb=True)
            acovs_orig.append(alpha_o[0, 0, 0].sum().item() / (64 * 64))
            pixcovs_orig.append((alpha_o[0, :, 0].argmax(dim=0) == 0).sum().item() / (64 * 64))

            a_t = solved[s][t]['a_t']
            d_t = solved[s][t]['d_t']
            slot_sa = torch.cat([a_t, slots_t[s, app_dim:].detach()])
            slot_sa[app_dim + 2] = d_t
            slots_nofg_sa = slots_nofg.clone()
            slots_nofg_sa[0, 0] = slot_sa
            with torch.no_grad():
                _, alpha_sa, _ = model.decoder(slots_nofg_sa, return_rgb=True)
            acovs_solved.append(alpha_sa[0, 0, 0].sum().item() / (64 * 64))
            pixcovs_solved.append((alpha_sa[0, :, 0].argmax(dim=0) == 0).sum().item() / (64 * 64))

        axes[0].plot(fr, d_origs, '--', color=colors[idx], alpha=0.3, label=f'Slot {s} orig')
        axes[0].plot(fr, d_ts, 's-', markersize=3, color=colors[idx],
                     label=f'Slot {s} bisect')
        axes[1].plot(fr, pixcovs_orig, '--', color=colors[idx], alpha=0.3)
        axes[1].plot(fr, pixcovs_solved, 's-', markersize=3, color=colors[idx])
        axes[2].plot(fr, acovs_orig, '--', color=colors[idx], alpha=0.3)
        axes[2].plot(fr, acovs_solved, 's-', markersize=3, color=colors[idx])

    axes[0].set_xlabel('Frame'); axes[0].set_ylabel('Depth')
    axes[0].set_title('d_t: orig / bisect')
    axes[0].legend(fontsize=6); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Frame'); axes[1].set_ylabel('Pixel Coverage')
    axes[1].set_title('pixcov (NoFG)'); axes[1].grid(True, alpha=0.3)
    axes[2].set_xlabel('Frame'); axes[2].set_ylabel('Alpha Coverage')
    axes[2].set_title('acov (NoFG)'); axes[2].grid(True, alpha=0.3)

    plt.suptitle('v16: Bisection depth + AppPredictor', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUT}/v16_solved_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for s in fg_slots:
        best_t_s = solved[s][0]['best_t']
        print(f"\nSlot {s} [best_t={best_t_s}]:")
        for t in range(T):
            d_o = solved[s][t]['d_orig']
            d_s = solved[s][t]['d_t']
            print(f"  t={t:>2}: d_orig={d_o:.4f}, d_bisect={d_s:.4f}, "
                  f"delta={d_s - d_o:+.4f}")


if __name__ == '__main__':
    main()
