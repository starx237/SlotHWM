#!/usr/bin/env python3
"""
Phase3 Sample9: BestApp NoFG Decode
每 FG slot 6行×16帧:
  Row1: GT
  Row2: Phase3 正常重建 (all slots)
  Row3: NoFG decode (target slot + BG slots)
  Row4: BestApp + orig depth + NoFG decode
  Row5: Row3 alpha mask 热力图
  Row6: Row4 alpha mask 热力图
每帧标注 depth
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn
import numpy as np, yaml
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset
from torch.utils.data import DataLoader
from train import Trainer, create_optimizer
from train.trainer import WandBLogger


class DepthInverter(nn.Module):
    """spread → depth inverter (spread has highest correlation with depth, r=0.93)"""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Softplus(),
        )

    def forward(self, spread):
        return self.net(spread)


def load_model(cfg_dict, ckpt_path):
    cfg = SimpleNamespace(**cfg_dict)
    model = SlotDynamicsModel(cfg)
    opt, sch = create_optimizer((p for p in model.parameters() if p.requires_grad), cfg)
    wb = WandBLogger(enabled=False)
    trainer = Trainer(model, opt, sch, cfg, wandb_logger=wb)
    trainer.load_checkpoint(ckpt_path)
    return model


def collect_inverter_data(model, ds, n_samples=5000):
    model.eval().cuda()
    app_dim = model.appearance_dim
    all_spreads, all_depths = [], []

    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    samples_done = 0
    with torch.no_grad():
        for batch in dl:
            if samples_done >= n_samples:
                break
            frames = batch["video"].cuda()
            out = model(frames)
            slots_c = out['slots']['corrected'] if isinstance(out['slots'], dict) else out['slots']
            alpha = out['alpha']
            slots_t = slots_c[:, 0]
            alpha_t = alpha[:, :, 0]
            if alpha_t.dim() == 5:
                alpha_2d = alpha_t.squeeze(2)
            else:
                alpha_2d = alpha_t
            B, N, H, W = alpha_2d.shape
            depth = slots_t[:, :, app_dim + 2]
            cov = alpha_2d.sum(dim=[-2, -1])
            a_max = alpha_2d.amax(dim=[-2, -1])
            cov_norm = cov / (H * W)
            # spread
            gy, gx = torch.meshgrid(
                torch.linspace(-1, 1, H, device='cuda'),
                torch.linspace(-1, 1, W, device='cuda'), indexing='ij')
            a_sum = alpha_2d.sum(dim=[-2, -1], keepdim=True) + 1e-8
            a_norm = alpha_2d / a_sum
            cx = (a_norm * gx.unsqueeze(0)).sum(dim=[-2, -1])
            cy = (a_norm * gy.unsqueeze(0)).sum(dim=[-2, -1])
            spread = torch.sqrt((a_norm * ((gx.unsqueeze(0) - cx.unsqueeze(-1).unsqueeze(-1))**2 +
                                            (gy.unsqueeze(0) - cy.unsqueeze(-1).unsqueeze(-1))**2)).sum(dim=[-2, -1]))
            dominant = alpha_2d.argmax(dim=1)
            one_hot = torch.zeros(B, N, H, W, device=alpha_2d.device)
            one_hot.scatter_(1, dominant.unsqueeze(1), 1)
            pixcov = one_hot.sum(dim=[-2, -1]).float()
            pixcov_consist = pixcov > cov.float() * 0.65
            fg = (cov > 20) & (cov < 1500) & (a_max > 0.7) & (depth < 0.3) & pixcov_consist

            for b in range(B):
                for s in range(N):
                    if fg[b, s]:
                        all_spreads.append(spread[b, s].item())
                        all_depths.append(depth[b, s].item())
            samples_done += B

    return np.array(all_spreads), np.array(all_depths)


def train_inverter(spreads, depths, n_epochs=3000, lr=1e-3, device='cuda'):
    sp_t = torch.tensor(spreads, dtype=torch.float32, device=device).unsqueeze(1)
    dep_t = torch.tensor(depths, dtype=torch.float32, device=device).unsqueeze(1)

    model = DepthInverter(hidden=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
        pred = model(sp_t)
        loss = nn.functional.mse_loss(pred, dep_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss={loss.item():.8f}")

    with torch.no_grad():
        pred = model(sp_t).squeeze().cpu().numpy()
    ss_res = np.sum((depths - pred) ** 2)
    ss_tot = np.sum((depths - depths.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  Inverter R² = {r2:.4f}")
    return model


def main():
    with open('config/pretrain_phase3.yaml') as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict['burnin_frames'] = 16
    ckpt_path = 'experiments/phase3_gru2_full/checkpoints/best.pt'

    # === Step 1: Run Phase3 on sample9 ===
    print("\n=== Step 3: Running on sample9 ===")
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

    # Identify FG/BG
    fg_slots, bg_slots = [], []
    for s in range(N):
        mean_amax = np.mean([alpha_out[0, s, t].amax().item() for t in range(T)])
        max_depth = max(slots_c[0, t, s, app_dim + 2].item() for t in range(T))
        if mean_amax > 0.3 and max_depth < 0.4:
            fg_slots.append(s)
        else:
            bg_slots.append(s)
    print(f"FG={fg_slots}, BG={bg_slots}")

    # Per-frame decode + metrics
    all_recon = []
    all_metrics = []
    for t in range(T):
        slots_t = slots_c[0, t].unsqueeze(0)
        dec_img, dec_alpha, _ = model.decoder(slots_t, return_rgb=True)
        all_recon.append(dec_img[0].cpu())

        alpha_t = alpha_out[0, :, t]
        alpha_2d = alpha_t.squeeze(1) if alpha_t.dim() == 4 else alpha_t
        H, W = alpha_2d.shape[-2], alpha_2d.shape[-1]
        cov_norm = alpha_2d.sum(dim=[-2, -1]) / (H * W)

        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, H, device='cuda'),
            torch.linspace(-1, 1, W, device='cuda'), indexing='ij')
        a_sum = alpha_2d.sum(dim=[-2, -1], keepdim=True) + 1e-8
        a_norm = alpha_2d / a_sum
        cx = (a_norm * gx.unsqueeze(0)).sum(dim=[-2, -1])
        cy = (a_norm * gy.unsqueeze(0)).sum(dim=[-2, -1])
        spread = torch.sqrt((a_norm * ((gx.unsqueeze(0) - cx.unsqueeze(-1).unsqueeze(-1))**2 +
                                        (gy.unsqueeze(0) - cy.unsqueeze(-1).unsqueeze(-1))**2)).sum(dim=[-2, -1]))

        metrics = {}
        for s in fg_slots:
            metrics[s] = {
                'depth': slots_t[0, s, app_dim + 2].item(),
                'cov': cov_norm[s].item(),
                'spread': spread[s].item(),
            }
        all_metrics.append(metrics)

    # === Step 2: Per-slot 8-row images ===
    print("\n=== Step 2: Generating per-slot images ===")
    for s_target in fg_slots:
        best_t = max(range(T), key=lambda t: all_metrics[t][s_target]['cov'])
        best_app = slots_c[0, best_t, s_target, :app_dim].clone()
        best_depth = slots_c[0, best_t, s_target, app_dim + 2].item()
        print(f"  Slot {s_target}: best_t={best_t}, best_cov={all_metrics[best_t][s_target]['cov']:.4f}, "
              f"best_depth={best_depth:.4f}")

        fig, axes = plt.subplots(8, T, figsize=(2.5 * T, 20))

        for t in range(T):
            gt = sample['video'][t].permute(1, 2, 0).numpy()
            gt = np.clip((gt + 1) / 2, 0, 1)
            d_orig = all_metrics[t][s_target]['depth']

            # Row 1: GT
            axes[0][t].imshow(gt)
            axes[0][t].set_title(f't={t}', fontsize=8)
            axes[0][t].axis('off')

            # Row 2: Normal decode (all slots)
            recon = all_recon[t].permute(1, 2, 0).numpy()
            axes[1][t].imshow(np.clip(recon, 0, 1))
            axes[1][t].set_title(f'd={d_orig:.3f}', fontsize=7)
            axes[1][t].axis('off')

            # Row 3: NoFG decode (target + BG slots, original slot values)
            slots_t = slots_c[0, t].unsqueeze(0)
            keep = [s_target] + bg_slots
            slots_nofg = slots_t[:, keep]
            target_idx = keep.index(s_target)
            with torch.no_grad():
                dec_nofg, alpha_r3, _ = model.decoder(slots_nofg, return_rgb=True)
            recon_nofg = dec_nofg[0].cpu().permute(1, 2, 0).numpy()
            alpha_r3_target = alpha_r3[0, target_idx, 0].cpu().numpy()
            acov_r3 = alpha_r3_target.sum() / (64 * 64)
            axes[2][t].imshow(np.clip(recon_nofg, 0, 1))
            axes[2][t].set_title(f'd={d_orig:.3f} acov={acov_r3:.4f}', fontsize=6)
            axes[2][t].axis('off')

            # Row 4: BestApp + original depth + NoFG decode
            slot_mod = slots_t[0, s_target].clone()
            slot_mod[:app_dim] = best_app
            slots_best = slots_nofg.clone()
            slots_best[0, target_idx] = slot_mod
            with torch.no_grad():
                dec_best, alpha_r4, _ = model.decoder(slots_best, return_rgb=True)
            recon_best = dec_best[0].cpu().permute(1, 2, 0).numpy()
            alpha_r4_target = alpha_r4[0, target_idx, 0].cpu().numpy()
            acov_r4 = alpha_r4_target.sum() / (64 * 64)
            axes[3][t].imshow(np.clip(recon_best, 0, 1))
            diff_acov = acov_r4 - acov_r3
            color = 'red' if abs(diff_acov) > 0.002 else 'green'
            axes[3][t].set_title(f'd={d_orig:.3f} acov={acov_r4:.4f} ({diff_acov:+.4f})', fontsize=6, color=color)
            axes[3][t].axis('off')

            # Row 5: BestApp + corrected depth (d_orig * acov_r3 / acov_r4) + NoFG decode
            corr_d = d_orig * acov_r3 / max(acov_r4, 1e-8)
            slot_corr = slot_mod.clone()
            slot_corr[app_dim + 2] = corr_d
            slots_corr = slots_nofg.clone()
            slots_corr[0, target_idx] = slot_corr
            with torch.no_grad():
                dec_corr, alpha_r5, _ = model.decoder(slots_corr, return_rgb=True)
            recon_corr = dec_corr[0].cpu().permute(1, 2, 0).numpy()
            alpha_r5_target = alpha_r5[0, target_idx, 0].cpu().numpy()
            acov_r5 = alpha_r5_target.sum() / (64 * 64)
            axes[4][t].imshow(np.clip(recon_corr, 0, 1))
            diff_r5 = acov_r5 - acov_r3
            color5 = 'red' if abs(diff_r5) > 0.002 else 'green'
            axes[4][t].set_title(f'd={corr_d:.3f} acov={acov_r5:.4f} ({diff_r5:+.4f})', fontsize=6, color=color5)
            axes[4][t].axis('off')

            # Row 6: Row3 alpha mask
            axes[5][t].imshow(alpha_r3_target, cmap='hot', vmin=0, vmax=1)
            axes[5][t].set_title(f'acov={acov_r3:.4f}', fontsize=6)
            axes[5][t].axis('off')

            # Row 7: Row4 alpha mask
            axes[6][t].imshow(alpha_r4_target, cmap='hot', vmin=0, vmax=1)
            axes[6][t].set_title(f'acov={acov_r4:.4f}', fontsize=6)
            axes[6][t].axis('off')

            # Row 8: Row5 (corrected) alpha mask
            axes[7][t].imshow(alpha_r5_target, cmap='hot', vmin=0, vmax=1)
            axes[7][t].set_title(f'acov={acov_r5:.4f}', fontsize=6)
            axes[7][t].axis('off')

        for row, label in enumerate(['GT', 'Normal', 'NoFG', 'BestApp', 'BestApp+CorrD',
                                      'α NoFG', 'α BestApp', 'α CorrD']):
            axes[row][0].set_ylabel(label, fontsize=8, rotation=0, labelpad=45,
                                     ha='right', va='center')

        plt.suptitle(f'Slot {s_target} (best_t={best_t}, best_d={best_depth:.3f})',
                     fontsize=13)
        plt.tight_layout()
        save_path = f'pos_depth_debug/phase3_sample9_bestapp_slot{s_target}.png'
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")


if __name__ == '__main__':
    main()
