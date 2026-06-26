#!/usr/bin/env python3
"""
depth² vs pixel coverage 散点图（Prior Phase2, 10000样本）
pixel coverage = argmax_slot == s 的像素计数
"""
import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
import torch, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace
import yaml

sys.path.insert(0, '.')
from models.dynamics import SlotDynamicsModel
from train import Trainer, create_optimizer
from train.trainer import WandBLogger
from data import get_dataset

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

with open('config/pretrain_phase2.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg_dict['continue_pretrain'] = True
cfg_dict['workdir'] = '/tmp/pixcov_eval'
cfg = SimpleNamespace(**cfg_dict)

model = SlotDynamicsModel(cfg)
opt, sch = create_optimizer((p for p in model.parameters() if p.requires_grad), cfg)
wb = WandBLogger(enabled=False)
trainer = Trainer(model, opt, sch, cfg, wandb_logger=wb)
trainer.load_checkpoint('experiments/phase2_depth_spread/checkpoints/best.pt')
model.eval().cuda()

a_val = model.depth_spread_a.item()
b_val = model.depth_spread_b.item()
c_val = model.depth_spread_c.item()
d_val = model.depth_spread_d.item()

num_frames = getattr(cfg, 'num_frames', None) or 1
ds = get_dataset(cfg.dataset, data_path=cfg.data_root, num_frames=num_frames,
                 stride=getattr(cfg, 'slide_stride', 1), subsample=getattr(cfg, 'subsample', 1))
loader = ds.get_dataloader(batch_size=cfg.batch_size, shuffle=True, num_workers=0)

app_dim = model.appearance_dim
all_d, all_pixcov, all_alpha_cov, all_amax = [], [], [], []

n_samples = 10000
samples_done = 0
with torch.no_grad():
    for i, batch in enumerate(loader):
        if samples_done >= n_samples: break
        frames = batch["video"].cuda()
        out = model(frames)
        burnin_T = out['slots']['corrected'].shape[1]
        for t in range(burnin_T):
            slots_t = out['slots']['corrected'][:, t]
            alpha_t = out['alpha'][:, :, t]
            if alpha_t.dim() == 5: alpha_2d = alpha_t.squeeze(2)
            else: alpha_2d = alpha_t
            B, N, H, W = alpha_2d.shape
            depth = slots_t[:, :, app_dim + 2]
            a_max = alpha_2d.amax(dim=[-2, -1])
            alpha_cov = alpha_2d.sum(dim=[-2, -1])
            # pixel coverage: argmax over slots
            dominant = alpha_2d.argmax(dim=1)  # (B, H, W)
            # spread for FG filter
            gy, gx = torch.meshgrid(torch.linspace(-1,1,H,device='cuda'), torch.linspace(-1,1,W,device='cuda'), indexing='ij')
            gx_b = gx.unsqueeze(0).unsqueeze(0).expand(B,N,H,W)
            gy_b = gy.unsqueeze(0).unsqueeze(0).expand(B,N,H,W)
            a_sum = alpha_2d.sum(dim=[-2,-1], keepdim=True) + 1e-8
            a_norm = alpha_2d / a_sum
            cx = (a_norm * gx_b).sum(dim=[-2,-1])
            cy = (a_norm * gy_b).sum(dim=[-2,-1])
            spread = torch.sqrt((a_norm * ((gx_b-cx.unsqueeze(-1).unsqueeze(-1))**2 + (gy_b-cy.unsqueeze(-1).unsqueeze(-1))**2)).sum(dim=[-2,-1]))

            fg = (alpha_cov > 20) & (alpha_cov < 1500) & (spread > 0.01) & (a_max >= 0.9) & (depth <= 0.25)

            for b_idx in range(B):
                for s_idx in range(N):
                    if fg[b_idx, s_idx]:
                        pixcov = (dominant[b_idx] == s_idx).sum().item()
                        all_d.append(depth[b_idx, s_idx].item())
                        all_pixcov.append(pixcov)
                        all_alpha_cov.append(alpha_cov[b_idx, s_idx].item() / (H * W))
                        all_amax.append(a_max[b_idx, s_idx].item())
        samples_done += frames.shape[0]
        if i % 30 == 0:
            print(f"  {i} batches, {samples_done} samples, {len(all_d)} FG pts", flush=True)

dm = np.array(all_d)
pm = np.array(all_pixcov)
cm = np.array(all_alpha_cov)
d2m = dm**2
mask = dm > 0.04
dm, pm, cm, d2m = dm[mask], pm[mask], cm[mask], d2m[mask]

print(f"\nTotal FG points: {len(dm)}")

# Pixel coverage normalization (optional, plot raw count)
# Huber fit
coef_d2p = huber_fit(d2m, pm.astype(float))
r2_d2p = huber_r2(pm.astype(float), coef_d2p[0] * d2m + coef_d2p[1])

# Also alpha cov for comparison
coef_d2c = huber_fit(d2m, cm)
r2_d2c = huber_r2(cm, coef_d2c[0] * d2m + coef_d2c[1])

# Prior params R²
r2_d2p_prior = huber_r2(pm.astype(float), b_val * d2m + d_val)
r2_d2c_prior = huber_r2(cm, b_val * d2m + d_val)

print(f"Huber fit: depth² vs pixel_cov: k={coef_d2p[0]:.2f} b={coef_d2p[1]:.2f}  R²={r2_d2p:.4f}")
print(f"Huber fit: depth² vs alpha_cov: k={coef_d2c[0]:.2f} b={coef_d2c[1]:.2f}  R²={r2_d2c:.4f}")
print(f"Prior params: depth² vs pixel_cov  R²={r2_d2p_prior:.4f}")
print(f"Prior params: depth² vs alpha_cov  R²={r2_d2c_prior:.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
ax.scatter(d2m, pm, s=2, alpha=0.15, c='steelblue')
x_line = np.linspace(0, d2m.max() * 1.05, 100)
ax.plot(x_line, coef_d2p[0] * x_line + coef_d2p[1], 'b-', linewidth=2,
        label=f'Huber y={coef_d2p[0]:.1f}x+{coef_d2p[1]:.1f}  R²={r2_d2p:.4f}')
ax.plot(x_line, b_val * x_line + d_val, 'r--', linewidth=2,
        label=f'Prior y={b_val:.3f}x+{d_val:.4f}  R²={r2_d2p_prior:.4f}')
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
ax.set_xlabel('Depth²'); ax.set_ylabel('Pixel Coverage')
ax.set_title(f'Depth² vs Pixel Coverage ({len(dm)} pts)'); ax.legend()

ax = axes[1]
ax.scatter(d2m, cm, s=2, alpha=0.15, c='darkorange')
ax.plot(x_line, coef_d2c[0] * x_line + coef_d2c[1], 'b-', linewidth=2,
        label=f'Huber y={coef_d2c[0]:.3f}x+{coef_d2c[1]:.3f}  R²={r2_d2c:.4f}')
ax.plot(x_line, b_val * x_line + d_val, 'r--', linewidth=2,
        label=f'Prior y={b_val:.3f}x+{d_val:.4f}  R²={r2_d2c_prior:.4f}')
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
ax.set_xlabel('Depth²'); ax.set_ylabel('Alpha Coverage (norm)')
ax.set_title(f'Depth² vs Alpha Coverage ({len(dm)} pts)'); ax.legend()

plt.suptitle(f'Prior Phase2: Depth² vs Coverage (10000 samples, FG: a_max>=0.9, depth<=0.25)', fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('pos_depth_debug/depth2_vs_coverage.png', dpi=150, bbox_inches='tight')
print(f"\nSaved to pos_depth_debug/depth2_vs_coverage.png")
