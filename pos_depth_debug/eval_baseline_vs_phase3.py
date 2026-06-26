"""Baseline vs Phase3 ISA 对比散点图 + 重新评估单调性（depth=scale）"""
import torch, yaml, sys, warnings, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P
warnings.filterwarnings('ignore')
sys.path.insert(0, '..')
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset
from types import SimpleNamespace

device = torch.device('cuda')
app_dim = 64; H, W = 64, 64

def load_model(ckpt_path, cfg):
    model = SlotDynamicsModel(cfg).cuda()
    ckpt = torch.load(ckpt_path, map_location='cuda')
    model_sd = model.state_dict()
    ckpt_sd = ckpt['model']
    loaded = {}
    for ck_key, v in ckpt_sd.items():
        ck_clean = ck_key.replace('_orig_mod.', '')
        for mk_key in model_sd:
            if ck_clean == mk_key.replace('_orig_mod.', '') and v.shape == model_sd[mk_key].shape:
                loaded[mk_key] = v; break
    model.load_state_dict(loaded, strict=False)
    return model

def collect_data(model, ds, num_samples, max_frames=None):
    all_depth = []; all_spread = []; all_cov = []; all_depth_raw = []
    with torch.no_grad():
        gy, gx = torch.meshgrid(torch.linspace(-1,1,H,device=device), torch.linspace(-1,1,W,device=device), indexing='ij')
        for si in range(num_samples):
            sample = ds[si]
            video = sample['video'].unsqueeze(0).to(device)
            out = model(video)
            slots = out['slots']['corrected'][0]
            T = slots.shape[0]; N = slots.shape[1]
            T_use = min(T, max_frames) if max_frames else T
            for t in range(T_use):
                _, a_full, _ = model.decoder(slots[t].unsqueeze(0), return_alphas=True, return_rgb=True)
                a = a_full[0, :, 0]
                dominant_slot = a.argmax(dim=0)
                for s in range(N):
                    alpha_cov = a[s].sum().item()
                    pixel_cov = (dominant_slot == s).sum().item()
                    a_max = a[s].max().item()
                    depth = slots[t, s, app_dim+2].item()
                    all_depth_raw.append(depth)
                    if a_max > 0.3 and 50 < pixel_cov < 1200 and depth < 0.5:
                        a_s = a[s]; a_n = a_s/(alpha_cov+1e-8)
                        cx = (a_n*gx).sum().item(); cy = (a_n*gy).sum().item()
                        sp = np.sqrt((a_n*((gx-cx)**2+(gy-cy)**2)).sum().item())
                        all_depth.append(depth)
                        all_spread.append(sp)
                        all_cov.append(alpha_cov/(H*W))
    return (np.array(all_depth), np.array(all_spread), np.array(all_cov), np.array(all_depth_raw))

def compute_r2(x, y, deg=1):
    co = P.polyfit(x, y, deg)
    return 1 - np.sum((y-P.polyval(x, co))**2)/np.sum((y-y.mean())**2)

# ============ Load models ============
ds = OBJ3DDataset(data_path='../data/obj3d', subsample=2)

# Baseline
print('Loading Baseline (depth_pred_best)...')
with open('../config/pretrain_phase2.yaml') as f: cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)
cfg.pretrain = True; cfg.freeze_slot = False; cfg.continue_pretrain = False
cfg.depth_spread_weight = 0.0
model_base = load_model('../good_checkpoints/depth_pred_best.pt', cfg)
model_base.eval()

# Phase3
print('Loading Phase3...')
with open('../config/pretrain_phase3.yaml') as f: cfg_dict = yaml.safe_load(f)
cfg3 = SimpleNamespace(**cfg_dict)
cfg3.pretrain = True; cfg3.freeze_slot = False; cfg3.continue_pretrain = False
model_p3 = load_model('../experiments/phase3_gru2_full/checkpoints/best.pt', cfg3)
model_p3.eval()

# ============ Collect data ============
print('Collecting baseline data (2500 samples, 1 frame)...')
d_b, sp_b, c_b, dr_b = collect_data(model_base, ds, 2500, max_frames=1)
print(f'  Baseline: {len(d_b)} FG points, depth range [{dr_b.min():.4f}, {dr_b.max():.4f}]')

print('Collecting Phase3 data (2500 samples, 1 frame)...')
d_p, sp_p, c_p, dr_p = collect_data(model_p3, ds, 2500, max_frames=1)
print(f'  Phase3: {len(d_p)} FG points, depth range [{dr_p.min():.4f}, {dr_p.max():.4f}]')

# ============ Compute R² ============
r2_b_ds = compute_r2(d_b, sp_b)
r2_b_d2c = compute_r2(d_b**2, c_b)
r2_b_dc = compute_r2(d_b, c_b)

r2_p_ds = compute_r2(d_p, sp_p)
r2_p_d2c = compute_r2(d_p**2, c_p)
r2_p_dc = compute_r2(d_p, c_p)

print(f'\nBaseline: R²(depth,spread)={r2_b_ds:.4f}, R²(depth²,cov)={r2_b_d2c:.4f}, R²(depth,cov)={r2_b_dc:.4f}')
print(f'Phase3:   R²(depth,spread)={r2_p_ds:.4f}, R²(depth²,cov)={r2_p_d2c:.4f}, R²(depth,cov)={r2_p_dc:.4f}')

# ============ Plot ============
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

datasets = [
    ('Baseline (depth_pred_best)', d_b, sp_b, c_b, r2_b_ds, r2_b_d2c, r2_b_dc),
    ('Phase3 ISA (gru2 full, 20k)', d_p, sp_p, c_p, r2_p_ds, r2_p_d2c, r2_p_dc),
]

for row, (label, d, sp, c, r2_ds, r2_d2c, r2_dc) in enumerate(datasets):
    # depth vs spread
    ax = axes[row, 0]
    ax.scatter(d, sp, s=2, alpha=0.2, c='steelblue')
    co = P.polyfit(d, sp, 1)
    x_line = np.linspace(d.min(), d.max(), 100)
    ax.plot(x_line, P.polyval(x_line, co), 'r-', linewidth=2, label=f'R²={r2_ds:.4f}')
    ax.set_xlabel('Depth (≈Scale)'); ax.set_ylabel('Alpha Spread')
    ax.set_title(f'{label}\nDepth vs Spread'); ax.legend(fontsize=10)
    
    # depth² vs coverage
    ax = axes[row, 1]
    d2 = d**2
    ax.scatter(d2, c, s=2, alpha=0.2, c='darkorange')
    co = P.polyfit(d2, c, 1)
    x_line = np.linspace(d2.min(), d2.max(), 100)
    ax.plot(x_line, P.polyval(x_line, co), 'r-', linewidth=2, label=f'R²={r2_d2c:.4f}')
    ax.set_xlabel('Depth² (≈Scale²)'); ax.set_ylabel('Alpha Coverage (norm)')
    ax.set_title(f'{label}\nDepth² vs Coverage'); ax.legend(fontsize=10)
    
    # depth vs coverage
    ax = axes[row, 2]
    ax.scatter(d, c, s=2, alpha=0.2, c='forestgreen')
    co = P.polyfit(d, c, 1)
    x_line = np.linspace(d.min(), d.max(), 100)
    ax.plot(x_line, P.polyval(x_line, co), 'r-', linewidth=2, label=f'R²={r2_dc:.4f}')
    ax.set_xlabel('Depth (≈Scale)'); ax.set_ylabel('Alpha Coverage (norm)')
    ax.set_title(f'{label}\nDepth vs Coverage'); ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('phase3_vs_baseline_scatter.png', dpi=150)
print('\nSaved: phase3_vs_baseline_scatter.png')

# ============ Monotonicity re-evaluation (depth=scale) ============
print('\n' + '='*60)
print('Monotonicity re-evaluation (depth=scale, same direction is correct)')
print('='*60)

# depth=scale: depth↑ should mean object bigger (cov↑), depth↓ should mean object smaller (cov↓)
# So "consistent" = same direction (both up or both down)
# "inconsistent" = opposite direction

# Reuse Phase3 model data
up_up = 0; down_down = 0; up_down = 0; down_up = 0
n_consistent = 0; n_inconsistent = 0

with torch.no_grad():
    for si in range(100):
        sample = ds[si]
        video = sample['video'].unsqueeze(0).to(device)
        out = model_p3(video)
        slots = out['slots']['corrected'][0]
        T = slots.shape[0]; N = slots.shape[1]

        slot_depths = np.zeros((T, N))
        slot_covs = np.zeros((T, N))
        for t in range(T):
            _, a_full, _ = model_p3.decoder(slots[t].unsqueeze(0), return_alphas=True, return_rgb=True)
            a = a_full[0, :, 0]
            dominant = a.argmax(dim=0)
            for s in range(N):
                slot_depths[t, s] = slots[t, s, app_dim+2].item()
                slot_covs[t, s] = (dominant == s).sum().item()

        for s in range(N):
            if slot_depths[:, s].max() > 0.5:
                continue
            avg_cov = slot_covs[:, s].mean()
            if avg_cov < 50:
                continue
            for t in range(1, T):
                dd = slot_depths[t, s] - slot_depths[t-1, s]
                dc = slot_covs[t, s] - slot_covs[t-1, s]
                if abs(dd) < 0.002 or abs(dc) < 3:
                    continue
                if dd > 0 and dc > 0:
                    up_up += 1
                elif dd < 0 and dc < 0:
                    down_down += 1
                elif dd > 0 and dc < 0:
                    up_down += 1
                elif dd < 0 and dc > 0:
                    down_up += 1

total = up_up + down_down + up_down + down_up
same_dir = up_up + down_down
opp_dir = up_down + down_up
print(f'depth↑cov↑: {up_up}  depth↓cov↓: {down_down}  → same direction (consistent for scale): {same_dir}')
print(f'depth↑cov↓: {up_down}  depth↓cov↑: {down_up}  → opposite direction (inconsistent for scale): {opp_dir}')
print(f'Consistent (same direction): {same_dir}/{total} ({same_dir/total*100:.1f}%)')
print(f'Inconsistent (opposite direction): {opp_dir}/{total} ({opp_dir/total*100:.1f}%)')
