"""Attn Spread^2 vs Alpha Coverage 散点图 (FG, dominant_ratio>0.4)"""
import torch, yaml, sys, warnings, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '..')
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset
from types import SimpleNamespace

device = torch.device('cuda')
app_dim = 64

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

with open('../config/pretrain_phase3.yaml') as f: cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)
cfg.pretrain = True; cfg.freeze_slot = True; cfg.continue_pretrain = False
cfg.burnin_frames = 16
model = load_model('../experiments/phase3_gru2_full/checkpoints/best.pt', cfg)
model.eval()

ds = OBJ3DDataset(data_path='../data/obj3d', subsample=2)
H, W = 64, 64
side = 16
gy, gx = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W), indexing='ij')
gy, gx = gy.to(device), gx.to(device)
gy16, gx16 = torch.meshgrid(torch.linspace(-1,1,side), torch.linspace(-1,1,side), indexing='ij')
gy16, gx16 = gy16.to(device), gx16.to(device)

all_attn_sp2 = []
all_alpha_cov = []
all_depths = []
all_alpha_sp2 = []

num_samples = 50
with torch.no_grad():
    for si in range(min(num_samples, len(ds))):
        sample = ds[si]
        video = sample['video'].unsqueeze(0).to(device)
        out = model(video)
        attn = out['attn']
        slots = out['slots']['corrected'][0]
        T = slots.shape[0]; N = slots.shape[1]

        for t in range(T):
            _, a_full, _ = model.decoder(slots[t].unsqueeze(0), return_alphas=True, return_rgb=True)
            a = a_full[0, :, 0]
            dominant_slot = a.argmax(dim=0)
            attn_t = attn[0, t]

            for s in range(N):
                alpha_cov = a[s].sum().item()
                pixel_cov = (dominant_slot == s).sum().item()
                a_max = a[s].max().item()
                depth = slots[t, s, app_dim+2].item()

                # 严格FG filter: a_max>0.5, pixel_cov 50~1200, depth<0.5, dominant_ratio>0.5
                if a_max > 0.5 and 50 < pixel_cov < 1200 and depth < 0.5:
                    dominant_mask = (dominant_slot == s)
                    alpha_on_dominant = a[s][dominant_mask].sum().item()
                    dominant_ratio = alpha_on_dominant / (alpha_cov + 1e-8)
                    if dominant_ratio < 0.5:
                        continue

                    # attn spread
                    at = attn_t[s].reshape(side, side)
                    at_sum = at.sum().item(); at_n = at/(at_sum+1e-8)
                    cx16 = (at_n*gx16).sum().item(); cy16 = (at_n*gy16).sum().item()
                    attn_sp = torch.sqrt((at_n*((gx16-cx16)**2+(gy16-cy16)**2)).sum()).item()

                    # alpha spread
                    a_s = a[s]; a_n = a_s/(alpha_cov+1e-8)
                    cx = (a_n*gx).sum().item(); cy = (a_n*gy).sum().item()
                    alpha_sp = torch.sqrt((a_n*((gx-cx)**2+(gy-cy)**2)).sum()).item()

                    all_attn_sp2.append(attn_sp**2)
                    all_alpha_cov.append(alpha_cov)
                    all_depths.append(depth)
                    all_alpha_sp2.append(alpha_sp**2)

        if (si+1) % 10 == 0:
            print(f"Processed {si+1} samples, {len(all_alpha_cov)} pts")

print(f"Total: {len(all_alpha_cov)} pts")

from numpy.polynomial import polynomial as P

x = np.array(all_attn_sp2); y = np.array(all_alpha_cov)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 左图：attn_sp^2 vs alpha_cov
coeffs1 = P.polyfit(x, y, 1)
ss_res1 = np.sum((y - P.polyval(x, coeffs1))**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2_lin = 1 - ss_res1/ss_tot
coeffs2 = P.polyfit(x, y, 2)
ss_res2 = np.sum((y - P.polyval(x, coeffs2))**2)
r2_quad = 1 - ss_res2/ss_tot
x_sort = np.sort(x)

ax1.scatter(x, y, s=6, alpha=0.3, c='steelblue')
ax1.plot(x_sort, P.polyval(x_sort, coeffs1), 'r-', lw=1.5, label=f'linear R²={r2_lin:.4f}')
ax1.plot(x_sort, P.polyval(x_sort, coeffs2), 'g--', lw=1.5, label=f'quad R²={r2_quad:.4f}')
ax1.set_xlabel('Attn Spread²'); ax1.set_ylabel('Alpha Coverage')
ax1.set_title(f'Attn Spread² vs Alpha Coverage ({len(x)} pts)')
ax1.legend(); ax1.grid(alpha=0.3)

# 右图：alpha_sp^2 vs alpha_cov
x2 = np.array(all_alpha_sp2)
coeffs3 = P.polyfit(x2, y, 1)
ss_res3 = np.sum((y - P.polyval(x2, coeffs3))**2)
r2_alpha = 1 - ss_res3/ss_tot
coeffs4 = P.polyfit(x2, y, 2)
ss_res4 = np.sum((y - P.polyval(x2, coeffs4))**2)
r2_alpha_q = 1 - ss_res4/ss_tot
x2_sort = np.sort(x2)

ax2.scatter(x2, y, s=6, alpha=0.3, c='coral')
ax2.plot(x2_sort, P.polyval(x2_sort, coeffs3), 'r-', lw=1.5, label=f'linear R²={r2_alpha:.4f}')
ax2.plot(x2_sort, P.polyval(x2_sort, coeffs4), 'g--', lw=1.5, label=f'quad R²={r2_alpha_q:.4f}')
ax2.set_xlabel('Alpha Spread²'); ax2.set_ylabel('Alpha Coverage')
ax2.set_title(f'Alpha Spread² vs Alpha Coverage ({len(x)} pts)')
ax2.legend(); ax2.grid(alpha=0.3)

plt.suptitle('Spread² vs Alpha Coverage (FG, dominant_ratio>0.5)', fontsize=13)
plt.tight_layout()
plt.savefig('spread2_vs_alpha_cov.png', dpi=150, bbox_inches='tight')
plt.close()

# depth vs alpha_cov
x3 = np.array(all_depths)
coeffs5 = P.polyfit(x3, y, 1)
ss_res5 = np.sum((y - P.polyval(x3, coeffs5))**2)
r2_depth = 1 - ss_res5/ss_tot

print(f'\nAttn Spread² vs Alpha Coverage: R² lin={r2_lin:.4f} quad={r2_quad:.4f}')
print(f'Alpha Spread² vs Alpha Coverage: R² lin={r2_alpha:.4f} quad={r2_alpha_q:.4f}')
print(f'Depth vs Alpha Coverage: R² lin={r2_depth:.4f}')

# 诊断：打印attn_sp范围
attn_sps = np.sqrt(x)
print(f'\nAttn spread range: [{attn_sps.min():.3f}, {attn_sps.max():.3f}]')
print(f'Alpha cov range: [{y.min():.1f}, {y.max():.1f}]')
print(f'Attn_sp > 0.5 count: {(attn_sps > 0.5).sum()} / {len(attn_sps)}')
print(f'Attn_sp > 0.6 count: {(attn_sps > 0.6).sum()} / {len(attn_sps)}')

# 加滤attn_sp < 0.5
mask_sp = attn_sps < 0.5
x_f = x[mask_sp]; y_f = y[mask_sp]
if len(x_f) > 10:
    c_f = P.polyfit(x_f, y_f, 1)
    ss_f = np.sum((y_f - P.polyval(x_f, c_f))**2)
    st_f = np.sum((y_f - np.mean(y_f))**2)
    r2_f = 1 - ss_f/st_f
    print(f'\nFiltered attn_sp < 0.5: {len(x_f)} pts, R² = {r2_f:.4f}')
