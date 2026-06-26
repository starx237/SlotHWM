"""Alpha Coverage vs Pixel Coverage 散点图 (FG only)"""
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

all_alpha_cov = []
all_pixel_cov = []
all_depths = []
all_spreads = []
all_dominant_ratio = []  # alpha在自身dominant像素上的占比

num_samples = 50
with torch.no_grad():
    for si in range(min(num_samples, len(ds))):
        sample = ds[si]
        video = sample['video'].unsqueeze(0).to(device)
        out = model(video)
        slots = out['slots']['corrected'][0]
        T = slots.shape[0]; N = slots.shape[1]

        for t in range(T):
            _, a_full, _ = model.decoder(slots[t].unsqueeze(0), return_alphas=True, return_rgb=True)
            a = a_full[0, :, 0]  # (N, H, W)
            dominant_slot = a.argmax(dim=0)  # (H, W)

            for s in range(N):
                alpha_cov = a[s].sum().item()
                pixel_cov = (dominant_slot == s).sum().item()
                a_max = a[s].max().item()
                depth = slots[t, s, app_dim+2].item()

                # FG filter: a_max > 0.3, pixel_cov in reasonable range, depth < 0.5
                if a_max > 0.3 and pixel_cov > 20 and pixel_cov < 1500 and depth < 0.5:
                    # dominant ratio: 该slot在自身dominant像素上的alpha总和 / 总alpha
                    dominant_mask = (dominant_slot == s)
                    alpha_on_dominant = a[s][dominant_mask].sum().item()
                    dominant_ratio = alpha_on_dominant / (alpha_cov + 1e-8)

                    gy, gx = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W), indexing='ij')
                    gy, gx = gy.to(device), gx.to(device)
                    a_s = a[s]; a_n = a_s/(alpha_cov+1e-8)
                    cx = (a_n*gx).sum().item(); cy = (a_n*gy).sum().item()
                    sp = torch.sqrt((a_n*((gx-cx)**2+(gy-cy)**2)).sum()).item()

                    all_alpha_cov.append(alpha_cov)
                    all_pixel_cov.append(float(pixel_cov))
                    all_depths.append(depth)
                    all_spreads.append(sp)
                    all_dominant_ratio.append(dominant_ratio)

        if (si+1) % 10 == 0:
            print(f"Processed {si+1} samples, {len(all_alpha_cov)} FG points")

print(f"Total FG points: {len(all_alpha_cov)}")

from numpy.polynomial import polynomial as P

x = np.array(all_pixel_cov); y = np.array(all_alpha_cov)

# 线性拟合
coeffs1 = P.polyfit(x, y, 1)
ss_res1 = np.sum((y - P.polyval(x, coeffs1))**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2_lin = 1 - ss_res1/ss_tot

# 二次拟合
coeffs2 = P.polyfit(x, y, 2)
ss_res2 = np.sum((y - P.polyval(x, coeffs2))**2)
r2_quad = 1 - ss_res2/ss_tot

x_sort = np.sort(x)

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(x, y, s=6, alpha=0.3, c='steelblue')
ax.plot(x_sort, P.polyval(x_sort, coeffs1), 'r-', lw=1.5, label=f'linear R²={r2_lin:.4f}')
ax.plot(x_sort, P.polyval(x_sort, coeffs2), 'g--', lw=1.5, label=f'quad R²={r2_quad:.4f}')
ax.plot([0, max(x)*1.1], [0, max(x)*1.1], 'k:', alpha=0.3, label='y=x')
ax.set_xlabel('Pixel Coverage (dominant pixel count, FG only)')
ax.set_ylabel('Alpha Mask Coverage (sum of alpha, FG only)')
ax.set_title(f'Alpha vs Pixel Coverage (FG filtered: a_max>0.3, depth<0.5)\n{len(x)} pts from {num_samples} samples')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('alpha_vs_pixel_cov_fg_only.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'R² linear={r2_lin:.4f}, quad={r2_quad:.4f}')
print(f'Linear fit: y = {coeffs1[1]:.4f}*x + {coeffs1[0]:.2f}')

# dominant ratio分布
dr = np.array(all_dominant_ratio)
print(f'\nDominant ratio stats: mean={dr.mean():.3f} median={np.median(dr):.3f}')

# 不同dominant ratio阈值下的R²
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    mask = dr > thresh
    x2 = x[mask]; y2 = y[mask]
    if len(x2) < 10:
        continue
    c1 = P.polyfit(x2, y2, 1)
    ss1 = np.sum((y2 - P.polyval(x2, c1))**2)
    st = np.sum((y2 - np.mean(y2))**2)
    r2 = 1 - ss1/st
    print(f'  dr > {thresh}: {len(x2)} pts, R² = {r2:.4f}')

# 用dominant ratio > 0.5（覆盖大部分FG slot）
mask = dr > 0.5
x2 = x[mask]; y2 = y[mask]
c1b = P.polyfit(x2, y2, 1)
ss1b = np.sum((y2 - P.polyval(x2, c1b))**2)
st2 = np.sum((y2 - np.mean(y2))**2)
r2b = 1 - ss1b/st2
print(f'\nBest filter (dr>0.5): {len(x2)} pts, R² = {r2b:.4f}')

# 画两张图对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 左图：全部FG
ax1.scatter(x, y, s=6, alpha=0.3, c='steelblue')
x_sort = np.sort(x)
ax1.plot(x_sort, P.polyval(x_sort, coeffs1), 'r-', lw=1.5, label=f'linear R²={r2_lin:.4f}')
ax1.plot([0, max(x)*1.1], [0, max(x)*1.1], 'k:', alpha=0.3, label='y=x')
ax1.set_xlabel('Pixel Coverage'); ax1.set_ylabel('Alpha Coverage')
ax1.set_title(f'All FG ({len(x)} pts)'); ax1.legend(); ax1.grid(alpha=0.3)

# 右图：dominant_ratio > 0.5
ax2.scatter(x2, y2, s=6, alpha=0.3, c='steelblue')
x2_sort = np.sort(x2)
ax2.plot(x2_sort, P.polyval(x2_sort, c1b), 'r-', lw=1.5, label=f'linear R²={r2b:.4f}')
ax2.plot([0, max(x2)*1.1], [0, max(x2)*1.1], 'k:', alpha=0.3, label='y=x')
ax2.set_xlabel('Pixel Coverage'); ax2.set_ylabel('Alpha Coverage')
ax2.set_title(f'FG + dominant_ratio>0.5 ({len(x2)} pts)'); ax2.legend(); ax2.grid(alpha=0.3)

plt.suptitle('Alpha vs Pixel Coverage', fontsize=13)
plt.tight_layout()
plt.savefig('alpha_vs_pixel_cov_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved comparison')
