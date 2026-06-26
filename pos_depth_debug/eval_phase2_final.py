"""Phase2 完成后 R² 评估 + 散点图 (1500样本)"""
import torch, yaml, sys, warnings, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset
from types import SimpleNamespace

device = torch.device('cuda')
app_dim = 64
H, W = 64, 64

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

with open('config/pretrain_phase2.yaml') as f: cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)
cfg.pretrain = True; cfg.freeze_slot = False; cfg.continue_pretrain = False

ds = OBJ3DDataset(data_path='data/obj3d', subsample=2)
model = load_model('experiments/phase2_depth_spread/checkpoints/best.pt', cfg)
model.eval()

all_depth = []; all_spread = []; all_cov = []
all_depth_raw = []; all_pixel_cov = []

with torch.no_grad():
    gy, gx = torch.meshgrid(torch.linspace(-1,1,H,device=device), torch.linspace(-1,1,W,device=device), indexing='ij')
    for si in range(1500):
        sample = ds[si]
        video = sample['video'].unsqueeze(0).to(device)
        out = model(video)
        slots = out['slots']['corrected'][0]
        T = slots.shape[0]; N = slots.shape[1]
        for t in range(T):
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
                    all_pixel_cov.append(pixel_cov)

d = np.array(all_depth); sp = np.array(all_spread); c = np.array(all_cov)
d_all = np.array(all_depth_raw)
pc = np.array(all_pixel_cov)

print(f'FG points: {len(d)}, depth range (all): [{d_all.min():.4f}, {d_all.max():.4f}]')
print(f'FG depth range (filtered): [{d.min():.4f}, {d.max():.4f}]')
print(f'Points with depth>0.5 filtered out: {(d_all > 0.5).sum()} / {len(d_all)}')

co1 = P.polyfit(d, sp, 1)
r2_sp = 1 - np.sum((sp-P.polyval(d,co1))**2)/np.sum((sp-sp.mean())**2)
d2 = d**2
co2 = P.polyfit(d2, c, 1)
r2_cov = 1 - np.sum((c-P.polyval(d2,co2))**2)/np.sum((c-c.mean())**2)
print(f'R²(depth, spread) = {r2_sp:.4f}')
print(f'R²(depth², cov) = {r2_cov:.4f}')

co3 = P.polyfit(d, c, 1)
r2_d_cov = 1 - np.sum((c-P.polyval(d,co3))**2)/np.sum((c-c.mean())**2)
print(f'R²(depth, cov) = {r2_d_cov:.4f}')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.scatter(d, sp, s=2, alpha=0.3)
x_line = np.linspace(d.min(), d.max(), 100)
ax.plot(x_line, P.polyval(x_line, co1), 'r-', linewidth=2, label=f'R²={r2_sp:.4f}')
ax.set_xlabel('Depth'); ax.set_ylabel('Alpha Spread')
ax.set_title('Depth vs Alpha Spread'); ax.legend()

ax = axes[1]
ax.scatter(d2, c, s=2, alpha=0.3)
x_line = np.linspace(d2.min(), d2.max(), 100)
ax.plot(x_line, P.polyval(x_line, co2), 'r-', linewidth=2, label=f'R²={r2_cov:.4f}')
ax.set_xlabel('Depth²'); ax.set_ylabel('Alpha Coverage (norm)')
ax.set_title('Depth² vs Alpha Coverage'); ax.legend()

ax = axes[2]
ax.scatter(d, c, s=2, alpha=0.3)
x_line = np.linspace(d.min(), d.max(), 100)
ax.plot(x_line, P.polyval(x_line, co3), 'r-', linewidth=2, label=f'R²={r2_d_cov:.4f}')
ax.set_xlabel('Depth'); ax.set_ylabel('Alpha Coverage (norm)')
ax.set_title('Depth vs Alpha Coverage'); ax.legend()

plt.tight_layout()
plt.savefig('pos_depth_debug/phase2_final_scatter_30k.png', dpi=150)
print('Saved: pos_depth_debug/phase2_final_scatter_30k.png')

np.savez('pos_depth_debug/phase2_final_30k_data.npz',
         depth=d, spread=sp, cov=c, depth_raw=d_all, pixel_cov=pc)
print('Saved: pos_depth_debug/phase2_final_30k_data.npz')
