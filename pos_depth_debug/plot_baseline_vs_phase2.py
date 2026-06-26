"""对比 baseline (depth_pred_best) vs phase2 best 的 depth-spread 和 depth²-cov 散点图"""
import torch, yaml, sys, warnings, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import polynomial as P
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

with open('../config/pretrain_phase2.yaml') as f: cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)
cfg.pretrain = True; cfg.freeze_slot = False; cfg.continue_pretrain = False

ds = OBJ3DDataset(data_path='../data/obj3d', subsample=2)
H, W = 64, 64

ckpts = {
    'Baseline (depth_pred_best)': '../good_checkpoints/depth_pred_best.pt',
    'Phase2 best': '../experiments/phase2_depth_spread/checkpoints/best.pt',
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

num_samples = 1000
for row, (name, ckpt_path) in enumerate(ckpts.items()):
    print(f'Loading {name}...')
    model = load_model(ckpt_path, cfg)
    model.eval()

    all_depth = []; all_depth2 = []; all_spread = []; all_cov = []

    with torch.no_grad():
        gy, gx = torch.meshgrid(torch.linspace(-1,1,H,device=device), torch.linspace(-1,1,W,device=device), indexing='ij')
        for si in range(min(num_samples, len(ds))):
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

                    if a_max > 0.3 and 50 < pixel_cov < 1200 and depth < 0.5:
                        a_s = a[s]; a_n = a_s/(alpha_cov+1e-8)
                        cx = (a_n*gx).sum().item(); cy = (a_n*gy).sum().item()
                        sp = np.sqrt((a_n*((gx-cx)**2+(gy-cy)**2)).sum().item())
                        all_depth.append(depth)
                        all_depth2.append(depth**2)
                        all_spread.append(sp)
                        all_cov.append(alpha_cov / (H*W))

    d = np.array(all_depth); d2 = np.array(all_depth2)
    sp = np.array(all_spread); c = np.array(all_cov)
    print(f'  {len(d)} FG points, depth range [{d.min():.4f}, {d.max():.4f}]')

    # 左列: depth vs alpha_spread
    ax = axes[row, 0]
    ax.scatter(d, sp, s=6, alpha=0.3, c='steelblue')
    co1 = P.polyfit(d, sp, 1)
    d_sort = np.sort(d)
    ax.plot(d_sort, P.polyval(d_sort, co1), 'r-', lw=1.5)
    r2_sp = 1 - np.sum((sp-P.polyval(d,co1))**2)/np.sum((sp-sp.mean())**2)
    ax.set_xlabel('ISA Depth'); ax.set_ylabel('Alpha Spread')
    ax.set_title(f'R²(depth, spread) = {r2_sp:.4f}')
    ax.grid(alpha=0.3)
    if row == 0:
        ax.set_ylabel(f'{name}\nAlpha Spread', fontsize=10)
    else:
        ax.set_ylabel(f'{name}\nAlpha Spread', fontsize=10)

    # 右列: depth² vs alpha_coverage/4096
    ax = axes[row, 1]
    ax.scatter(d2, c, s=6, alpha=0.3, c='coral')
    co2 = P.polyfit(d2, c, 1)
    d2_sort = np.sort(d2)
    ax.plot(d2_sort, P.polyval(d2_sort, co2), 'r-', lw=1.5)
    r2_cov = 1 - np.sum((c-P.polyval(d2,co2))**2)/np.sum((c-c.mean())**2)
    ax.set_xlabel('ISA Depth²'); ax.set_ylabel('Alpha Coverage / 4096')
    ax.set_title(f'R²(depth², cov) = {r2_cov:.4f}')
    ax.grid(alpha=0.3)

    print(f'  R²(depth, spread) = {r2_sp:.4f}, R²(depth², cov) = {r2_cov:.4f}')

plt.suptitle('Baseline vs Phase2: Depth-Spread & Depth²-Coverage', fontsize=13)
plt.tight_layout()
plt.savefig('baseline_vs_phase2_r2_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved')
