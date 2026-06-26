"""Phase2: depth² vs alpha_coverage 散点图对比 (7k, 8k, 9k)"""
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

with open('../config/pretrain_phase2.yaml') as f: cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)
cfg.pretrain = True; cfg.freeze_slot = False; cfg.continue_pretrain = False

ds = OBJ3DDataset(data_path='../data/obj3d', subsample=2)
H, W = 64, 64

steps = [7000, 8000, 9000]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

num_samples = 50
for ax, step in zip(axes, steps):
    ckpt_path = f'../experiments/phase2_depth_spread/checkpoints/step_{step}.pt'
    print(f'Loading step {step}...')
    model = load_model(ckpt_path, cfg)
    model.eval()

    all_depth2 = []
    all_cov = []

    with torch.no_grad():
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
                        all_depth2.append(depth ** 2)
                        all_cov.append(alpha_cov / (H * W))

    x = np.array(all_depth2); y = np.array(all_cov)
    ax.scatter(x, y, s=4, alpha=0.3, c='steelblue')

    if len(x) > 10:
        from numpy.polynomial import polynomial as P
        coeffs = P.polyfit(x, y, 1)
        x_sort = np.sort(x)
        ax.plot(x_sort, P.polyval(x_sort, coeffs), 'r-', lw=1.5)
        ss_res = np.sum((y - P.polyval(x, coeffs))**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot
        ax.set_title(f'step {step} ({len(x)} pts)\nR²={r2:.4f}')
    else:
        ax.set_title(f'step {step} ({len(x)} pts)')

    ax.set_xlabel('depth²')
    ax.set_ylabel('alpha coverage / 4096')
    ax.grid(alpha=0.3)

plt.suptitle('Phase2: depth² vs Alpha Coverage (FG filtered)', fontsize=13)
plt.tight_layout()
plt.savefig('phase2_depth2_vs_cov_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved')
