"""Sample9 全部16帧 depth/spread/alpha_coverage 折线图"""
import torch, yaml, sys, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
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

with open('config/pretrain_phase3.yaml') as f: cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)
cfg.pretrain = True; cfg.freeze_slot = False; cfg.continue_pretrain = False
cfg.burnin_frames = 16; cfg.rollout_frames = 0
model = load_model('good_checkpoints/coveragedepth_pred_best_gru2.pt', cfg)
model.eval()

sample9 = torch.load('data/sample9/sample9.pt')['video']  # (16, 3, 64, 64)
N = 6; T = 16

depth_ts = np.zeros((T, N))
spread_ts = np.zeros((T, N))
cov_ts = np.zeros((T, N))
pixel_cov_ts = np.zeros((T, N))
is_fg = np.zeros((T, N), dtype=bool)

with torch.no_grad():
    gy, gx = torch.meshgrid(torch.linspace(-1,1,H,device=device), torch.linspace(-1,1,W,device=device), indexing='ij')
    video = sample9.unsqueeze(0).to(device)  # (1, 16, 3, 64, 64)
    out = model(video)
    slots = out['slots']['corrected'][0]  # (T, N, 67)
    for t in range(T):
        _, a_full, _ = model.decoder(slots[t].unsqueeze(0), return_alphas=True, return_rgb=True)
        a = a_full[0, :, 0]
        dominant = a.argmax(dim=0)
        for s in range(N):
            depth_ts[t, s] = slots[t, s, app_dim+2].item()
            alpha_cov = a[s].sum().item()
            pix_cov = (dominant == s).sum().item()
            cov_ts[t, s] = alpha_cov / (H*W)
            pixel_cov_ts[t, s] = pix_cov
            a_max = a[s].max().item()
            if a_max > 0.3 and pix_cov > 50 and depth_ts[t, s] < 0.5:
                is_fg[t, s] = True
                a_n = a[s] / (alpha_cov + 1e-8)
                cx = (a_n * gx).sum().item(); cy = (a_n * gy).sum().item()
                spread_ts[t, s] = np.sqrt((a_n * ((gx-cx)**2 + (gy-cy)**2)).sum().item())

ts = np.arange(T)
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

fg_slots = [s for s in range(N) if is_fg[:, s].any()]
n_fg = len(fg_slots)

fig, axes = plt.subplots(n_fg, 3, figsize=(22, 4*n_fg))
if n_fg == 1:
    axes = axes.reshape(1, -1)

for i, s in enumerate(fg_slots):
    fg_mask = is_fg[:, s]
    
    ax = axes[i, 0]
    ax.plot(ts, depth_ts[:, s], '-o', color=colors[s], markersize=4, linewidth=1.5)
    ax.set_ylabel(f'Slot {s}', fontsize=11, fontweight='bold')
    if i == 0: ax.set_title('Depth (≈Scale)')
    if i == n_fg-1: ax.set_xlabel('Frame')
    ax.set_xticks(ts); ax.grid(True, alpha=0.3)
    
    ax = axes[i, 1]
    valid = fg_mask & (spread_ts[:, s] > 0)
    ax.plot(ts, spread_ts[:, s], '-s', color=colors[s], markersize=4, linewidth=1.5)
    if i == 0: ax.set_title('Alpha Spread')
    if i == n_fg-1: ax.set_xlabel('Frame')
    ax.set_xticks(ts); ax.grid(True, alpha=0.3)
    
    ax = axes[i, 2]
    ax.plot(ts, cov_ts[:, s], '-^', color=colors[s], markersize=4, linewidth=1.5, label='alpha_cov')
    ax.plot(ts, pixel_cov_ts[:, s]/(H*W), '--', color=colors[s], markersize=3, linewidth=1, alpha=0.5, label='pixel_cov')
    if i == 0: ax.set_title('Coverage (norm)')
    if i == n_fg-1: ax.set_xlabel('Frame')
    ax.set_xticks(ts); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle('Sample9: Depth / Spread / Coverage across 16 frames', fontsize=14)
plt.tight_layout()
plt.savefig('pos_depth_debug/sample9_depth_spread_cov_16f.png', dpi=150)
print('Saved: pos_depth_debug/sample9_depth_spread_cov_16f.png')

# 打印 FG slots 的详细数据
print('\nFG slots detail:')
for s in range(N):
    if not is_fg[:, s].any():
        continue
    print(f'\nSlot {s}:')
    print(f'  {"Frame":>5} {"Depth":>8} {"Spread":>8} {"Cov":>8} {"PixCov":>8} {"FG":>3}')
    for t in range(T):
        print(f'  {t:5d} {depth_ts[t,s]:8.4f} {spread_ts[t,s]:8.4f} {cov_ts[t,s]:8.4f} {pixel_cov_ts[t,s]:8.0f} {"Y" if is_fg[t,s] else "N":>3}')
