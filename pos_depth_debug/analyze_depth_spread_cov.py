"""深入分析三个问题：
1. depth vs spread 线性关系是否变弱（Baseline vs Phase3 对比分位数残差）
2. spread 和 coverage 是否单调
3. 为什么 depth vs cov R² > depth² vs cov R²
"""
import torch, yaml, sys, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P
from scipy.stats import spearmanr, pearsonr
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

def collect_data(model, ds, num_samples):
    all_depth = []; all_spread = []; all_cov = []; all_pixel_cov = []
    with torch.no_grad():
        gy, gx = torch.meshgrid(torch.linspace(-1,1,H,device=device), torch.linspace(-1,1,W,device=device), indexing='ij')
        for si in range(num_samples):
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
                        all_spread.append(sp)
                        all_cov.append(alpha_cov/(H*W))
                        all_pixel_cov.append(pixel_cov)
    return (np.array(all_depth), np.array(all_spread), np.array(all_cov), np.array(all_pixel_cov))

ds = OBJ3DDataset(data_path='../data/obj3d', subsample=2)

# Load models
print('Loading models...')
with open('../config/pretrain_phase2.yaml') as f: cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)
cfg.pretrain = True; cfg.freeze_slot = False; cfg.continue_pretrain = False
cfg.depth_spread_weight = 0.0
model_base = load_model('../good_checkpoints/depth_pred_best.pt', cfg)
model_base.eval()

with open('../config/pretrain_phase3.yaml') as f: cfg_dict = yaml.safe_load(f)
cfg3 = SimpleNamespace(**cfg_dict)
cfg3.pretrain = True; cfg3.freeze_slot = False; cfg3.continue_pretrain = False
model_p3 = load_model('../experiments/phase3_gru2_full/checkpoints/best.pt', cfg3)
model_p3.eval()

print('Collecting data (2500 samples)...')
d_b, sp_b, c_b, pc_b = collect_data(model_base, ds, 2500)
d_p, sp_p, c_p, pc_p = collect_data(model_p3, ds, 2500)

# ============ 问题 1: depth vs spread 线性关系变弱 ============
print('\n' + '='*60)
print('Q1: depth vs spread linearity')
print('='*60)

# 计算 Spearman (单调性) vs Pearson (线性)
for label, d, sp in [('Baseline', d_b, sp_b), ('Phase3', d_p, sp_p)]:
    pearson, _ = pearsonr(d, sp)
    spearman, _ = spearmanr(d, sp)
    print(f'{label}: Pearson={pearson:.4f}, Spearman={spearman:.4f}, gap={spearman-pearson:.4f}')

# 计算条件方差：按 depth 分桶，看每桶内 spread 的方差
print('\nConditional variance of spread given depth:')
for label, d, sp in [('Baseline', d_b, sp_b), ('Phase3', d_p, sp_p)]:
    n_bins = 10
    d_bins = np.linspace(d.min(), d.max(), n_bins+1)
    cond_var = []
    for i in range(n_bins):
        mask = (d >= d_bins[i]) & (d < d_bins[i+1])
        if mask.sum() > 5:
            cond_var.append(sp[mask].var())
    avg_cond_var = np.mean(cond_var)
    total_var = sp.var()
    ratio = avg_cond_var / total_var
    print(f'  {label}: avg_cond_var={avg_cond_var:.6f}, total_var={total_var:.6f}, ratio={ratio:.4f}')

# ============ 问题 2: spread 和 coverage 是否单调 ============
print('\n' + '='*60)
print('Q2: spread vs coverage monotonicity')
print('='*60)

for label, sp, c, pc in [('Baseline', sp_b, c_b, pc_b), ('Phase3', sp_p, c_p, pc_p)]:
    pearson_sc, _ = pearsonr(sp, c)
    spearman_sc, _ = spearmanr(sp, c)
    pearson_spc, _ = pearsonr(sp, pc)
    spearman_spc, _ = spearmanr(sp, pc)
    r2_sc = 1 - np.sum((c - P.polyval(sp, P.polyfit(sp, c, 1)))**2) / np.sum((c - c.mean())**2)
    r2_sc2 = 1 - np.sum((c - P.polyval(sp**2, P.polyfit(sp**2, c, 1)))**2) / np.sum((c - c.mean())**2)
    print(f'{label}:')
    print(f'  spread vs alpha_cov: Pearson={pearson_sc:.4f}, Spearman={spearman_sc:.4f}, R²(linear)={r2_sc:.4f}')
    print(f'  spread² vs alpha_cov: R²={r2_sc2:.4f}')
    print(f'  spread vs pixel_cov: Pearson={pearson_spc:.4f}, Spearman={spearman_spc:.4f}')

# ============ 问题 3: 为什么 depth vs cov R² > depth² vs cov R² ============
print('\n' + '='*60)
print('Q3: Why R²(depth,cov) > R²(depth²,cov)?')
print('='*60)

for label, d, c in [('Baseline', d_b, c_b), ('Phase3', d_p, c_p)]:
    r2_dc = 1 - np.sum((c - P.polyval(d, P.polyfit(d, c, 1)))**2) / np.sum((c - c.mean())**2)
    r2_d2c = 1 - np.sum((c - P.polyval(d**2, P.polyfit(d**2, c, 1)))**2) / np.sum((c - c.mean())**2)
    
    # 分析：depth 的分布范围
    print(f'{label}:')
    print(f'  R²(depth, cov)={r2_dc:.4f}, R²(depth², cov)={r2_d2c:.4f}')
    print(f'  depth range: [{d.min():.4f}, {d.max():.4f}], std={d.std():.4f}')
    print(f'  depth² range: [{(d**2).min():.6f}, {(d**2).max():.6f}], std={(d**2).std():.6f}')
    
    # 尝试 depth^alpha vs cov 最优 alpha
    best_alpha = 1.0; best_r2 = 0
    for alpha in np.arange(0.5, 3.01, 0.1):
        d_a = d**alpha
        r2 = 1 - np.sum((c - P.polyval(d_a, P.polyfit(d_a, c, 1)))**2) / np.sum((c - c.mean())**2)
        if r2 > best_r2:
            best_r2 = r2; best_alpha = alpha
    print(f'  Best alpha for depth^alpha vs cov: alpha={best_alpha:.1f}, R²={best_r2:.4f}')
    
    # depth vs cov 的残差分布
    co = P.polyfit(d, c, 1)
    res_dc = c - P.polyval(d, co)
    co2 = P.polyfit(d**2, c, 1)
    res_d2c = c - P.polyval(d**2, co2)
    print(f'  Residual std: depth->cov={res_dc.std():.6f}, depth²->cov={res_d2c.std():.6f}')

# ============ 综合散点图 ============
print('\nPlotting comprehensive figure...')

fig, axes = plt.subplots(2, 4, figsize=(28, 12))

datasets = [
    ('Baseline', d_b, sp_b, c_b, pc_b),
    ('Phase3', d_p, sp_p, c_p, pc_p),
]

for row, (label, d, sp, c, pc) in enumerate(datasets):
    # depth vs spread (tighter view)
    ax = axes[row, 0]
    ax.scatter(d, sp, s=2, alpha=0.15, c='steelblue')
    co = P.polyfit(d, sp, 1)
    x_line = np.linspace(d.min(), d.max(), 100)
    ax.plot(x_line, P.polyval(x_line, co), 'r-', linewidth=2)
    pearson, _ = pearsonr(d, sp)
    spearman, _ = spearmanr(d, sp)
    ax.set_xlabel('Depth (≈Scale)'); ax.set_ylabel('Alpha Spread')
    ax.set_title(f'{label}\nDepth vs Spread\nP={pearson:.3f} S={spearman:.3f}')
    
    # spread vs coverage
    ax = axes[row, 1]
    ax.scatter(sp, c, s=2, alpha=0.15, c='darkorange')
    co = P.polyfit(sp, c, 1)
    x_line = np.linspace(sp.min(), sp.max(), 100)
    ax.plot(x_line, P.polyval(x_line, co), 'r-', linewidth=2)
    pearson, _ = pearsonr(sp, c)
    spearman, _ = spearmanr(sp, c)
    ax.set_xlabel('Alpha Spread'); ax.set_ylabel('Alpha Coverage (norm)')
    ax.set_title(f'{label}\nSpread vs Coverage\nP={pearson:.3f} S={spearman:.3f}')
    
    # depth vs coverage (linear)
    ax = axes[row, 2]
    ax.scatter(d, c, s=2, alpha=0.15, c='forestgreen')
    co = P.polyfit(d, c, 1)
    x_line = np.linspace(d.min(), d.max(), 100)
    ax.plot(x_line, P.polyval(x_line, co), 'r-', linewidth=2)
    r2 = 1 - np.sum((c-P.polyval(d,co))**2)/np.sum((c-c.mean())**2)
    ax.set_xlabel('Depth (≈Scale)'); ax.set_ylabel('Alpha Coverage (norm)')
    ax.set_title(f'{label}\nDepth vs Coverage\nR²={r2:.4f}')
    
    # depth vs coverage 分桶条件方差可视化
    ax = axes[row, 3]
    n_bins = 20
    d_bins = np.linspace(d.min(), d.max(), n_bins+1)
    means = []; stds = []; centers = []
    for i in range(n_bins):
        mask = (d >= d_bins[i]) & (d < d_bins[i+1])
        if mask.sum() > 10:
            means.append(c[mask].mean())
            stds.append(c[mask].std())
            centers.append((d_bins[i]+d_bins[i+1])/2)
    means = np.array(means); stds = np.array(stds); centers = np.array(centers)
    ax.plot(centers, means, 'b-o', markersize=4, label='mean')
    ax.fill_between(centers, means-stds, means+stds, alpha=0.2, color='blue', label='±1σ')
    ax.set_xlabel('Depth (≈Scale)'); ax.set_ylabel('Alpha Coverage (norm)')
    ax.set_title(f'{label}\nDepth vs Coverage (binned)\n±1σ spread')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('depth_spread_cov_analysis.png', dpi=150)
print('Saved: depth_spread_cov_analysis.png')
