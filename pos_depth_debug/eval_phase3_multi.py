"""Phase3 多帧 depth/coverage 锚定效果评估
重点关注:
1. 多帧下 depth 和 coverage 的 R²
2. sample9 中 depth 变化和 coverage 随时间变化的单调性是否一致
3. 多个样本的评估
"""
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

with open('config/pretrain_phase3.yaml') as f: cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)
cfg.pretrain = True; cfg.freeze_slot = False; cfg.continue_pretrain = False

ds = OBJ3DDataset(data_path='data/obj3d', subsample=2)
model = load_model('experiments/phase3_gru2_full/checkpoints/best.pt', cfg)
model.eval()

# ============ Part 1: 大规模 R² 评估 (500样本) ============
print("=" * 60)
print("Part 1: R² evaluation (500 samples, multi-frame)")
print("=" * 60)

all_depth = []; all_spread = []; all_cov = []; all_depth_raw = []

with torch.no_grad():
    gy, gx = torch.meshgrid(torch.linspace(-1,1,H,device=device), torch.linspace(-1,1,W,device=device), indexing='ij')
    for si in range(500):
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

d = np.array(all_depth); sp = np.array(all_spread); c = np.array(all_cov)
d_all = np.array(all_depth_raw)
print(f'FG points: {len(d)}, total slots: {len(d_all)}')
print(f'FG depth range: [{d.min():.4f}, {d.max():.4f}]')
print(f'depth>0.5 filtered: {(d_all > 0.5).sum()} / {len(d_all)}')

co1 = P.polyfit(d, sp, 1)
r2_sp = 1 - np.sum((sp-P.polyval(d,co1))**2)/np.sum((sp-sp.mean())**2)
d2 = d**2
co2 = P.polyfit(d2, c, 1)
r2_cov = 1 - np.sum((c-P.polyval(d2,co2))**2)/np.sum((c-c.mean())**2)
co3 = P.polyfit(d, c, 1)
r2_d_cov = 1 - np.sum((c-P.polyval(d,co3))**2)/np.sum((c-c.mean())**2)
print(f'R²(depth, spread) = {r2_sp:.4f}')
print(f'R²(depth², cov) = {r2_cov:.4f}')
print(f'R²(depth, cov) = {r2_d_cov:.4f}')

# ============ Part 2: Sample9 深度时间序列分析 ============
print("\n" + "=" * 60)
print("Part 2: Sample9 depth/coverage temporal analysis")
print("=" * 60)

sample9 = torch.load('data/sample9/sample9.pt')['video']  # (16, 3, 64, 64)

with torch.no_grad():
    video9 = sample9.unsqueeze(0).to(device)
    out9 = model(video9)
    slots9 = out9['slots']['corrected'][0]  # (T, N, 67)
    T9 = slots9.shape[0]; N9 = slots9.shape[1]

    depth_ts = np.zeros((T9, N9))
    cov_ts = np.zeros((T9, N9))
    pixel_cov_ts = np.zeros((T9, N9))
    spread_ts = np.zeros((T9, N9))

    for t in range(T9):
        _, a_full, _ = model.decoder(slots9[t].unsqueeze(0), return_alphas=True, return_rgb=True)
        a = a_full[0, :, 0]
        dominant_slot = a.argmax(dim=0)
        for s in range(N9):
            depth_ts[t, s] = slots9[t, s, app_dim+2].item()
            alpha_cov_ts = a[s].sum().item()
            pix_cov = (dominant_slot == s).sum().item()
            cov_ts[t, s] = alpha_cov_ts / (H*W)
            pixel_cov_ts[t, s] = pix_cov
            a_max = a[s].max().item()
            if a_max > 0.3 and pix_cov > 50:
                a_s = a[s]; a_n = a_s/(alpha_cov_ts+1e-8)
                cx = (a_n*gx).sum().item(); cy = (a_n*gy).sum().item()
                spread_ts[t, s] = np.sqrt((a_n*((gx-cx)**2+(gy-cy)**2)).sum().item())

# 识别 FG slots（至少在半数帧中满足 FG 条件）
fg_slots = []
for s in range(N9):
    fg_count = sum(1 for t in range(T9) if pixel_cov_ts[t, s] > 50 and depth_ts[t, s] < 0.5)
    if fg_count >= T9 * 0.5:
        fg_slots.append(s)

print(f'FG slots: {fg_slots}')
for s in fg_slots:
    d_arr = depth_ts[:, s]
    c_arr = pixel_cov_ts[:, s]
    d_mono = d_arr[-1] - d_arr[0]
    c_mono = c_arr[-1] - c_arr[0]
    # 计算符号一致性：depth 变化方向和 coverage 变化方向
    # 如果 depth 增大 -> 物体应变小 -> coverage 应减小
    # 所以 depth 和 coverage 的单调性应该相反
    consistent = (d_mono > 0 and c_mono < 0) or (d_mono < 0 and c_mono > 0) or (abs(d_mono) < 0.01 or abs(c_mono) < 5)
    print(f'  Slot {s}: depth [{d_arr[0]:.4f} -> {d_arr[-1]:.4f}] (Δ={d_mono:+.4f}), '
          f'pixel_cov [{c_arr[0]:.0f} -> {c_arr[-1]:.0f}] (Δ={c_mono:+.0f}), '
          f'consistent={consistent}')

# 绘制 sample9 时间序列图
fig, axes = plt.subplots(len(fg_slots), 3, figsize=(18, 4*len(fg_slots)))
if len(fg_slots) == 1:
    axes = axes.reshape(1, -1)

for i, s in enumerate(fg_slots):
    ts = np.arange(T9)
    axes[i, 0].plot(ts, depth_ts[:, s], 'b-o', markersize=3)
    axes[i, 0].set_title(f'Slot {s}: Depth'); axes[i, 0].set_xlabel('Frame')
    axes[i, 1].plot(ts, pixel_cov_ts[:, s], 'r-o', markersize=3)
    axes[i, 1].set_title(f'Slot {s}: Pixel Coverage'); axes[i, 1].set_xlabel('Frame')
    axes[i, 2].plot(ts, cov_ts[:, s], 'g-o', markersize=3, label='alpha_cov')
    axes[i, 2].plot(ts, spread_ts[:, s], 'm-s', markersize=3, label='spread')
    axes[i, 2].set_title(f'Slot {s}: Alpha Cov & Spread'); axes[i, 2].set_xlabel('Frame')
    axes[i, 2].legend()

plt.tight_layout()
plt.savefig('pos_depth_debug/phase3_sample9_temporal.png', dpi=150)
print('Saved: pos_depth_debug/phase3_sample9_temporal.png')

# ============ Part 3: 多样本单调性统计 ============
print("\n" + "=" * 60)
print("Part 3: Multi-sample monotonicity consistency (50 samples)")
print("=" * 60)

n_consistent = 0; n_inconsistent = 0; n_ambiguous = 0
n_total_trajectories = 0
all_mono_data = []

with torch.no_grad():
    for si in range(50):
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
                pix_cov = (dominant_slot == s).sum().item()
                depth = slots[t, s, app_dim+2].item()
                depth_ts_local = np.zeros(T)
                cov_ts_local = np.zeros(T)
                valid = True
                for t2 in range(T):
                    _, a2, _ = model.decoder(slots[t2].unsqueeze(0), return_alphas=True, return_rgb=True)
                    a2s = a2[0, :, 0]
                    pix_cov2 = (a2s.argmax(dim=0) == s).sum().item()
                    depth2 = slots[t2, s, app_dim+2].item()
                    depth_ts_local[t2] = depth2
                    cov_ts_local[t2] = pix_cov2
                    if depth2 > 0.5:
                        valid = False

                if not valid:
                    continue
                avg_cov = cov_ts_local.mean()
                if avg_cov < 50:
                    continue

                n_total_trajectories += 1
                d_mono = depth_ts_local[-1] - depth_ts_local[0]
                c_mono = cov_ts_local[-1] - cov_ts_local[0]

                if abs(d_mono) < 0.005 or abs(c_mono) < 10:
                    n_ambiguous += 1
                elif (d_mono > 0 and c_mono < 0) or (d_mono < 0 and c_mono > 0):
                    n_consistent += 1
                else:
                    n_inconsistent += 1
                    all_mono_data.append({'sample': si, 'slot': s, 'd_mono': d_mono, 'c_mono': c_mono})
                break  # 只评估一次 per slot

# 修正：上面逻辑有 bug，一个 slot 只能被评估一次。重写。
print("Rewriting with correct logic...")

n_consistent = 0; n_inconsistent = 0; n_ambiguous = 0
n_total = 0
inconsistent_details = []

with torch.no_grad():
    for si in range(50):
        sample = ds[si]
        video = sample['video'].unsqueeze(0).to(device)
        out = model(video)
        slots = out['slots']['corrected'][0]
        T = slots.shape[0]; N = slots.shape[1]

        # 预计算所有帧的 alpha
        slot_depths = np.zeros((T, N))
        slot_pixel_covs = np.zeros((T, N))
        for t in range(T):
            _, a_full, _ = model.decoder(slots[t].unsqueeze(0), return_alphas=True, return_rgb=True)
            a = a_full[0, :, 0]
            dominant = a.argmax(dim=0)
            for s in range(N):
                slot_depths[t, s] = slots[t, s, app_dim+2].item()
                slot_pixel_covs[t, s] = (dominant == s).sum().item()

        # 对每个 slot 检查
        for s in range(N9):
            avg_cov = slot_pixel_covs[:, s].mean()
            if avg_cov < 50:
                continue
            if (slot_depths[:, s] > 0.5).any():
                continue

            n_total += 1
            d_mono = slot_depths[-1, s] - slot_depths[0, s]
            c_mono = slot_pixel_covs[-1, s] - slot_pixel_covs[0, s]

            if abs(d_mono) < 0.005 or abs(c_mono) < 10:
                n_ambiguous += 1
            elif (d_mono > 0 and c_mono < 0) or (d_mono < 0 and c_mono > 0):
                n_consistent += 1
            else:
                n_inconsistent += 1
                inconsistent_details.append({
                    'sample': si, 'slot': s,
                    'd_range': f'[{slot_depths[0,s]:.4f}, {slot_depths[-1,s]:.4f}]',
                    'c_range': f'[{slot_pixel_covs[0,s]:.0f}, {slot_pixel_covs[-1,s]:.0f}]',
                    'd_mono': d_mono, 'c_mono': c_mono
                })

print(f'Total FG trajectories: {n_total}')
print(f'Consistent (depth↑cov↓ or depth↓cov↑): {n_consistent} ({n_consistent/max(n_total,1)*100:.1f}%)')
print(f'Inconsistent (same direction): {n_inconsistent} ({n_inconsistent/max(n_total,1)*100:.1f}%)')
print(f'Ambiguous (too small change): {n_ambiguous} ({n_ambiguous/max(n_total,1)*100:.1f}%)')

if inconsistent_details:
    print('\nInconsistent cases:')
    for d in inconsistent_details[:10]:
        print(f"  Sample {d['sample']} Slot {d['slot']}: depth {d['d_range']} Δd={d['d_mono']:+.4f}, cov {d['c_range']} Δc={d['c_mono']:+.0f}")

# ============ Part 4: Sample9 散点图 ============
print("\n" + "=" * 60)
print("Part 4: Phase3 R² scatter plots")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.scatter(d, sp, s=2, alpha=0.3)
x_line = np.linspace(d.min(), d.max(), 100)
ax.plot(x_line, P.polyval(x_line, co1), 'r-', linewidth=2, label=f'R²={r2_sp:.4f}')
ax.set_xlabel('Depth'); ax.set_ylabel('Alpha Spread')
ax.set_title('Phase3: Depth vs Alpha Spread'); ax.legend()

ax = axes[1]
ax.scatter(d2, c, s=2, alpha=0.3)
x_line = np.linspace(d2.min(), d2.max(), 100)
ax.plot(x_line, P.polyval(x_line, co2), 'r-', linewidth=2, label=f'R²={r2_cov:.4f}')
ax.set_xlabel('Depth²'); ax.set_ylabel('Alpha Coverage (norm)')
ax.set_title('Phase3: Depth² vs Alpha Coverage'); ax.legend()

ax = axes[2]
ax.scatter(d, c, s=2, alpha=0.3)
x_line = np.linspace(d.min(), d.max(), 100)
ax.plot(x_line, P.polyval(x_line, co3), 'r-', linewidth=2, label=f'R²={r2_d_cov:.4f}')
ax.set_xlabel('Depth'); ax.set_ylabel('Alpha Coverage (norm)')
ax.set_title('Phase3: Depth vs Alpha Coverage'); ax.legend()

plt.tight_layout()
plt.savefig('pos_depth_debug/phase3_r2_scatter.png', dpi=150)
print('Saved: pos_depth_debug/phase3_r2_scatter.png')

np.savez('pos_depth_debug/phase3_eval_data.npz',
         depth=d, spread=sp, cov=c, depth_raw=d_all,
         depth_ts9=depth_ts, cov_ts9=cov_ts, pixel_cov_ts9=pixel_cov_ts,
         fg_slots9=fg_slots)
print('Saved: pos_depth_debug/phase3_eval_data.npz')
