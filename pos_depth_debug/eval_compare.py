#!/usr/bin/env python3
"""
逐项对比 success 和 fail 版本的全流程差异
在同一数据上对比，逐步消除差异来源
"""
import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch, numpy as np
from types import SimpleNamespace
import yaml

def setup_cuda(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    return torch.Generator().manual_seed(seed)

seed_gen = setup_cuda(42)

sys.path.insert(0, '.')
from models.dynamics import SlotDynamicsModel
from train import Trainer, create_optimizer
from train.trainer import WandBLogger
from data import get_dataset
from data.obj3d_dataset import OBJ3DDataset

with open('config/pretrain_phase2.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg_dict['continue_pretrain'] = True
cfg_dict['workdir'] = '/tmp/compare_eval'
cfg = SimpleNamespace(**cfg_dict)

# ========== 1. 两种方式加载模型 ==========
print("=" * 60)
print("STEP 1: Compare model loading methods")
print("=" * 60)

# Success way: Trainer.load_checkpoint
model_s = SlotDynamicsModel(cfg)
opt, sch = create_optimizer((p for p in model_s.parameters() if p.requires_grad), cfg)
wb = WandBLogger(enabled=False)
trainer = Trainer(model_s, opt, sch, cfg, wandb_logger=wb)
trainer.load_checkpoint('experiments/phase2_depth_spread/checkpoints/best.pt')
model_s.eval().cuda()

# Fail way: model.load_state_dict
model_f = SlotDynamicsModel(cfg)
ckpt = torch.load('experiments/phase2_depth_spread/checkpoints/best.pt', map_location='cpu')
model_f.load_state_dict(ckpt['model'], strict=False)
model_f.eval().cuda()

# Compare weights
sd_s = model_s.state_dict()
sd_f = model_f.state_dict()
n_match = 0; n_mismatch = 0
for k in sd_s:
    if k in sd_f:
        if torch.equal(sd_s[k], sd_f[k]):
            n_match += 1
        else:
            n_mismatch += 1
            print(f"  MISMATCH: {k}, diff_max={((sd_s[k]-sd_f[k]).abs().max()).item():.2e}")
    else:
        print(f"  ONLY IN SUCCESS: {k}")
for k in sd_f:
    if k not in sd_s:
        print(f"  ONLY IN FAIL: {k}")

print(f"Weight comparison: {n_match} match, {n_mismatch} mismatch")

# Compare prior params
print(f"\nPrior params (success): a={model_s.depth_spread_a.item():.4f}, b={model_s.depth_spread_b.item():.4f}, c={model_s.depth_spread_c.item():.4f}, d={model_s.depth_spread_d.item():.4f}")
print(f"Prior params (fail):    a={model_f.depth_spread_a.item():.4f}, b={model_f.depth_spread_b.item():.4f}, c={model_f.depth_spread_c.item():.4f}, d={model_f.depth_spread_d.item():.4f}")

# ========== 2. 同一批数据，对比 forward 输出 ==========
print("\n" + "=" * 60)
print("STEP 2: Forward pass comparison on same data")
print("=" * 60)

num_frames = getattr(cfg, 'num_frames', None) or (getattr(cfg, 'burnin_frames', 6) + getattr(cfg, 'rollout_frames', 10))
ds = get_dataset(cfg.dataset, data_path=cfg.data_root, num_frames=num_frames,
                 stride=getattr(cfg, 'slide_stride', 1), subsample=getattr(cfg, 'subsample', 1))
loader = ds.get_dataloader(batch_size=2, shuffle=False, num_workers=0)

batch = next(iter(loader))
frames = batch["video"].cuda()
print(f"Input frames shape: {frames.shape}")

with torch.no_grad():
    out_s = model_s(frames)
    out_f = model_f(frames)

# Compare slots
slots_s = out_s['slots']['corrected']
slots_f = out_f['slots']['corrected']
print(f"\nSlots shape: success={slots_s.shape}, fail={slots_f.shape}")
slot_diff = (slots_s - slots_f).abs()
print(f"Slots diff: max={slot_diff.max().item():.2e}, mean={slot_diff.mean().item():.2e}")

# Compare alpha
alpha_s = out_s['alpha']
alpha_f = out_f['alpha']
print(f"Alpha shape: success={alpha_s.shape}, fail={alpha_f.shape}")
alpha_diff = (alpha_s - alpha_f).abs()
print(f"Alpha diff: max={alpha_diff.max().item():.2e}, mean={alpha_diff.mean().item():.2e}")

# ========== 3. 对比 depth 值 ==========
print("\n" + "=" * 60)
print("STEP 3: Compare depth values per slot")
print("=" * 60)

app_dim = model_s.appearance_dim
for b in range(slots_s.shape[0]):
    for t in range(slots_s.shape[1]):
        depth_s = slots_s[b, t, :, app_dim+2]
        depth_f = slots_f[b, t, :, app_dim+2]
        print(f"  batch={b}, t={t}:")
        print(f"    success depth: {depth_s.cpu().numpy().round(4)}")
        print(f"    fail    depth: {depth_f.cpu().numpy().round(4)}")
        print(f"    diff: {(depth_s-depth_f).abs().max().item():.2e}")

# ========== 4. 对比两种 alpha 处理方式 ==========
print("\n" + "=" * 60)
print("STEP 4: Compare alpha processing methods")
print("=" * 60)

# Success way: iterate time, keep batch dim
burnin_T = slots_s.shape[1]
for t in range(burnin_T):
    alpha_t = alpha_s[:, :, t]
    if alpha_t.dim() == 5:
        alpha_2d = alpha_t.squeeze(2)
    else:
        alpha_2d = alpha_t
    B, N, H, W = alpha_2d.shape
    gy, gx = torch.meshgrid(torch.linspace(-1,1,H,device='cuda'), torch.linspace(-1,1,W,device='cuda'), indexing='ij')
    gx_b = gx.unsqueeze(0).unsqueeze(0).expand(B,N,H,W)
    gy_b = gy.unsqueeze(0).unsqueeze(0).expand(B,N,H,W)
    a_sum = alpha_2d.sum(dim=[-2,-1], keepdim=True) + 1e-8
    a_norm = alpha_2d / a_sum
    cx_s = (a_norm * gx_b).sum(dim=[-2,-1])
    cy_s = (a_norm * gy_b).sum(dim=[-2,-1])
    sp_s = torch.sqrt((a_norm * ((gx_b - cx_s.unsqueeze(-1).unsqueeze(-1))**2 +
                                  (gy_b - cy_s.unsqueeze(-1).unsqueeze(-1))**2)).sum(dim=[-2,-1]))
    depth_s_all = slots_s[:, t, :, app_dim+2]
    a_max_s = alpha_2d.amax(dim=[-2,-1])
    cov_s = alpha_2d.sum(dim=[-2,-1])
    fg_s = (cov_s > 20) & (cov_s < 1500) & (sp_s > 0.01) & (a_max_s > 0.3) & (depth_s_all < 0.5)

    print(f"\n  Time {t}: alpha_2d shape={alpha_2d.shape}")

    # Fail way: squeeze everything, process one sample at a time
    for b_idx in range(B):
        a_fail = alpha_s[b_idx].squeeze()  # same data but fail-style squeeze
        print(f"    batch {b_idx}: success alpha_2d[{b_idx}] shape={alpha_2d[b_idx].shape}, fail squeeze shape={a_fail.shape}")
        if a_fail.shape != alpha_2d[b_idx].shape:
            print(f"    *** SHAPE MISMATCH! success={alpha_2d[b_idx].shape} vs fail={a_fail.shape}")

        # Compute spread the fail way
        N_f, H_f, W_f = a_fail.shape
        gy2, gx2 = torch.meshgrid(torch.linspace(-1,1,H_f,device='cuda'), torch.linspace(-1,1,W_f,device='cuda'), indexing='ij')
        a_sum2 = a_fail.sum(dim=[-2,-1], keepdim=True) + 1e-8
        a_norm2 = a_fail / a_sum2
        cx2 = (a_norm2 * gx2.unsqueeze(0)).sum(dim=[-2,-1])
        cy2 = (a_norm2 * gy2.unsqueeze(0)).sum(dim=[-2,-1])
        sp2 = torch.sqrt((a_norm2 * ((gx2.unsqueeze(0)-cx2.unsqueeze(-1).unsqueeze(-1))**2 +
                                      (gy2.unsqueeze(0)-cy2.unsqueeze(-1).unsqueeze(-1))**2)).sum(dim=[-2,-1]))

        # Compare spread values
        sp_success = sp_s[b_idx]
        sp_fail = sp2
        print(f"    spread: success={sp_success.cpu().numpy().round(4)}")
        print(f"    spread: fail   ={sp_fail.cpu().numpy().round(4)}")
        print(f"    spread diff max: {(sp_success-sp_fail).abs().max().item():.2e}")

        # Compare cov
        cov_success = cov_s[b_idx]
        cov_fail = a_fail.sum(dim=[-2,-1])
        print(f"    cov: success={cov_success.cpu().numpy().round(2)}")
        print(f"    cov: fail   ={cov_fail.cpu().numpy().round(2)}")

        # Compare fg masks
        depth_f_val = depth_s_all[b_idx]
        a_max_f = a_fail.amax(dim=[-2,-1])
        fg_fail = (cov_fail > 20) & (cov_fail < 1500) & (sp_fail > 0.01) & (a_max_f > 0.3) & (depth_f_val < 0.5)
        print(f"    fg_success={fg_s[b_idx].cpu().numpy()}")
        print(f"    fg_fail   ={fg_fail.cpu().numpy()}")

# ========== 5. 用 success loader 跑 fail-style 处理 ==========
print("\n" + "=" * 60)
print("STEP 5: Full 3000-sample comparison")
print("=" * 60)

# Reset loader
loader = ds.get_dataloader(batch_size=cfg.batch_size, shuffle=True, num_workers=0, generator=seed_gen)

all_d_s, all_s_s, all_c_s = [], [], []  # success-style
all_d_f, all_s_f, all_c_f = [], [], []  # fail-style (on same data)

with torch.no_grad():
    for i, batch in enumerate(loader):
        if i >= 47: break
        frames = batch["video"].cuda()
        out = model_s(frames)
        burnin_T = out['slots']['corrected'].shape[1]

        # --- Success style: iterate all time frames ---
        for t in range(burnin_T):
            slots_t = out['slots']['corrected'][:, t]
            alpha_t = out['alpha'][:, :, t]
            if alpha_t.dim() == 5:
                alpha_2d = alpha_t.squeeze(2)
            else:
                alpha_2d = alpha_t
            B, N, H, W = alpha_2d.shape
            gy, gx = torch.meshgrid(torch.linspace(-1,1,H,device='cuda'), torch.linspace(-1,1,W,device='cuda'), indexing='ij')
            gx_b = gx.unsqueeze(0).unsqueeze(0).expand(B,N,H,W)
            gy_b = gy.unsqueeze(0).unsqueeze(0).expand(B,N,H,W)
            a_sum = alpha_2d.sum(dim=[-2,-1], keepdim=True) + 1e-8
            a_norm = alpha_2d / a_sum
            cx = (a_norm * gx_b).sum(dim=[-2,-1])
            cy = (a_norm * gy_b).sum(dim=[-2,-1])
            sp = torch.sqrt((a_norm * ((gx_b - cx.unsqueeze(-1).unsqueeze(-1))**2 +
                                        (gy_b - cy.unsqueeze(-1).unsqueeze(-1))**2)).sum(dim=[-2,-1]))
            depth = slots_t[:, :, app_dim+2]
            a_max = alpha_2d.amax(dim=[-2,-1])
            cov = alpha_2d.sum(dim=[-2,-1])
            fg = (cov > 20) & (cov < 1500) & (sp > 0.01) & (a_max > 0.3) & (depth < 0.5)
            for b_idx in range(B):
                for s_idx in range(N):
                    if fg[b_idx, s_idx]:
                        all_d_s.append(depth[b_idx, s_idx].item())
                        all_s_s.append(sp[b_idx, s_idx].item())
                        all_c_s.append(cov[b_idx, s_idx].item() / (H*W))

        # --- Fail style: only frame 0, per-sample squeeze ---
        for b_idx in range(frames.shape[0]):
            slots_b = out['slots']['corrected'][b_idx, 0]  # frame 0
            a_fail = out['alpha'][b_idx].squeeze()
            N2, H2, W2 = a_fail.shape
            gy2, gx2 = torch.meshgrid(torch.linspace(-1,1,H2,device='cuda'), torch.linspace(-1,1,W2,device='cuda'), indexing='ij')
            a_sum2 = a_fail.sum(dim=[-2,-1], keepdim=True) + 1e-8
            a_norm2 = a_fail / a_sum2
            cx2 = (a_norm2 * gx2.unsqueeze(0)).sum(dim=[-2,-1])
            cy2 = (a_norm2 * gy2.unsqueeze(0)).sum(dim=[-2,-1])
            sp2 = torch.sqrt((a_norm2 * ((gx2.unsqueeze(0)-cx2.unsqueeze(-1).unsqueeze(-1))**2 +
                                          (gy2.unsqueeze(0)-cy2.unsqueeze(-1).unsqueeze(-1))**2)).sum(dim=[-2,-1]))
            depth2 = slots_b[:, app_dim+2]
            a_max2 = a_fail.amax(dim=[-2,-1])
            cov2 = a_fail.sum(dim=[-2,-1])
            fg2 = (cov2 > 20) & (cov2 < 1500) & (sp2 > 0.01) & (a_max2 > 0.3) & (depth2 < 0.5)
            for s_idx in range(N2):
                if fg2[s_idx] and depth2[s_idx] > 0.04:
                    all_d_f.append(depth2[s_idx].item())
                    all_s_f.append(sp2[s_idx].item())
                    all_c_f.append(cov2[s_idx].item() / (H2*W2))

        if i % 10 == 0:
            print(f"  {i}/47 done (success: {len(all_d_s)}, fail: {len(all_d_f)})", flush=True)

# Results
dm_s = np.array(all_d_s); sm_s = np.array(all_s_s); cm_s = np.array(all_c_s)
dm_f = np.array(all_d_f); sm_f = np.array(all_s_f); cm_f = np.array(all_c_f)
mask_s = dm_s > 0.04; mask_f = dm_f > 0.04
dm_s, sm_s, cm_s = dm_s[mask_s], sm_s[mask_s], cm_s[mask_s]
dm_f, sm_f, cm_f = dm_f[mask_f], sm_f[mask_f], cm_f[mask_f]

print(f"\n--- SUCCESS STYLE (all frames) ---")
print(f"n_fg={len(dm_s)}, depth mean={dm_s.mean():.4f}")
print(f"spread/depth median={np.median(sm_s/dm_s):.4f}")

print(f"\n--- FAIL STYLE (frame 0 only, per-sample squeeze) ---")
print(f"n_fg={len(dm_f)}, depth mean={dm_f.mean():.4f}")
print(f"spread/depth median={np.median(sm_f/dm_f):.4f}")

# The big question: same data, same model, do depth distributions differ?
print(f"\n--- DEPTH DISTRIBUTION COMPARISON ---")
print(f"Success depth: mean={dm_s.mean():.4f}, median={np.median(dm_s):.4f}, std={dm_s.std():.4f}, min={dm_s.min():.4f}, max={dm_s.max():.4f}")
print(f"Fail depth:    mean={dm_f.mean():.4f}, median={np.median(dm_f):.4f}, std={dm_f.std():.4f}, min={dm_f.min():.4f}, max={dm_f.max():.4f}")
