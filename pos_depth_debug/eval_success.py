#!/usr/bin/env python3
"""
成功复现版：R²(spread)≈0.82, spread/depth≈1.46
关键：用 ds.get_dataloader + Trainer.load_checkpoint + 训练代码流程
"""
import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import torch, numpy as np
from types import SimpleNamespace
import yaml

def setup_cuda(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    return torch.Generator().manual_seed(seed)

seed_gen = setup_cuda(42)

sys.path.insert(0, '.')
from models.dynamics import SlotDynamicsModel
from train import Trainer, create_optimizer
from train.trainer import WandBLogger
from data import get_dataset

with open('config/pretrain_phase2.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg_dict['continue_pretrain'] = True
cfg_dict['workdir'] = '/tmp/success_eval'
cfg = SimpleNamespace(**cfg_dict)

model = SlotDynamicsModel(cfg)
opt, sch = create_optimizer((p for p in model.parameters() if p.requires_grad), cfg)
wb = WandBLogger(enabled=False)
trainer = Trainer(model, opt, sch, cfg, wandb_logger=wb)
trainer.load_checkpoint('experiments/phase2_depth_spread/checkpoints/best.pt')

num_frames = getattr(cfg, 'num_frames', None) or (getattr(cfg, 'burnin_frames', 6) + getattr(cfg, 'rollout_frames', 10))
ds = get_dataset(cfg.dataset, data_path=cfg.data_root, num_frames=num_frames,
                 stride=getattr(cfg, 'slide_stride', 1), subsample=getattr(cfg, 'subsample', 1))
loader = ds.get_dataloader(batch_size=cfg.batch_size, shuffle=True, num_workers=0, generator=seed_gen)

app_dim = model.appearance_dim
all_d, all_s, all_c = [], [], []
model.eval()
with torch.no_grad():
    for i, batch in enumerate(loader):
        if i >= 47: break  # ~3000 samples
        frames = batch["video"].cuda()
        out = model(frames)
        burnin_T = out['slots']['corrected'].shape[1]
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
                        all_d.append(depth[b_idx, s_idx].item())
                        all_s.append(sp[b_idx, s_idx].item())
                        all_c.append(cov[b_idx, s_idx].item() / (H*W))
        if i % 10 == 0:
            print(f"  {i}/47 batches done", flush=True)

dm = np.array(all_d); sm = np.array(all_s); cm = np.array(all_c)
mask = dm > 0.04
dm, sm, cm = dm[mask], sm[mask], cm[mask]
d2m = dm**2

a_val = model.depth_spread_a.item()
c_val = model.depth_spread_c.item()
b_val = model.depth_spread_b.item()
d_val = model.depth_spread_d.item()
r2_s_prior = 1 - np.sum((sm - (a_val*dm+c_val))**2) / np.sum((sm - sm.mean())**2)
r2_c_prior = 1 - np.sum((cm - (b_val*d2m+d_val))**2) / np.sum((cm - cm.mean())**2)
coef_s = np.polyfit(dm, sm, 1)
coef_c = np.polyfit(d2m, cm, 1)
r2_s_poly = 1 - np.sum((sm - np.polyval(coef_s, dm))**2) / np.sum((sm - sm.mean())**2)
r2_c_poly = 1 - np.sum((cm - np.polyval(coef_c, d2m))**2) / np.sum((cm - cm.mean())**2)

print(f"\n=== SUCCESS VERSION ===")
print(f"n_fg={len(dm)}, depth mean={dm.mean():.4f}")
print(f"spread/depth median={np.median(sm/dm):.4f}")
print(f"polyfit: spread y={coef_s[0]:.4f}x+{coef_s[1]:.4f} R2={r2_s_poly:.4f}")
print(f"polyfit: cov    y={coef_c[0]:.4f}x+{coef_c[1]:.4f} R2={r2_c_poly:.4f}")
print(f"prior:   R2(spread)={r2_s_prior:.4f}, R2(cov)={r2_c_prior:.4f}")
