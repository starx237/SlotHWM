#!/usr/bin/env python3
"""续训1步 + 评估 + 保存所有中间数据，复现wandb评估结果"""
import os, sys, yaml, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from types import SimpleNamespace

with open('config/pretrain_phase2.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg_dict['continue_pretrain'] = True
cfg_dict['workdir'] = '/tmp/eval_debug'
cfg_dict['depth_spread_weight'] = 0.1
cfg = SimpleNamespace(**cfg_dict)

from models.dynamics import SlotDynamicsModel
from train import Trainer, create_optimizer
from data import get_dataset

model = SlotDynamicsModel(cfg)
ckpt = torch.load('experiments/phase2_depth_spread/checkpoints/step_30000.pt', map_location='cpu')
missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
print(f'Loaded: missing={len(missing)}, unexpected={len(unexpected)}')

# 打印 prior 参数
print(f'Prior params BEFORE any step:')
print(f'  a={model.depth_spread_a.item():.6f}')
print(f'  b={model.depth_spread_b.item():.6f}')
print(f'  c={model.depth_spread_c.item():.6f}')
print(f'  d={model.depth_spread_d.item():.6f}')

# 检查 checkpoint 中保存的 prior 参数
print(f'\nCheckpoint prior params:')
for k in ['depth_spread_a', 'depth_spread_b', 'depth_spread_c', 'depth_spread_d']:
    if k in ckpt['model']:
        print(f'  {k} = {ckpt["model"][k]}')
    else:
        print(f'  {k} NOT IN CHECKPOINT')

# 打印 optimizer 状态中的 prior 参数
if 'optimizer' in ckpt:
    opt_state = ckpt['optimizer']
    # 找到 prior 参数的 param index
    for i, pg in enumerate(opt_state['param_groups']):
        lr = pg.get('lr', 'N/A')
        initial_lr = pg.get('initial_lr', 'N/A')
        print(f'  Param group {i}: lr={lr}, initial_lr={initial_lr}')
    # 检查 param groups 中的 params
    param_names = [n for n, _ in model.named_parameters()]
    prior_indices = [i for i, n in enumerate(param_names) if 'depth_spread' in n]
    print(f'  Prior param indices in model: {prior_indices}')
    print(f'  Prior param names: {[param_names[i] for i in prior_indices]}')
    if 'state' in opt_state:
        for idx in prior_indices:
            if idx in opt_state['state']:
                st = opt_state['state'][idx]
                print(f'  State for param {idx} ({param_names[idx]}):')
                for sk, sv in st.items():
                    if isinstance(sv, torch.Tensor):
                        print(f'    {sk}: shape={sv.shape}, val={sv.flatten()[:5]}')
                    else:
                        print(f'    {sk}: {sv}')

model = model.cuda()

# 创建 trainer
optimizer, scheduler = create_optimizer(
    (p for p in model.parameters() if p.requires_grad), cfg)

# 跳过 optimizer 加载（param group 不匹配），只评估
print(f'\nSkipping optimizer load (param group mismatch), eval only')

# 创建数据集
num_frames = getattr(cfg, 'num_frames', None) or (getattr(cfg, 'burnin_frames', 6) + getattr(cfg, 'rollout_frames', 10))
slide_stride = getattr(cfg, 'slide_stride', 1)
subsample = getattr(cfg, 'subsample', 1)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
seed_gen = torch.Generator().manual_seed(42)
ds = get_dataset(cfg.dataset, data_path=cfg.data_root,
                 num_frames=num_frames, stride=slide_stride,
                 subsample=subsample)
loader = ds.get_dataloader(batch_size=cfg.batch_size, shuffle=True,
                           num_workers=0, generator=seed_gen)

# 创建 trainer
from train.trainer import WandBLogger
wb_logger = WandBLogger(enabled=False)
trainer = Trainer(model, optimizer, scheduler, cfg, wandb_logger=wb_logger)

# 加载 trainer 状态
trainer.global_step = ckpt.get('global_step', 0)
print(f'\nStarting from step {trainer.global_step}')

# 评估 - 1步训练前
print(f'\n=== EVAL BEFORE TRAINING ===')
r2_s, r2_c, _ = trainer._eval_depth_spread_r2(loader, n_batches=10, step=0)
print(f'BEFORE: R²(spread)={r2_s:.4f}, R²(cov)={r2_c:.4f}')

# 加载保存的数据
data = np.load('/tmp/eval_depth_spread_data.npz')
print(f'Data shapes: depth={data["depth"].shape}, spread={data["spread"].shape}, cov_norm={data["cov_norm"].shape}')
print(f'Data saved R²: spread={data["r2_spread"]:.4f}, cov={data["r2_cov"]:.4f}')
print(f'Depth range: [{data["depth"].min():.4f}, {data["depth"].max():.4f}]')
print(f'Spread range: [{data["spread"].min():.4f}, {data["spread"].max():.4f}]')
print(f'Cov_norm range: [{data["cov_norm"].min():.4f}, {data["cov_norm"].max():.4f}]')

# 用保存的数据原地重新计算 R²
dm = data["depth"]
sm = data["spread"]
cm = data["cov_norm"]
d2m = dm ** 2

a_val = model.depth_spread_a.item()
b_val = model.depth_spread_b.item()
c_val = model.depth_spread_c.item()
d_val = model.depth_spread_d.item()

print(f'\nPrior params at eval time: a={a_val:.6f}, b={b_val:.6f}, c={c_val:.6f}, d={d_val:.6f}')

y_pred_s = a_val * dm + c_val
y_pred_c = b_val * d2m + d_val

# Huber R²
def huber_r2(y_true, y_pred, delta_scale=1.0):
    res = np.abs(y_true - y_pred)
    delta = np.median(res) * delta_scale
    huber_res = np.where(res <= delta, 0.5 * res ** 2, delta * (res - 0.5 * delta))
    huber_var = np.where(np.abs(y_true - y_true.mean()) <= delta,
                         0.5 * (y_true - y_true.mean()) ** 2,
                         delta * (np.abs(y_true - y_true.mean()) - 0.5 * delta))
    return 1.0 - huber_res.sum() / max(huber_var.sum(), 1e-12)

r2_s_manual = huber_r2(sm, y_pred_s, delta_scale=1.5)
r2_c_manual = huber_r2(cm, y_pred_c, delta_scale=1.5)
print(f'Manual Huber R²: spread={r2_s_manual:.4f}, cov={r2_c_manual:.4f}')

# 传统 R²
r2_s_trad = 1 - np.sum((sm - y_pred_s)**2) / max(np.sum((sm - sm.mean())**2), 1e-12)
r2_c_trad = 1 - np.sum((cm - y_pred_c)**2) / max(np.sum((cm - cm.mean())**2), 1e-12)
print(f'Traditional R²: spread={r2_s_trad:.4f}, cov={r2_c_trad:.4f}')

# 也用 polyfit 来验证
coef_s = np.polyfit(dm, sm, 1)
y_pred_s_poly = np.polyval(coef_s, dm)
r2_s_poly = 1 - np.sum((sm - y_pred_s_poly)**2) / max(np.sum((sm - sm.mean())**2), 1e-12)
print(f'Polyfit (spread): slope={coef_s[0]:.4f}, intercept={coef_s[1]:.4f}, R²={r2_s_poly:.4f}')

coef_c = np.polyfit(d2m, cm, 1)
y_pred_c_poly = np.polyval(coef_c, d2m)
r2_c_poly = 1 - np.sum((cm - y_pred_c_poly)**2) / max(np.sum((cm - cm.mean())**2), 1e-12)
print(f'Polyfit (cov): slope={coef_c[0]:.4f}, intercept={coef_c[1]:.4f}, R²={r2_c_poly:.4f}')

# 保存完整数据到文件供后续分析
np.savez('/tmp/eval_debug_full.npz',
         depth=dm, spread=sm, cov_norm=cm, depth2=d2m,
         a=a_val, b=b_val, c=c_val, d=d_val,
         y_pred_s=y_pred_s, y_pred_c=y_pred_c,
         r2_spread_huber=r2_s_manual, r2_cov_huber=r2_c_manual,
         r2_spread_trad=r2_s_trad, r2_cov_trad=r2_c_trad,
         r2_spread_poly=r2_s_poly, r2_cov_poly=r2_c_poly,
         coef_s=coef_s, coef_c=coef_c,
         n_points=len(dm))
print(f'\nAll data saved to /tmp/eval_debug_full.npz')
print(f'Total FG points: {len(dm)}')
