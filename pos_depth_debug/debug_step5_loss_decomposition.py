"""
实验5: 深入分析 slot loss 的计算细节
特别关注: freeze_appearance 时 dyn 和 app loss 的比例，
以及 loss 是否真的在驱动正确的方向
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset

with open('/autodl-fs/data/SlotHWM/config/obj3d.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)

model = SlotDynamicsModel(cfg).cuda()
ckpt = torch.load(cfg.pretrained_path, map_location='cpu')
model_state = model.state_dict()
loaded = {}
for mk in model_state:
    mk_c = mk.replace('_orig_mod.', '')
    for ck in ckpt['model']:
        ck_c = ck.replace('_orig_mod.', '')
        if ck_c == mk_c and ckpt['model'][ck].shape == model_state[mk].shape:
            loaded[mk] = ckpt['model'][ck]
            break
model.load_state_dict(loaded, strict=False)
model.train()

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=4, shuffle=True, num_workers=0)

rollout = cfg.rollout_frames
app_dim = cfg.appearance_dim

# 检查 3 个不同 batch 的 loss 分解
for i in range(3):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    out = model(frames)
    
    pred_S = out['slots']['predicted'][:, :rollout]
    target_S = out['slots']['target'][:, :rollout]
    depth_mask = out['depth_mask'][:, :rollout]
    mask = depth_mask.unsqueeze(-1).float()
    
    pred_dyn = pred_S[:, :, :, app_dim:]
    target_dyn = target_S[:, :, :, app_dim:]
    pred_app = pred_S[:, :, :, :app_dim]
    target_app = target_S[:, :, :, :app_dim]
    
    # 各维度误差
    per_dim_err = ((pred_dyn - target_dyn)**2 * mask).mean(dim=[0, 1, 2])  # (3,)
    pos_x_err = per_dim_err[0].item()
    pos_y_err = per_dim_err[1].item()
    depth_err = per_dim_err[2].item()
    
    # 各维度的 target 数值范围
    target_dyn_range = target_dyn.abs().mean(dim=[0, 1, 2])
    
    # 相对误差
    rel_err = per_dim_err / (target_dyn_range**2 + 1e-8)
    
    print(f"\n=== Batch {i} ===")
    print(f"  pred_dyn range: [{pred_dyn.min():.4f}, {pred_dyn.max():.4f}]")
    print(f"  target_dyn range: [{target_dyn.min():.4f}, {target_dyn.max():.4f}]")
    print(f"  Per-dim abs error: pos_x={pos_x_err:.6f}, pos_y={pos_y_err:.6f}, depth={depth_err:.6f}")
    print(f"  Target dyn mean:  pos_x={target_dyn_range[0]:.4f}, pos_y={target_dyn_range[1]:.4f}, depth={target_dyn_range[2]:.4f}")
    print(f"  Relative error:   pos_x={rel_err[0]:.4f}, pos_y={rel_err[1]:.4f}, depth={rel_err[2]:.4f}")
    
    # 关键检查: pred 的 Z^d 到底是什么？是不是就是 burnin_last 复制？
    # 检查 rollout 预测的 Z^d 是否帧间不同
    for t in range(min(3, rollout)):
        d = pred_dyn[:, t]
        print(f"  pred frame {t}: mean={d.mean(dim=0).tolist()}")
    
    # 检查: 如果预测完全=target 会怎样？(信息量上界)
    # 也就是 target 自身的方差
    target_dyn_var = target_dyn.var(dim=[0, 1, 2])
    print(f"  Target dyn variance: pos_x={target_dyn_var[0]:.6f}, pos_y={target_dyn_var[1]:.6f}, depth={target_dyn_var[2]:.6f}")
    
    # 检查: 每个 slot 的 depth 在 target 中是否变化?
    for s in range(target_dyn.shape[2]):
        depth_ts = target_dyn[:, :, s, 2]  # (B, T)
        var = depth_ts.var().item()
        print(f"  Slot {s} depth variance across rollout: {var:.8f}")

# 关键问题: 训练时 loss 通过 recon 路径有没有梯度？
# 检查 freeze_appearance 时 recon_rollout_grad 是否真的被 detach 了
print("\n=== Checking recon_loss gradient flow ===")
batch = next(iter(loader))
frames = batch['video'].cuda()
model.zero_grad()
out = model(frames)

# 检查 recon_rollout 的梯度
from train.trainer import Trainer
trainer_cfg = SimpleNamespace(**cfg_dict)
# 直接看 freeze_appearance 的逻辑
recon_pred = out['outputs']['video_pred']
target_rollout = frames[:, cfg.burnin_frames:cfg.burnin_frames+rollout]
recon_loss = F.mse_loss(recon_pred, target_rollout)

# 检查 recon_loss 对 decoder 参数的梯度
recon_loss.backward(retain_graph=True)
decoder_grad = sum(p.grad.norm().item()**2 for p in model.decoder.parameters() if p.grad is not None)
predictor_grad = sum(p.grad.norm().item()**2 for p in model.predictor.parameters() if p.grad is not None)
print(f"  recon_loss → decoder grad: {decoder_grad**0.5:.6f}")
print(f"  recon_loss → predictor grad: {predictor_grad**0.5:.6f}")

# 在 freeze_appearance 模式下，recon_rollout 是否应该给 predictor 梯度？
# trainer.py line 227: recon_rollout_grad = recon_rollout_val.detach()
# 所以 recon 路径不给 predictor 任何梯度！
# 唯一驱动 predictor 学习的是 slot_loss
