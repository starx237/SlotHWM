"""
实验20: 核心诊断 - 检查 recon_loss 通过 decoder 能否给 pos/depth 提供梯度

当前问题总结:
1. slot_loss (MSE on Z^d) 可以在单 batch 上收敛
2. 但跨 batch 不稳定 (灾难性遗忘 / 信号太弱)
3. velocity ≠ target_delta，简单的恒速外推不够
4. freeze_appearance=True 时，recon_rollout 的梯度被 detach

关键假设: 如果让 recon_rollout 不 detach（允许梯度流过 decoder → pred_S → pred_Z → predictor），
decoder 可以提供更丰富的梯度信号（基于像素级重建质量，而非原始 slot MSE）

验证: 对比 recon_rollout 有梯度 vs 无梯度时，predictor 的学习效果
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

app_dim = cfg.appearance_dim
rollout = cfg.rollout_frames
burnin = cfg.burnin_frames

# 先检查: decoder 对 pos/depth 的梯度有多大？
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
model.eval()

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
torch.manual_seed(42)
batch = next(iter(ds.get_dataloader(batch_size=4, shuffle=True, num_workers=0)))
frames = batch['video'].cuda()

# 检查 decoder 输出对 pos/depth 的敏感度
with torch.no_grad():
    out = model(frames)

# 取 burnin 最后一帧的 slots
S = out['slots']['corrected'][:, -1]  # (B, N, D)
decoded = model.decoder(S)  # (B, N, 3, H, W)

# 微扰 pos/depth，看 decoded 变化多大
S_perturbed = S.clone()
eps = 0.01
S_perturbed[:, :, app_dim+2] += eps  # perturb depth
decoded_perturbed = model.decoder(S_perturbed)

diff = (decoded - decoded_perturbed).abs().mean().item()
print(f"Depth perturbation eps={eps}: decoded diff = {diff:.8f}")

S_perturbed2 = S.clone()
S_perturbed2[:, :, app_dim] += eps  # perturb pos_x
decoded_perturbed2 = model.decoder(S_perturbed2)
diff2 = (decoded - decoded_perturbed2).abs().mean().item()
print(f"Pos_x perturbation eps={eps}: decoded diff = {diff2:.8f}")

# 现在计算 recon_loss 对 pos/depth 的梯度
recon = out['outputs']['video_pred'][:, 0]  # (B, 3, H, W) first rollout frame
target_frame = frames[:, burnin]  # (B, 3, H, W)
recon_loss = F.mse_loss(recon, target_frame)

model.zero_grad()
recon_loss.backward()

# 检查 pred_S 的 pos/depth 梯度
pred_S = out['slots']['predicted']  # has grad_fn
# 但 pred_S 是从 pred_Z 构建的，中间有 detach 吗？

# 检查 predictor 参数的梯度
for name, param in model.predictor.named_parameters():
    if param.grad is not None:
        gn = param.grad.norm().item()
        if gn > 0 and 'spatiotemporal' in name:
            print(f"  predictor.{name}: grad={gn:.6f}")

predictor_grad_from_recon = sum(
    p.grad.norm().item()**2 for p in model.predictor.parameters() if p.grad is not None
)**0.5
print(f"\nTotal predictor grad from recon_rollout: {predictor_grad_from_recon:.6f}")

# 对比: slot_loss 给 predictor 的梯度
model.zero_grad()
pred_slots = out['slots']['predicted'][:, :rollout]
target_slots = out['slots']['target'][:, :rollout]
depth_mask = out['depth_mask'][:, :rollout]
mask = depth_mask.unsqueeze(-1).float()

# 只用 Z^d 的 MSE
pred_dyn = pred_slots[:, :, :, app_dim:]
target_dyn = target_slots[:, :, :, app_dim:]
slot_loss = ((pred_dyn - target_dyn)**2 * mask).sum() / (mask.sum() * pred_dyn.shape[-1] + 1e-8)

slot_loss.backward()

predictor_grad_from_slot = sum(
    p.grad.norm().item()**2 for p in model.predictor.parameters() if p.grad is not None
)**0.5
print(f"Total predictor grad from slot_loss: {predictor_grad_from_slot:.6f}")

# 关键: 如果 recon 的梯度远大于 slot_loss，那 detach recon 就浪费了大量信号
print(f"\nGrad ratio: recon/slot = {predictor_grad_from_recon / (predictor_grad_from_slot + 1e-8):.2f}")
