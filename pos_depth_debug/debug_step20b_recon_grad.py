"""
实验20b: 直接测量 decoder 对 pos/depth 的梯度
以及比较 recon 信号 vs slot 信号的强度
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

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
torch.manual_seed(42)
batch = next(iter(ds.get_dataloader(batch_size=4, shuffle=True, num_workers=0)))
frames = batch['video'].cuda()

# 1. decoder 对 pos/depth 的 Jacobian 范数
print("=== Decoder Jacobian: d(decoded)/d(pos_depth) ===")
model.eval()
with torch.no_grad():
    out = model(frames)

S = out['slots']['corrected'][:, -1].detach().clone().requires_grad_(True)  # (B, N, D)
decoded = model.decoder(S)  # (B, N, 3, H, W)
# 取某个像素的 loss
loss_dec = decoded.sum()
loss_dec.backward()

# S 的梯度就是 Jacobian 的一种度量
grad_pos = S.grad[:, :, app_dim:app_dim+2].norm().item()
grad_depth = S.grad[:, :, app_dim+2:].norm().item()
grad_app = S.grad[:, :, :app_dim].norm().item()
print(f"  grad w.r.t. appearance: {grad_app:.4f}")
print(f"  grad w.r.t. pos: {grad_pos:.4f}")
print(f"  grad w.r.t. depth: {grad_depth:.4f}")
print(f"  pos+depth / appearance ratio: {(grad_pos+grad_depth)/(grad_app+1e-8):.4f}")

# 2. 直接测量: 给 pred_Z 的 pos/depth 一个微小变化，
#    看 slot_loss 和 recon_loss 各变化多少
print("\n=== Loss sensitivity to pos/depth perturbation ===")
model.eval()
with torch.no_grad():
    out = model(frames)

# 在 pred_Z 的 pos/depth 上加微扰
# 需要重新 forward 来获取 pred_Z
# 但 pred_Z 的计算涉及 predictor，无法直接扰动
# 换一种方式: 扰动 pred_S 的 pos/depth

pred_S_base = out['slots']['predicted'][:, :rollout].clone()
target_S = out['slots']['target'][:, :rollout]
depth_mask = out['depth_mask'][:, :rollout]
mask = depth_mask.unsqueeze(-1).float()

# 基础 slot_loss
pred_dyn_base = pred_S_base[:, :, :, app_dim:]
target_dyn = target_S[:, :, :, app_dim:]
base_slot = ((pred_dyn_base - target_dyn)**2 * mask).sum() / (mask.sum() * 3 + 1e-8)

# 扰动 depth (+0.01)
pred_S_pert = pred_S_base.clone()
pred_S_pert[:, :, :, app_dim+2] += 0.01
pred_dyn_pert = pred_S_pert[:, :, :, app_dim:]
pert_slot = ((pred_dyn_pert - target_dyn)**2 * mask).sum() / (mask.sum() * 3 + 1e-8)

print(f"  slot_loss change (depth +0.01): {base_slot.item():.6f} → {pert_slot.item():.6f}, delta={pert_slot.item()-base_slot.item():.6f}")

# 3. recon_loss 变化
# 需要 decoder，但 decoder 在 model 内部
# 直接用 model.decoder
dec_base = torch.stack([model.decoder(pred_S_base[:, t]) for t in range(rollout)], dim=1)
target_frames = frames[:, burnin:burnin+rollout]
base_recon = F.mse_loss(dec_base, target_frames)

dec_pert = torch.stack([model.decoder(pred_S_pert[:, t]) for t in range(rollout)], dim=1)
pert_recon = F.mse_loss(dec_pert, target_frames)

print(f"  recon_loss change (depth +0.01): {base_recon.item():.6f} → {pert_recon.item():.6f}, delta={pert_recon.item()-base_recon.item():.6f}")

# 4. 关键: slot_loss 的梯度给 predictor 多少信息？
# slot_loss 直接是 pred_dyn vs target_dyn 的 MSE
# 它的梯度方向是: 把 pred_dyn 往 target_dyn 推
# 这和 "最小化重建误差" 可能不一致！
# 因为 pos/depth 在 decoder 中的影响可能是非线性的

# 让我检查: 如果 pred_dyn 的 depth 正好等于 target_dyn，
# 但 appearance 有误差，重建质量如何？
print("\n=== Does matching target Z^d guarantee good reconstruction? ===")
# 构造一个 "完美 Z^d 但有 appearance 误差" 的 pred_S
pred_S_perfect_dyn = target_S.clone()  # 完美匹配 target（包括 appearance）
# 但 freeze_C 模式下 appearance 来自 C，不来自 target

# 实际上让我检查: 真正的 recon_loss 和 slot_loss 之间的相关性
# 如果 slot_loss 低但 recon_loss 高，说明 Z^d 匹配 ≠ 重建质量
