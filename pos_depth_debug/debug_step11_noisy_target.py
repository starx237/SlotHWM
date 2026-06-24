"""
实验11: 核心排查 - 是否 optimizer 状态被 appearance loss (detach) 误导？

在 freeze_appearance 模式下:
- slot_loss = slot_val_dyn + slot_val_app
- slot_val_app 中 pred_app 被 detach，所以对 predictor 没有梯度
- 但 slot_val_app 的值仍然贡献到 total_loss 中
- total_loss 被 backward，optimizer 会为没有梯度的参数做什么？

答案: 不会做错什么。Adam 只更新有梯度的参数。detach 部分不影响梯度计算。

但等等！让我重新检查代码:
losses.py line 64-68:
  pred_app = pred_slots[:, :, :, :app_dim].detach()
  target_app = target_slots[:, :, :, :app_dim]
  slot_val_app = F.mse_loss(pred_app, target_app)
  slot_val = slot_val_dyn + slot_val_app

slot_val 被乘以 lambda_slots 并加到 total_grad 中。
backward 时，slot_val_dyn 有梯度到 predictor，slot_val_app 的 pred_app 被 detach 
所以只回传到 target_app（没有参数需要更新）。

这应该是正确的。slot_val_app 不影响 predictor 的梯度。

那问题出在哪？让我回到更根本的问题：
为什么训练 500 步，单帧 loss 都在震荡？

一个新假设: **target 在 train 模式下不稳定！**
因为 ISA 有 dropout，每次 forward 的 target 都不同。
即使同一个 batch，两次 forward 的 target 差异很大 (max diff 0.85!)。

这意味着: 模型学到的 "正确预测" 在下一个 batch 可能就变成了 "错误预测"。
因为 target 本身在变！

验证: 用 eval 模式计算 target（稳定），然后训练 predictor
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

# 方案A: 正常训练（target 有 dropout 噪声）
model_A = SlotDynamicsModel(cfg).cuda()
ckpt = torch.load(cfg.pretrained_path, map_location='cpu')
model_state = model_A.state_dict()
loaded = {}
for mk in model_state:
    mk_c = mk.replace('_orig_mod.', '')
    for ck in ckpt['model']:
        ck_c = ck.replace('_orig_mod.', '')
        if ck_c == mk_c and ckpt['model'][ck].shape == model_state[mk].shape:
            loaded[mk] = ckpt['model'][ck]
            break
model_A.load_state_dict(loaded, strict=False)
for name, param in model_A.named_parameters():
    if 'spatiotemporal' not in name:
        param.requires_grad = False

# 方案B: eval 模式获取 target，然后 train predictor
model_B = SlotDynamicsModel(cfg).cuda()
model_B.load_state_dict(loaded, strict=False)
for name, param in model_B.named_parameters():
    if 'spatiotemporal' not in name:
        param.requires_grad = False

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=8, shuffle=True, num_workers=0)

opt_A = torch.optim.Adam(filter(lambda p: p.requires_grad, model_A.parameters()), lr=1e-4)
opt_B = torch.optim.Adam(filter(lambda p: p.requires_grad, model_B.parameters()), lr=1e-4)

print("=== A: Normal train (target with dropout) vs B: Eval target ===")
for step in range(300):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    
    # A: normal
    model_A.train()
    out_A = model_A(frames)
    pred_A = out_A['slots']['predicted'][:, :rollout]
    target_A = out_A['slots']['target'][:, :rollout]
    mask_A = out_A['depth_mask'][:, :rollout].unsqueeze(-1).float()
    pred_dyn_A = pred_A[:, :, :, app_dim:]
    target_dyn_A = target_A[:, :, :, app_dim:]
    loss_A = ((pred_dyn_A - target_dyn_A)**2 * mask_A).sum() / (mask_A.sum() * pred_dyn_A.shape[-1] + 1e-8)
    opt_A.zero_grad()
    loss_A.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model_A.parameters()), 1.0)
    opt_A.step()
    
    # B: eval target
    model_B.eval()
    with torch.no_grad():
        out_B_eval = model_B(frames)
        target_B = out_B_eval['slots']['target'][:, :rollout]
        mask_B = out_B_eval['depth_mask'][:, :rollout].unsqueeze(-1).float()
    
    model_B.train()  # predictor in train mode for dropout
    out_B = model_B(frames)
    pred_B = out_B['slots']['predicted'][:, :rollout]
    pred_dyn_B = pred_B[:, :, :, app_dim:]
    target_dyn_B = target_B[:, :, :, app_dim:]
    loss_B = ((pred_dyn_B - target_dyn_B)**2 * mask_B).sum() / (mask_B.sum() * pred_dyn_B.shape[-1] + 1e-8)
    opt_B.zero_grad()
    loss_B.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model_B.parameters()), 1.0)
    opt_B.step()
    
    if step % 30 == 0:
        print(f"step {step:3d}: A_loss={loss_A.item():.6f} B_loss={loss_B.item():.6f}")

# 最终评估
print("\n=== Final evaluation ===")
model_A.eval()
model_B.eval()
torch.manual_seed(999)
batch = next(iter(ds.get_dataloader(batch_size=8, shuffle=True, num_workers=0)))
frames = batch['video'].cuda()

with torch.no_grad():
    out_A = model_A(frames)
    out_B = model_B(frames)

for name, mdl in [("A (noisy target)", model_A), ("B (stable target)", model_B)]:
    with torch.no_grad():
        out = mdl(frames)
    pred_dyn = out['slots']['predicted'][:, :rollout, :, app_dim:]
    target_dyn = out['slots']['target'][:, :rollout, :, app_dim:]
    mask = out['depth_mask'][:, :rollout].unsqueeze(-1).float()
    
    pos_mse = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask).sum() / (mask.sum()*2+1e-8)
    depth_mse = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask).sum() / (mask.sum()+1e-8)
    
    print(f"  {name}: pos_mse={pos_mse.item():.6f}, depth_mse={depth_mse.item():.6f}")
