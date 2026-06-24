"""
实验17: 检查 decoder 是否通过 recon_loss 给 pos/depth 提供梯度

关键路径: recon_burnin → decoder → burnin_S → slots
burnin_S 包含 pos/depth，但 encoder/slot_attention/decoder 都冻结了
所以 recon_burnin 不会给 predictor 梯度

recon_rollout → decoder → pred_S → pred_Z → predictor
但 freeze_appearance=True 时，recon_rollout_grad = recon_rollout_val.detach()
所以这条路径也被切断了！

结论: predictor 唯一的梯度来源是 slot_loss（直接 MSE）。
recon_loss 完全不参与 predictor 的训练。

这可能是问题所在: 没有 recon_loss 的梯度信号，
predictor 无法从 "重建质量" 获得反馈。
它只能从 "slot 值的 MSE" 获得反馈，
而 slot 值的 MSE 可能不够准确（因为 target 本身有 dropout 噪声）。

但更重要的问题是: 为什么 slot_loss 的 MSE 无法驱动学习？
让我做一个更精确的实验: 用固定的 target (eval 模式)，连续优化同一 batch
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

for name, param in model.named_parameters():
    if 'spatiotemporal' not in name:
        param.requires_grad = False

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# 固定同一个 batch，反复训练
torch.manual_seed(42)
batch = next(iter(ds.get_dataloader(batch_size=8, shuffle=False, num_workers=0)))
frames = batch['video'].cuda()

# 用 eval 模式获取稳定的 target
model.eval()
with torch.no_grad():
    out = model(frames)
target_S = out['slots']['target'][:, :rollout]
depth_mask = out['depth_mask'][:, :rollout]

print("=== Training on SAME batch (eval target, no noise) ===")
for step in range(500):
    model.train()
    out = model(frames)
    pred_S = out['slots']['predicted'][:, :rollout]
    
    mask = depth_mask.unsqueeze(-1).float()
    pred_dyn = pred_S[:, :, :, app_dim:]
    target_dyn = target_S[:, :, :, app_dim:]
    
    dyn_loss = ((pred_dyn - target_dyn)**2 * mask).sum() / (mask.sum() * pred_dyn.shape[-1] + 1e-8)
    
    optimizer.zero_grad()
    dyn_loss.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
    optimizer.step()
    
    if step % 50 == 0:
        with torch.no_grad():
            pos_mse = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask).sum() / (mask.sum()*2+1e-8)
            depth_mse = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask).sum() / (mask.sum()+1e-8)
        print(f"step {step:3d}: dyn={dyn_loss.item():.6f} pos={pos_mse.item():.6f} depth={depth_mse.item():.6f}")

# 用训练好的模型评估不同 batch
print("\n=== Cross-batch evaluation ===")
model.eval()
torch.manual_seed(999)
batch2 = next(iter(ds.get_dataloader(batch_size=8, shuffle=True, num_workers=0)))
frames2 = batch2['video'].cuda()
with torch.no_grad():
    out2 = model(frames2)
pred_dyn2 = out2['slots']['predicted'][:, :rollout, :, app_dim:]
target_dyn2 = out2['slots']['target'][:, :rollout, :, app_dim:]
mask2 = out2['depth_mask'][:, :rollout].unsqueeze(-1).float()
pos_mse2 = ((pred_dyn2[..., :2] - target_dyn2[..., :2])**2 * mask2).sum() / (mask2.sum()*2+1e-8)
depth_mse2 = ((pred_dyn2[..., 2:3] - target_dyn2[..., 2:3])**2 * mask2).sum() / (mask2.sum()+1e-8)
print(f"New batch: pos_mse={pos_mse2.item():.6f}, depth_mse={depth_mse2.item():.6f}")
