"""
实验10: 核心问题 - 从 z_buffer (历史 Z^d) 能否推断运动方向？

z_buffer 包含过去几帧的 Z^d (pos, depth)。如果 burnin=6，buffer_len=16，
那 z_buffer 最多有 burnin + rollout 帧的历史。

从历史 pos 可以计算速度。问题是：时空模块能否学到这个？

更深的问题: 目前的 slot_loss 计算方式是否正确？
slot_loss = MSE(pred_S, target_S)，其中 pred_S 经过 f_z.inverse 变换回 S 空间
但 Z^d 只有3维，pred_S 的 Z^d 和 target_S 的 Z^d 是在 Z 空间比较还是在 S 空间？

让我检查: pred_S 和 target_S 的 Z^d 到底在哪里计算？
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')

import torch
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
model.eval()

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
torch.manual_seed(42)
batch = next(iter(ds.get_dataloader(batch_size=4, shuffle=True, num_workers=0)))
frames = batch['video'].cuda()

with torch.no_grad():
    out = model(frames)

app_dim = cfg.appearance_dim

# 关键: 检查 pred_S 和 target_S 的关系
# dynamics.py line 380-385:
# Z_appearance = pred_Z[:, t, :, :appearance_dim]
# pos_depth = pred_Z[:, t, :, appearance_dim:]
# S_raw = f_z.inverse(Z_appearance)
# S = cat([S_raw, pos_depth], dim=-1)
# 所以 pred_S 的前 64 维 = f_z.inverse(Z_appearance)，后 3 维 = pred_Z 的 Z^d

# 而 target_S 是直接从 ISA slot attention 出来的，后 3 维 = ISA 的 pos/depth

# 问题: pred_Z 的 Z^d 和 ISA 的 pos/depth 是否在同一个空间？
# pred_Z 的 Z^d 是从 predictor 输出的，predictor 输入的 Z^d 来自 burnin 的 Z
# burnin 的 Z 是通过 f_z 变换得到的: Z_core = f_z(slots[:, :, :appearance_dim])
# Z_full = cat([Z_core, slots[:, :, -3:]], dim=-1)
# 所以 Z^d = slots[:, :, -3:] = ISA 直接输出的 (pos_x, pos_y, depth)

# 结论: Z^d 直接是 ISA 输出的 (pos_x, pos_y, depth)，没有经过任何变换
# 所以 pred_S 的 pos/depth 和 target_S 的 pos/depth 在同一个空间
# 这排除了空间不匹配的问题

# 那现在关键问题: slot_loss 在 S 空间计算还是在 Z 空间计算？
# losses.py: pred_slots 是 pred_S，target_slots 是 target_S
# 是在 S 空间计算 MSE

# 但等一下！losses.py line 60-61:
# pred_dyn = pred_slots[:, :, :, app_dim:]  → 这是 pred_S 的后 3 维 = pos/depth
# target_dyn = target_slots[:, :, :, app_dim:]  → 这是 target_S 的后 3 维 = pos/depth
# 所以 pos/depth 的 loss 是在 S 空间（=ISA 原始空间）计算的，这是对的

# 现在让我检查一个更根本的问题: 
# pred_S 的 appearance 部分 (前 64 维) 是 f_z.inverse(Z_appearance)
# 在 freeze_appearance 模式下，pred_app = pred_S[:, :, :, :app_dim].detach()
# 也就是 appearance 部分的梯度被 detach 了
# 但 pred_app 和 target_app 的 MSE 仍然被计算并加到 slot_loss 中
# 这意味着 appearance 部分的误差会增加 slot_loss 的值，但不提供梯度

# 关键检查: slot_loss_app 的值有多大？如果比 slot_loss_dyn 大很多，
# 那 slot_loss 的主要"压力"来自 appearance，但梯度只来自 dyn

# 让我看看完整的 loss 分解
print("=== Full loss decomposition ===")
for b in range(1):
    pred_S = out['slots']['predicted']
    target_S = out['slots']['target']
    
    # 按维度分解
    for t in range(min(3, pred_S.shape[1])):
        p = pred_S[0, t]
        tg = target_S[0, t]
        
        app_err = ((p[:, :app_dim] - tg[:, :app_dim])**2).mean().item()
        pos_err = ((p[:, app_dim:app_dim+2] - tg[:, app_dim:app_dim+2])**2).mean().item()
        depth_err = ((p[:, app_dim+2:] - tg[:, app_dim+2:])**2).mean().item()
        
        print(f"  frame {t}: app_mse={app_err:.6f}, pos_mse={pos_err:.6f}, depth_mse={depth_err:.6f}")

# 现在关键: 检查 appearance 部分是否有误差
# freeze_C=True 时，pred 的 appearance 来自 C (global_C = C_time_attn 输出)
# target 的 appearance 来自 ISA 对 rollout 帧的直接编码
# 这两个应该是不同的！因为 ISA 每帧重新编码 appearance
# 而 C 是从 burnin 帧的 appearance 聚合的

print("\n=== Appearance consistency: C vs target appearance ===")
for b in range(1):
    for t in range(min(3, pred_S.shape[1])):
        pred_app = pred_S[0, t, :, :app_dim]
        target_app = target_S[0, t, :, :app_dim]
        
        # cosine similarity
        p_norm = pred_app / (pred_app.norm(dim=-1, keepdim=True) + 1e-8)
        t_norm = target_app / (target_app.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (p_norm * t_norm).sum(dim=-1)
        
        print(f"  frame {t}: cos_sim per slot = {[f'{c:.3f}' for c in cos.tolist()]}")

# 核心问题: 检查 slot_loss 中 appearance 误差 vs dyn 误差的比例
from train.losses import SlotPiLoss
loss_fn = SlotPiLoss(cfg)

pred_slots = out['slots']['predicted'][:, :cfg.rollout_frames]
target_slots = out['slots']['target'][:, :cfg.rollout_frames]
depth_mask = out['depth_mask'][:, :cfg.rollout_frames]

total_loss, aux = loss_fn(pred_slots, target_slots, depth_mask=depth_mask)

print(f"\n=== Loss components ===")
for k, v in aux.items():
    print(f"  {k}: {v:.6f}")
print(f"\n  slot_loss_dyn / slot_loss_app = {aux['slot_loss_dyn'] / (aux['slot_loss_app'] + 1e-8):.2f}")
print(f"  slot_loss_app 值: {aux['slot_loss_app']:.6f} (但没有梯度！)")
print(f"  slot_loss_dyn 值: {aux['slot_loss_dyn']:.6f} (有梯度)")
