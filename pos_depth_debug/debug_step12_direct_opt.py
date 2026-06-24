"""
实验12: 关键测试 - 给时空模块加入 appearance 信息 (C) 作为额外输入
如果加上 C 后能收敛，说明问题确实是 Z^d 信息不足

当前: spatiotemporal_module 的 embed_dim = dyn_total_dim = 3
测试: 将 embed_dim 改为 slot_dim = 67（包含 appearance），或者把 C 拼接到 Z^d

为了不修改主代码，我在这里临时构建一个简单的 MLP 来测试
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

# 构建一个简单的测试: 手动用 optimizer 直接优化 pred_Z
# 跳过 predictor，直接看 "给定 burnin_Z，最优的 pred_Z 是什么"
# 如果优化 pred_Z 本身也震荡，说明 target 有问题
# 如果优化 pred_Z 能收敛，说明 predictor 的信息瓶颈是问题

# 方案: 直接优化 pred_Z 使得 slot_loss 最小（绕过 predictor）
print("=== Direct optimization of pred_Z (bypassing predictor) ===")
ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
torch.manual_seed(42)
batch = next(iter(ds.get_dataloader(batch_size=4, shuffle=True, num_workers=0)))
frames = batch['video'].cuda()

model.eval()
with torch.no_grad():
    out = model(frames)

# 获取 target
target_S = out['slots']['target'][:, :rollout]  # (B, T, N, D)
depth_mask = out['depth_mask'][:, :rollout]
burnin_last_S = out['slots']['corrected'][:, -1]  # (B, N, D)

# 初始化 pred_Z 为 burnin_last 的复制
B, T, N, D = target_S.shape
pred_S_param = nn.Parameter(burnin_last_S.unsqueeze(1).expand(-1, T, -1, -1).clone())

opt = torch.optim.Adam([pred_S_param], lr=1e-3)

for step in range(1000):
    mask = depth_mask.unsqueeze(-1).float()
    pred_dyn = pred_S_param[:, :, :, app_dim:]
    target_dyn = target_S[:, :, :, app_dim:]
    
    dyn_loss = ((pred_dyn - target_dyn)**2 * mask).sum() / (mask.sum() * pred_dyn.shape[-1] + 1e-8)
    
    opt.zero_grad()
    dyn_loss.backward()
    opt.step()
    
    if step % 100 == 0:
        pos_mse = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask).sum() / (mask.sum()*2+1e-8)
        depth_mse = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask).sum() / (mask.sum()+1e-8)
        print(f"step {step:4d}: dyn_loss={dyn_loss.item():.6f} pos={pos_mse.item():.6f} depth={depth_mse.item():.6f}")

# 最终结果
print("\n=== Direct optimization result ===")
pred_dyn = pred_S_param[:, :, :, app_dim:]
target_dyn = target_S[:, :, :, app_dim:]
mask = depth_mask.unsqueeze(-1).float()
pos_mse = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask).sum() / (mask.sum()*2+1e-8)
depth_mse = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask).sum() / (mask.sum()+1e-8)
print(f"Final: pos_mse={pos_mse.item():.6f}, depth_mse={depth_mse.item():.6f}")

# 逐帧看
for t in range(rollout):
    p = pred_S_param[0, t, :, app_dim:]
    tg = target_S[0, t, :, app_dim:]
    print(f"  frame {t}: pred=[{p[0,0]:.4f},{p[0,1]:.4f},{p[0,2]:.4f}] target=[{tg[0,0]:.4f},{tg[0,1]:.4f},{tg[0,2]:.4f}]")

# 这告诉我们: 如果 pred_S 可以独立优化每帧，target 是否可以被精确匹配？
# 如果能，说明问题在 predictor 的信息瓶颈
# 如果不能（因为 target 需要满足 rollout 自回归约束），说明问题在训练方式
