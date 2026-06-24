"""
实验13: 核心问题确认 - predictor 能否从 Z^d + buffer 推断运动？

从 buffer 中可以计算 pos 的变化（=速度）。问题是时空模块能否学到这一点。

让我构建一个更简单的测试: 用一个简单的 MLP 直接从
(burnin_Z_buffer的Z^d) → (rollout每帧的Z^d delta) 来训练

如果连这个简单任务都学不到，说明 Z^d 空间本身就有问题
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
rollout_frames = cfg.rollout_frames
burnin_frames = cfg.burnin_frames
static_dim = cfg.static_dim
dyn_dim = cfg.dynamic_dim

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

# 收集训练数据: (burnin_buffer_Z^d, cur_Z^d) → next_Z^d_delta
ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=8, shuffle=True, num_workers=0)

all_buffer_dyn = []
all_cur_dyn = []
all_next_dyn = []
all_C = []

for i in range(50):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = model(frames)
    
    # burnin_Z: (B, burnin, N, D)
    burnin_Z = torch.stack([model.f_z(out['slots']['corrected'][:, t, :, :app_dim]) 
                            for t in range(burnin_frames)], dim=1)
    burnin_Z = torch.cat([burnin_Z, out['slots']['corrected'][:, :, :, app_dim:]], dim=-1)
    
    # target_Z: 需要重新计算
    # 实际上 out['slots']['target'] 已经是 S 空间了，不是 Z 空间
    # 让我直接用 S 空间的 dyn 部分
    
    target_S = out['slots']['target']  # (B, rollout, N, D)
    corrected_S = out['slots']['corrected']  # (B, burnin, N, D)
    
    # buffer = burnin frames
    B = frames.shape[0]
    N = corrected_S.shape[2]
    
    for b in range(B):
        # burnin 的 Z^d (每帧)
        for t in range(rollout_frames):
            if t == 0:
                cur_dyn = corrected_S[b, -1, :, app_dim:]  # (N, 3)
            else:
                cur_dyn = target_S[b, t-1, :, app_dim:]
            next_dyn = target_S[b, t, :, app_dim:]  # (N, 3)
            delta = next_dyn - cur_dyn
            
            # burnin buffer 的 Z^d
            buffer_dyn = corrected_S[b, :, :, app_dim:]  # (burnin, N, 3)
            
            # C (appearance)
            # 在 freeze_C 模式下，C = global_C
            C = out.get('S_c')
            if C is not None:
                C_val = C[b, -1, :, :static_dim]  # (N, static_dim)
            else:
                C_val = torch.zeros(N, static_dim, device=frames.device)
            
            all_buffer_dyn.append(buffer_dyn.cpu())
            all_cur_dyn.append(cur_dyn.cpu())
            all_next_dyn.append(delta.cpu())
            all_C.append(C_val.cpu())

print(f"Collected {len(all_buffer_dyn)} samples")

# 简单 MLP: (buffer_dyn_flat + cur_dyn + C) → delta_dyn
buffer_len = burnin_frames
input_dim = buffer_len * N * dyn_dim + N * dyn_dim + N * static_dim
output_dim = N * dyn_dim

print(f"Input dim: {input_dim} (buffer={buffer_len*N*dyn_dim} + cur={N*dyn_dim} + C={N*static_dim})")
print(f"Output dim: {output_dim}")

# 方案1: 只用 buffer_dyn + cur_dyn (无 C)
input_dim_noC = buffer_len * N * dyn_dim + N * dyn_dim
mlp_noC = nn.Sequential(
    nn.Linear(input_dim_noC, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, output_dim),
).cuda()

# 方案2: 用 buffer_dyn + cur_dyn + C
mlp_withC = nn.Sequential(
    nn.Linear(input_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, output_dim),
).cuda()

opt_noC = torch.optim.Adam(mlp_noC.parameters(), lr=1e-3)
opt_withC = torch.optim.Adam(mlp_withC.parameters(), lr=1e-3)

# 训练
for step in range(500):
    # 随机选 32 个样本
    idx = torch.randint(0, len(all_buffer_dyn), (32,))
    
    buffer_batch = torch.stack([all_buffer_dyn[i] for i in idx]).cuda()  # (32, burnin, N, 3)
    cur_batch = torch.stack([all_cur_dyn[i] for i in idx]).cuda()  # (32, N, 3)
    delta_batch = torch.stack([all_next_dyn[i] for i in idx]).cuda()  # (32, N, 3)
    C_batch = torch.stack([all_C[i] for i in idx]).cuda()  # (32, N, static_dim)
    
    # flatten
    buf_flat = buffer_batch.reshape(32, -1)
    cur_flat = cur_batch.reshape(32, -1)
    C_flat = C_batch.reshape(32, -1)
    delta_flat = delta_batch.reshape(32, -1)
    
    # noC
    input_noC = torch.cat([buf_flat, cur_flat], dim=-1)
    pred_noC = mlp_noC(input_noC)
    loss_noC = F.mse_loss(pred_noC, delta_flat)
    opt_noC.zero_grad()
    loss_noC.backward()
    opt_noC.step()
    
    # withC
    input_withC = torch.cat([buf_flat, cur_flat, C_flat], dim=-1)
    pred_withC = mlp_withC(input_withC)
    loss_withC = F.mse_loss(pred_withC, delta_flat)
    opt_withC.zero_grad()
    loss_withC.backward()
    opt_withC.step()
    
    if step % 50 == 0:
        print(f"step {step:3d}: noC_loss={loss_noC.item():.6f} withC_loss={loss_withC.item():.6f} ratio={loss_noC.item()/(loss_withC.item()+1e-8):.2f}")
