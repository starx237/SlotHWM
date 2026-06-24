"""
实验14: 精确模拟 predictor 的自回归 rollout
用简单的 MLP 逐步预测，和 predictor 完全一样的工作方式

关键区别于实验13:
- 实验13 是独立的 per-frame 预测（每帧都从 burnin buffer 开始）
- 实验14 是自回归的（用上一步的预测作为下一步的输入）
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
model.eval()

# 训练一个简单的 MLP: (cur_Z^d + buffer_Z^d) → delta_Z^d
# 每步自回归: cur_Z^d = cur_Z^d + delta, 然后把新 cur 加入 buffer
N = cfg.num_slots
dyn_dim = cfg.dynamic_dim

class SimplePredictor(nn.Module):
    def __init__(self, buffer_len, N, dyn_dim, hidden=128):
        super().__init__()
        # 输入: 当前帧所有 slot 的 Z^d (N*dyn_dim) + buffer 的 Z^d (buffer_len*N*dyn_dim)
        self.buffer_len = buffer_len
        self.N = N
        self.dyn_dim = dyn_dim
        input_dim = N * dyn_dim + buffer_len * N * dyn_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N * dyn_dim),
        )
        # zero init last layer
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward_step(self, cur_dyn, buffer_dyn):
        """
        cur_dyn: (B, N, dyn_dim)
        buffer_dyn: (B, buffer_len, N, dyn_dim)
        Returns: delta (B, N, dyn_dim)
        """
        B = cur_dyn.shape[0]
        cur_flat = cur_dyn.reshape(B, -1)
        buf_flat = buffer_dyn.reshape(B, -1)
        x = torch.cat([cur_flat, buf_flat], dim=-1)
        delta = self.mlp(x).reshape(B, self.N, self.dyn_dim)
        return delta

# 收集数据
ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=8, shuffle=True, num_workers=0)

# 训练
predictor = SimplePredictor(burnin, N, dyn_dim).cuda()
optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-4)

for step in range(1000):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    
    model.eval()
    with torch.no_grad():
        out = model(frames)
    
    # 获取 burnin Z^d
    corrected = out['slots']['corrected']  # (B, burnin, N, D)
    target = out['slots']['target']  # (B, rollout, N, D)
    depth_mask = out['depth_mask']  # (B, rollout, N)
    
    B = frames.shape[0]
    
    # 自回归 rollout
    buffer_dyn = corrected[:, :, :, app_dim:].clone()  # (B, burnin, N, dyn_dim)
    cur_dyn = corrected[:, -1, :, app_dim:].clone()  # (B, N, dyn_dim)
    
    total_loss = 0
    for t in range(rollout):
        delta = predictor.forward_step(cur_dyn, buffer_dyn)
        next_dyn = cur_dyn + delta  # 自回归
        
        target_dyn_t = target[:, t, :, app_dim:]  # (B, N, dyn_dim)
        mask_t = depth_mask[:, t].unsqueeze(-1).float()  # (B, N, 1)
        
        loss_t = ((next_dyn - target_dyn_t)**2 * mask_t).sum() / (mask_t.sum() * dyn_dim + 1e-8)
        total_loss = total_loss + loss_t
        
        # 更新 buffer 和 cur
        cur_dyn = next_dyn.detach()  # teacher forcing would use target, but we use pred
        buffer_dyn = torch.cat([buffer_dyn[:, 1:], next_dyn.detach().unsqueeze(1)], dim=1)
    
    total_loss = total_loss / rollout
    
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
    optimizer.step()
    
    if step % 100 == 0:
        print(f"step {step:4d}: loss={total_loss.item():.6f}")

# 评估
print("\n=== Evaluation with autoregressive rollout ===")
model.eval()
predictor.eval()
torch.manual_seed(999)
batch = next(iter(ds.get_dataloader(batch_size=4, shuffle=True, num_workers=0)))
frames = batch['video'].cuda()

with torch.no_grad():
    out = model(frames)
    corrected = out['slots']['corrected']
    target = out['slots']['target']
    depth_mask = out['depth_mask']
    B = frames.shape[0]
    
    # 使用训练好的 predictor
    buffer_dyn = corrected[:, :, :, app_dim:].clone()
    cur_dyn = corrected[:, -1, :, app_dim:].clone()
    
    for t in range(rollout):
        delta = predictor.forward_step(cur_dyn, buffer_dyn)
        next_dyn = cur_dyn + delta
        
        mask_t = depth_mask[:, t].unsqueeze(-1).float()
        target_dyn_t = target[:, t, :, app_dim:]
        
        pos_mse = ((next_dyn[..., :2] - target_dyn_t[..., :2])**2 * mask_t).sum() / (mask_t.sum()*2+1e-8)
        depth_mse = ((next_dyn[..., 2:3] - target_dyn_t[..., 2:3])**2 * mask_t).sum() / (mask_t.sum()+1e-8)
        
        if t in [0, 4, 9]:
            print(f"  frame {t}: pos_mse={pos_mse.item():.6f}, depth_mse={depth_mse.item():.6f}")
        
        cur_dyn = next_dyn
        buffer_dyn = torch.cat([buffer_dyn[:, 1:], next_dyn.unsqueeze(1)], dim=1)
