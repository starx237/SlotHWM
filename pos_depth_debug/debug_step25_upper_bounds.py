"""
实验25: 最终分析 - 理论上界

计算: 如果模型完美预测 (teacher forcing)，loss 是多少？
这给出了 slot_loss 的下界

然后计算: 不同策略的 free-running loss
1. zero_velocity: 保持 burnin_last 不变
2. target_velocity: 用真实的第一帧 delta 作为速度 (oracle)
3. learned_velocity: 从 buffer 计算的速度

这告诉我们: free-running 的性能瓶颈在哪
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

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=8, shuffle=True, num_workers=0)

# 方法: target_velocity - 用 target 的第一帧 delta (oracle)
# 这是最优的恒速预测: vel = target[0] - burnin_last
total_pos = {m: 0 for m in ['zero_vel', 'oracle_vel', 'buffer_vel']}
total_depth = {m: 0 for m in ['zero_vel', 'oracle_vel', 'buffer_vel']}
n = 0

for i in range(30):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = model(frames)
    
    target = out['slots']['target']  # (B, rollout, N, D)
    corrected = out['slots']['corrected']
    dm = out['depth_mask']
    B = frames.shape[0]
    
    burnin_dyn = corrected[:, :, :, app_dim:]  # (B, burnin, N, 3)
    burnin_last = burnin_dyn[:, -1]
    
    # Oracle velocity: target[0] - burnin_last
    oracle_vel = target[:, 0, :, app_dim:] - burnin_last
    
    # Buffer velocity: burnin[-1] - burnin[-2]
    buffer_vel = burnin_dyn[:, -1] - burnin_dyn[:, -2]
    
    for method in ['zero_vel', 'oracle_vel', 'buffer_vel']:
        if method == 'zero_vel':
            vel = torch.zeros_like(burnin_last)
        elif method == 'oracle_vel':
            vel = oracle_vel
        else:
            vel = buffer_vel
        
        cur = burnin_last.clone()
        for t in range(rollout):
            pred = cur + vel
            target_dyn = target[:, t, :, app_dim:]
            mask_t = dm[:, t].unsqueeze(-1).float()
            
            pos_mse = ((pred[..., :2] - target_dyn[..., :2])**2 * mask_t).sum() / (mask_t.sum()*2+1e-8)
            depth_mse = ((pred[..., 2:3] - target_dyn[..., 2:3])**2 * mask_t).sum() / (mask_t.sum()+1e-8)
            total_pos[method] += pos_mse.item()
            total_depth[method] += depth_mse.item()
            
            cur = pred  # free-running
        n += 1

print("=== Free-running with different velocity strategies ===")
for method in ['zero_vel', 'oracle_vel', 'buffer_vel']:
    print(f"  {method:12s}: pos={total_pos[method]/n:.6f} depth={total_depth[method]/n:.6f}")

# 关键: teacher forcing 下的 oracle（理论上界）
print("\n=== Teacher forcing (upper bound) ===")
total_pos_tf = 0
total_depth_tf = 0
n = 0
for i in range(30):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = model(frames)
    target = out['slots']['target']
    corrected = out['slots']['corrected']
    dm = out['depth_mask']
    burnin_dyn = corrected[:, :, :, app_dim:]
    burnin_last = burnin_dyn[:, -1]
    
    cur = burnin_last
    for t in range(rollout):
        # Teacher forcing: 用真实速度 oracle_vel_t = target[t] - target[t-1]
        if t == 0:
            oracle_vel_t = target[:, 0, :, app_dim:] - burnin_last
        else:
            oracle_vel_t = target[:, t, :, app_dim:] - target[:, t-1, :, app_dim:]
        
        pred = cur + oracle_vel_t  # teacher forcing: 用真实 cur
        target_dyn = target[:, t, :, app_dim:]
        mask_t = dm[:, t].unsqueeze(-1).float()
        
        pos_mse = ((pred[..., :2] - target_dyn[..., :2])**2 * mask_t).sum() / (mask_t.sum()*2+1e-8)
        depth_mse = ((pred[..., 2:3] - target_dyn[..., 2:3])**2 * mask_t).sum() / (mask_t.sum()+1e-8)
        total_pos_tf += pos_mse.item()
        total_depth_tf += depth_mse.item()
        
        cur = target_dyn  # teacher forcing: 用真实值
    n += 1

print(f"  oracle_tf:   pos={total_pos_tf/n:.6f} depth={total_depth_tf/n:.6f}")

# 这应该接近 0 (因为 teacher forcing + oracle 速度 = 几乎完美)
# 如果不是 0，说明恒速假设即使在 teacher forcing 下也有误差
