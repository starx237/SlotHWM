"""
实验24: 确认 free-running 下 zero-velocity vs constant-velocity
以及理解为什么模型输出比 zero-velocity 差
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

# Free-running 对比
methods = {
    'zero_velocity': lambda cur, prev, t: cur,  # 复制当前帧
    'constant_vel': lambda cur, prev, t: cur + (cur - prev),
}

for name, predict_fn in methods.items():
    total_pos = 0
    total_depth = 0
    per_frame_pos = [0]*rollout
    per_frame_depth = [0]*rollout
    n = 0
    
    for i in range(30):
        batch = next(iter(loader))
        frames = batch['video'].cuda()
        with torch.no_grad():
            out = model(frames)
        
        target = out['slots']['target']
        corrected = out['slots']['corrected']
        dm = out['depth_mask']
        B = frames.shape[0]
        
        burnin_dyn = corrected[:, :, :, app_dim:]
        cur = burnin_dyn[:, -1].clone()
        prev = burnin_dyn[:, -2].clone()
        
        for t in range(rollout):
            pred = predict_fn(cur, prev, t)
            target_dyn = target[:, t, :, app_dim:]
            mask_t = dm[:, t].unsqueeze(-1).float()
            
            pos_mse = ((pred[..., :2] - target_dyn[..., :2])**2 * mask_t).sum() / (mask_t.sum()*2+1e-8)
            depth_mse = ((pred[..., 2:3] - target_dyn[..., 2:3])**2 * mask_t).sum() / (mask_t.sum()+1e-8)
            total_pos += pos_mse.item()
            total_depth += depth_mse.item()
            per_frame_pos[t] += pos_mse.item()
            per_frame_depth[t] += depth_mse.item()
            n += 1
            
            # free-running: 用预测值更新
            prev = cur.clone()
            cur = pred.clone()
    
    print(f"\n{name}: avg_pos={total_pos/n:.6f} avg_depth={total_depth/n:.6f}")
    print(f"  Per-frame pos:  {['%.4f' % (p/30) for p in per_frame_pos]}")
    print(f"  Per-frame depth:{['%.4f' % (d/30) for d in per_frame_depth]}")

# 模型 baseline (free-running, zero-init)
total_pos = 0
total_depth = 0
per_frame_pos = [0]*rollout
per_frame_depth = [0]*rollout
n = 0

for i in range(30):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = model(frames)
    pred_dyn = out['slots']['predicted'][:, :, :, app_dim:]
    target_dyn = out['slots']['target'][:, :, :, app_dim:]
    mask = out['depth_mask'].unsqueeze(-1).float()
    
    for t in range(rollout):
        pos_mse = ((pred_dyn[:, t, :, :2] - target_dyn[:, t, :, :2])**2 * mask[:, t]).sum() / (mask[:, t].sum()*2+1e-8)
        depth_mse = ((pred_dyn[:, t, :, 2:3] - target_dyn[:, t, :, 2:3])**2 * mask[:, t]).sum() / (mask[:, t].sum()+1e-8)
        total_pos += pos_mse.item()
        total_depth += depth_mse.item()
        per_frame_pos[t] += pos_mse.item()
        per_frame_depth[t] += depth_mse.item()
        n += 1

print(f"\nmodel_baseline: avg_pos={total_pos/n:.6f} avg_depth={total_depth/n:.6f}")
print(f"  Per-frame pos:  {['%.4f' % (p/30) for p in per_frame_pos]}")
print(f"  Per-frame depth:{['%.4f' % (d/30) for d in per_frame_depth]}")

# 核心: 模型 baseline (zero-init) = zero_velocity?
# 如果相等，说明模型初始时输出就是 burnin_last 的复制
# 如果不等，说明模型的 rollout 有累积误差
