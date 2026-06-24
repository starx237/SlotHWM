"""
实验3: 在 debug 环境中做小规模训练，观察 loss 走势
关键：只训练 predictor（spatiotemporal 模块），其他冻结
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
from train.losses import SlotPiLoss

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

# 只训练 predictor
for name, param in model.named_parameters():
    if 'predictor' not in name:
        param.requires_grad = False

# 确认哪些 predictor 参数有梯度
trainable = sum(1 for p in model.predictor.parameters() if p.requires_grad)
total = sum(1 for p in model.predictor.parameters())
print(f"Predictor: {trainable}/{total} params trainable")

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
loss_fn = SlotPiLoss(cfg)

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=4, shuffle=True, num_workers=0)

rollout = cfg.rollout_frames
app_dim = cfg.appearance_dim

for step in range(200):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    
    model.train()
    out = model(frames)
    
    pred_S = out['slots']['predicted'][:, :rollout]
    target_S = out['slots']['target'][:, :rollout]
    depth_mask = out['depth_mask'][:, :rollout]
    
    total_loss, aux = loss_fn(pred_S, target_S,
                              energy=out.get('energy_pairs'),
                              depth_mask=depth_mask)
    
    optimizer.zero_grad()
    total_loss.backward()
    
    # 检查梯度
    if step % 20 == 0:
        grad_norms = {}
        for name, param in model.predictor.named_parameters():
            if param.grad is not None and param.grad.norm() > 0:
                if 'spatiotemporal' in name:
                    key = 'st'
                elif 'physics_module' in name:
                    key = 'phys'
                else:
                    key = 'other'
                grad_norms[key] = grad_norms.get(key, 0) + param.grad.norm().item()**2
        gn_str = "  ".join(f"{k}={v**0.5:.4f}" for k, v in sorted(grad_norms.items()))
        
        # 也记录预测的 pos/depth 误差
        with torch.no_grad():
            mask = depth_mask.unsqueeze(-1).float()
            pred_dyn = pred_S[:, :, :, app_dim:]
            target_dyn = target_S[:, :, :, app_dim:]
            pos_mse = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask).sum() / (mask.sum()*2+1e-8)
            depth_mse = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask).sum() / (mask.sum()+1e-8)
            
            # 逐帧 depth error
            frame_depth_errs = []
            for t in range(rollout):
                d_err = ((pred_dyn[:, t, :, 2:3] - target_dyn[:, t, :, 2:3])**2 * mask[:, t]).sum() / (mask[:, t].sum()+1e-8)
                frame_depth_errs.append(f"{d_err.item():.6f}")
        
        print(f"step {step:3d}: slot={aux['slot_loss']:.6f} pos={aux['slot_loss_pos']:.6f} depth={aux['slot_loss_depth']:.6f} | pos_mse={pos_mse:.6f} depth_mse={depth_mse:.6f} | grad: {gn_str}")
        if step % 100 == 0:
            print(f"  frame depth errs: {frame_depth_errs}")
    
    optimizer.step()
