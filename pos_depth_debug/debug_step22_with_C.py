"""
实验22: 最终验证 - 给时空模块加入 C (appearance) 作为额外输入

当前 spatiotemporal_module 的 embed_dim = dyn_total_dim = 3
测试: 把 C 拼接到 Z^d 输入中，embed_dim = 3 + static_dim

这是为了验证: appearance 信息是否有助于预测运动方向
如果加入 C 后能跨 batch 泛化，说明问题确实是 Z^d 信息不足

由于不能修改主代码，这里直接构建一个自定义的 predictor forward
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from models.attention import TimeSpaceTransformerBlock2
from data.obj3d_dataset import OBJ3DDataset

with open('/autodl-fs/data/SlotHWM/config/obj3d.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)

app_dim = cfg.appearance_dim
rollout = cfg.rollout_frames
burnin = cfg.burnin_frames
static_dim = cfg.static_dim

# 构建一个自定义的时空模块，输入包含 C
st_dim_with_C = cfg.dynamic_dim + static_dim  # 3 + 64 = 67 (当 static_dim=64)

st_module_with_C = nn.ModuleList([
    TimeSpaceTransformerBlock2(
        embed_dim=st_dim_with_C,
        num_heads=cfg.num_heads,
        qkv_size=cfg.qkv_size,
        mlp_size=cfg.mlp_size,
        pre_norm=getattr(cfg, 'spatiotemporal_pre_norm', cfg.pre_norm),
    ) for _ in range(cfg.num_spatiotemporal_blocks)
]).cuda()

opt_C = torch.optim.Adam(st_module_with_C.parameters(), lr=1e-4)

# 同样构建只有 Z^d 的模块作为对比
st_dim_no_C = cfg.dynamic_dim  # 3

st_module_no_C = nn.ModuleList([
    TimeSpaceTransformerBlock2(
        embed_dim=st_dim_no_C,
        num_heads=1,  # 3维只能1个head
        qkv_size=3,
        mlp_size=64,
        pre_norm=True,
    ) for _ in range(2)
]).cuda()

opt_no_C = torch.optim.Adam(st_module_no_C.parameters(), lr=1e-4)

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

def run_st_module(module, st_input, st_buffer):
    """Run spatiotemporal module with residual accumulation"""
    h = st_input
    total_residual = torch.zeros_like(h)
    for block in module:
        total_residual = total_residual + block(h, st_buffer)
        h = st_input + total_residual
    return total_residual

print("=== With C (appearance) vs Without C ===")
for step in range(300):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    
    with torch.no_grad():
        out = model(frames)
    
    target_S = out['slots']['target'][:, :rollout]
    depth_mask = out['depth_mask'][:, :rollout]
    corrected_S = out['slots']['corrected']
    
    # 构建 Z 空间
    burnin_Z = []
    for t in range(burnin):
        S_t = corrected_S[:, t]
        Z_app = model.f_z(S_t[:, :, :app_dim])
        Z_t = torch.cat([Z_app, S_t[:, :, app_dim:]], dim=-1)
        burnin_Z.append(Z_t)
    burnin_Z = torch.stack(burnin_Z, dim=1)
    
    # 计算 C
    C = model.predictor.compute_C(burnin_Z)  # (B, N, static_dim)
    
    # --- With C ---
    # 输入: cat([C, Z^d]) for each slot
    burnin_dyn = corrected_S[:, :, :, app_dim:]  # (B, burnin, N, 3)
    cur_dyn = burnin_dyn[:, -1]  # (B, N, 3)
    
    buffer_with_C = torch.cat([
        C.unsqueeze(1).expand(-1, burnin, -1, -1),  # (B, burnin, N, 64)
        burnin_dyn  # (B, burnin, N, 3)
    ], dim=-1)  # (B, burnin, N, 67)
    
    cur_with_C = torch.cat([C, cur_dyn], dim=-1)  # (B, N, 67)
    
    # 自回归 rollout
    total_loss_C = 0
    h = cur_with_C
    buf = buffer_with_C
    for t in range(rollout):
        residual = run_st_module(st_module_with_C, h, buf)
        next_dyn_pred = cur_dyn + residual[:, :, -3:]  # 只取 Z^d 部分
        
        target_dyn_t = target_S[:, t, :, app_dim:]
        mask_t = depth_mask[:, t].unsqueeze(-1).float()
        loss_t = ((next_dyn_pred - target_dyn_t)**2 * mask_t).sum() / (mask_t.sum() * 3 + 1e-8)
        total_loss_C = total_loss_C + loss_t
        
        # 更新
        cur_dyn = next_dyn_pred.detach()
        h = torch.cat([C, cur_dyn], dim=-1)
        buf = torch.cat([buf[:, 1:], h.unsqueeze(1)], dim=1)
    
    total_loss_C = total_loss_C / rollout
    opt_C.zero_grad()
    total_loss_C.backward()
    torch.nn.utils.clip_grad_norm_(st_module_with_C.parameters(), 1.0)
    opt_C.step()
    
    # --- Without C ---
    # 重新初始化
    burnin_dyn2 = corrected_S[:, :, :, app_dim:]
    cur_dyn2 = burnin_dyn2[:, -1].clone()
    buf2 = burnin_dyn2.clone()
    
    total_loss_noC = 0
    for t in range(rollout):
        residual = run_st_module(st_module_no_C, cur_dyn2, buf2)
        next_dyn_pred2 = cur_dyn2 + residual
        
        target_dyn_t = target_S[:, t, :, app_dim:]
        mask_t = depth_mask[:, t].unsqueeze(-1).float()
        loss_t = ((next_dyn_pred2 - target_dyn_t)**2 * mask_t).sum() / (mask_t.sum() * 3 + 1e-8)
        total_loss_noC = total_loss_noC + loss_t
        
        cur_dyn2 = next_dyn_pred2.detach()
        buf2 = torch.cat([buf2[:, 1:], cur_dyn2.unsqueeze(1)], dim=1)
    
    total_loss_noC = total_loss_noC / rollout
    opt_no_C.zero_grad()
    total_loss_noC.backward()
    torch.nn.utils.clip_grad_norm_(st_module_no_C.parameters(), 1.0)
    opt_no_C.step()
    
    if step % 50 == 0:
        print(f"step {step:3d}: with_C={total_loss_C.item():.6f} no_C={total_loss_noC.item():.6f}")

# 最终跨 batch 评估
print("\n=== Cross-batch evaluation ===")
torch.manual_seed(999)
batch = next(iter(ds.get_dataloader(batch_size=8, shuffle=True, num_workers=0)))
frames = batch['video'].cuda()

with torch.no_grad():
    out = model(frames)
    target_S = out['slots']['target']
    corrected_S = out['slots']['corrected']
    depth_mask = out['depth_mask']
    
    burnin_Z = []
    for t in range(burnin):
        S_t = corrected_S[:, t]
        Z_app = model.f_z(S_t[:, :, :app_dim])
        Z_t = torch.cat([Z_app, S_t[:, :, app_dim:]], dim=-1)
        burnin_Z.append(Z_t)
    burnin_Z = torch.stack(burnin_Z, dim=1)
    C = model.predictor.compute_C(burnin_Z)
    
    # With C
    burnin_dyn = corrected_S[:, :, :, app_dim:]
    cur_dyn = burnin_dyn[:, -1]
    buffer_with_C = torch.cat([C.unsqueeze(1).expand(-1, burnin, -1, -1), burnin_dyn], dim=-1)
    cur_with_C = torch.cat([C, cur_dyn], dim=-1)
    
    for name, module, use_C in [("With C", st_module_with_C, True), ("No C", st_module_no_C, False)]:
        if use_C:
            h = cur_with_C
            buf = buffer_with_C
            cur_d = cur_dyn.clone()
        else:
            h = cur_dyn.clone()
            buf = burnin_dyn.clone()
            cur_d = cur_dyn.clone()
        
        total_pos = 0
        total_depth = 0
        for t in range(rollout):
            residual = run_st_module(module, h, buf)
            if use_C:
                next_d = cur_d + residual[:, :, -3:]
            else:
                next_d = cur_d + residual
            
            target_dyn_t = target_S[:, t, :, app_dim:]
            mask_t = depth_mask[:, t].unsqueeze(-1).float()
            pos_mse = ((next_d[..., :2] - target_dyn_t[..., :2])**2 * mask_t).sum() / (mask_t.sum()*2+1e-8)
            depth_mse = ((next_d[..., 2:3] - target_dyn_t[..., 2:3])**2 * mask_t).sum() / (mask_t.sum()+1e-8)
            total_pos += pos_mse.item()
            total_depth += depth_mse.item()
            
            cur_d = next_d
            if use_C:
                h = torch.cat([C, cur_d], dim=-1)
                buf = torch.cat([buf[:, 1:], h.unsqueeze(1)], dim=1)
            else:
                h = cur_d
                buf = torch.cat([buf[:, 1:], cur_d.unsqueeze(1)], dim=1)
        
        print(f"  {name}: avg_pos={total_pos/rollout:.6f} avg_depth={total_depth/rollout:.6f}")
