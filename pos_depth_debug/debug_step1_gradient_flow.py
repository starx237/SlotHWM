"""
实验1: 检查梯度流 - slot_loss 对 pos/depth 的梯度是否真正到达 predictor 参数
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')

import torch
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
_isa = ('encoder.', 'slot_attention.', 'decoder.', 'gru2.', 'gru2_proj.', 'f_z.')
loaded = {}
for mk in model_state:
    mk_c = mk.replace('_orig_mod.', '')
    for ck in ckpt['model']:
        ck_c = ck.replace('_orig_mod.', '')
        if ck_c == mk_c and ckpt['model'][ck].shape == model_state[mk].shape:
            loaded[mk] = ckpt['model'][ck]
            break
model.load_state_dict(loaded, strict=False)
model.train()

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
torch.manual_seed(42)
batch = next(iter(ds.get_dataloader(batch_size=4, shuffle=True, num_workers=0)))
frames = batch['video'].cuda()

out = model(frames)

rollout = cfg.rollout_frames
pred_S = out['slots']['predicted'][:, :rollout]
target_S = out['slots']['target'][:, :rollout]
depth_mask = out['depth_mask'][:, :rollout]

loss_fn = SlotPiLoss(cfg)
total_loss, aux = loss_fn(
    pred_S, target_S,
    energy=out.get('energy_pairs'),
    depth_mask=depth_mask
)

print("=== Loss values ===")
print(f"slot_loss: {aux['slot_loss']:.6f}")
print(f"slot_loss_dyn: {aux['slot_loss_dyn']:.6f}")
print(f"slot_loss_app: {aux['slot_loss_app']:.6f}")
print(f"slot_loss_pos: {aux['slot_loss_pos']:.6f}")
print(f"slot_loss_depth: {aux['slot_loss_depth']:.6f}")
print(f"depth_mask_ratio: {aux.get('depth_mask_ratio', 'N/A')}")

# backward
total_loss.backward()

# 检查 predictor 各子模块梯度
print("\n=== Predictor gradient norms ===")
grad_groups = {}
for name, param in model.predictor.named_parameters():
    if param.grad is None:
        continue
    gn = param.grad.norm().item()
    # 归组
    if 'spatiotemporal' in name:
        key = 'spatiotemporal'
    elif 'physics_module' in name:
        key = 'physics_module'
    elif 'fusion_mlp' in name:
        key = 'fusion_mlp'
    elif 'C_time_attn' in name:
        key = 'C_time_attn'
    else:
        key = 'other'
    grad_groups[key] = grad_groups.get(key, 0) + gn ** 2

total_gn2 = sum(grad_groups.values())
for k, v in sorted(grad_groups.items(), key=lambda x: -x[1]):
    print(f"  {k:25s}: grad_norm={v**0.5:.6f}  ({v/total_gn2*100:.1f}%)")

# 检查 pred_S 对 pos/depth 的梯度
print("\n=== Direct gradient check on pred_S ===")
# 重新 forward
model.zero_grad()
out2 = model(frames)
pred2 = out2['slots']['predicted'][:, :rollout]
# 单独计算 pos/depth loss
app_dim = cfg.appearance_dim
mask = depth_mask.unsqueeze(-1).float()
pred_dyn = pred2[:, :, :, app_dim:]
target_dyn = out2['slots']['target'][:, :rollout, :, app_dim:]

pos_loss = ((pred_dyn[..., :2] - target_dyn[..., :2])**2 * mask).sum() / (mask.sum() * 2 + 1e-8)
depth_loss = ((pred_dyn[..., 2:3] - target_dyn[..., 2:3])**2 * mask).sum() / (mask.sum() + 1e-8)

pos_loss.backward(retain_graph=True)
pos_grad = {n: p.grad.norm().item() for n, p in model.predictor.named_parameters() if p.grad is not None}
print(f"  pos_loss backward - predictor params with grad: {len(pos_grad)}")
if pos_grad:
    max_g = max(pos_grad.values())
    print(f"  max grad: {max_g:.8f}")

model.zero_grad()
depth_loss.backward()
depth_grad = {n: p.grad.norm().item() for n, p in model.predictor.named_parameters() if p.grad is not None}
print(f"  depth_loss backward - predictor params with grad: {len(depth_grad)}")
if depth_grad:
    max_g = max(depth_grad.values())
    print(f"  max grad: {max_g:.8f}")
