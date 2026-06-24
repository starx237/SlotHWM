"""
实验6: 核心问题 - target slots 的 pos/depth 是否每个 batch 都不同？
如果 ISA 对同一个视频的 target slot 在每次 forward 中都不完全确定（有随机性），
那 loss 就会震荡，因为 target 本身在变。

另一个关键问题: GRU2 burnin 的 slot ordering 是否每帧一致？
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

# 完全相同的输入，forward 两次，看 target 是否一致
with torch.no_grad():
    out1 = model(frames)
    out2 = model(frames)

app_dim = cfg.appearance_dim
target1 = out1['slots']['target']
target2 = out2['slots']['target']

print("=== Target consistency check (same input, two forwards) ===")
diff = (target1 - target2).abs()
print(f"  Max diff: {diff.max().item():.8f}")
print(f"  Mean diff: {diff.mean().item():.8f}")

# 检查 ISA slot attention 是否有随机性（dropout?）
print(f"\n  Slot attention dropout rate: {model.slot_attention.dropout_rate if hasattr(model.slot_attention, 'dropout_rate') else 'N/A'}")

# 但更重要的是: 在 train 模式下，同样的输入 forward 两次，结果是否一致？
model.train()
with torch.no_grad():
    out3 = model(frames)
    out4 = model(frames)

target3 = out3['slots']['target']
target4 = out4['slots']['target']
diff34 = (target3 - target4).abs()
print(f"\n=== Train mode target consistency ===")
print(f"  Max diff: {diff34.max().item():.8f}")
print(f"  Mean diff: {diff34.mean().item():.8f}")

# 检查: 不同 batch 的 target 的 pos/depth 分布是否一致
print("\n=== Target pos/depth distribution across batches ===")
torch.manual_seed(0)
for i in range(5):
    batch_i = next(iter(ds.get_dataloader(batch_size=8, shuffle=True, num_workers=0)))
    frames_i = batch_i['video'].cuda()
    with torch.no_grad():
        out_i = model(frames_i)
    target_i = out_i['slots']['target']
    dyn_i = target_i[:, :, :, app_dim:]
    print(f"  Batch {i}: pos_x mean={dyn_i[...,0].mean():.4f} std={dyn_i[...,0].std():.4f}, "
          f"pos_y mean={dyn_i[...,1].mean():.4f} std={dyn_i[...,1].std():.4f}, "
          f"depth mean={dyn_i[...,2].mean():.4f} std={dyn_i[...,2].std():.4f}")
