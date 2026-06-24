"""
验证遮挡与 depth 跳变的关系
用 hook 获取 slot attention 的 attn map
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')
import warnings; warnings.filterwarnings('ignore')
import torch
import numpy as np
import yaml, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset

with open('config/obj3d.yaml') as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))

app_dim = cfg.appearance_dim
depth_max = getattr(cfg, 'depth_max', 0.30)

model = SlotDynamicsModel(cfg).cuda()
ckpt = torch.load(cfg.pretrained_path, map_location='cpu')
sd = model.state_dict()
ld = {}
for mk in sd:
    mc = mk.replace('_orig_mod.','')
    for ck in ckpt['model']:
        cc = ck.replace('_orig_mod.','')
        if cc==mc and ckpt['model'][ck].shape==sd[mk].shape:
            ld[mk]=ckpt['model'][ck]; break
model.load_state_dict(ld, strict=False)
model.eval()

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)

# Hook slot attention
attn_buffer = {}
def attn_hook(module, input, output):
    if isinstance(output, tuple) and len(output) == 2:
        attn_buffer['attn'] = output[1].detach().cpu()

handle = model.slot_attention.register_forward_hook(attn_hook)

# 用完整 forward，收集每帧的 attn
# 但 forward 一次处理16帧，slot_attention 被调用16次
# 需要收集所有调用

all_attns_per_call = []
original_hook = attn_hook

def multi_call_hook(module, input, output):
    if isinstance(output, tuple) and len(output) == 2:
        all_attns_per_call.append(output[1].detach().cpu())

handle.remove()
handle = model.slot_attention.register_forward_hook(multi_call_hook)

# 获取一个样本
for si in [1, 10, 14]:
    all_attns_per_call.clear()
    
    sample = ds[si]
    frames = sample['video'].unsqueeze(0).cuda()
    
    with torch.no_grad():
        out = model(frames)
    
    corrected = out['slots']['corrected']
    target = out['slots']['target']
    all_slots = torch.cat([corrected, target], dim=1)[0]
    
    N_frames = all_slots.shape[0]
    N_slots = all_slots.shape[1]
    
    print(f"\n=== Sample {si}: {len(all_attns_per_call)} attn maps collected (expect {N_frames}) ===")
    
    # slot attention 在 finetune 模式下会被调用: 
    # burnin 阶段每帧1次 (6次) + rollout target 每帧1次 (10次) = 16次
    # 但 burnin 阶段可能也有 target 的调用
    
    for s in range(N_slots):
        depth = all_slots[:, s, app_dim+2].cpu().numpy()
        if np.mean(depth) >= depth_max:
            continue
        
        dd = np.abs(np.diff(depth))
        max_jump_idx = dd.argmax()
        max_jump = dd[max_jump_idx]
        
        if max_jump < 0.003:
            continue
        
        total_move = np.sqrt((all_slots[-1,s,app_dim]-all_slots[0,s,app_dim]).item()**2 + 
                             (all_slots[-1,s,app_dim+1]-all_slots[0,s,app_dim+1]).item()**2)
        
        print(f"\n  Slot {s}: max_jump={max_jump:.4f}@t={max_jump_idx}→{max_jump_idx+1} move={total_move:.3f}")
        print(f"    depth: {[f'{d:.4f}' for d in depth]}")
        
        # 检查跳变帧的 attn
        if len(all_attns_per_call) >= N_frames:
            for t in [max(max_jump_idx, 0), min(max_jump_idx+1, N_frames-1)]:
                if t < len(all_attns_per_call):
                    attn_t = all_attns_per_call[t][0, s].numpy()  # (N_pixels,)
                    n_pix = attn_t.shape[0]
                    H = W = int(np.sqrt(n_pix))
                    if H*W == n_pix:
                        attn_2d = attn_t.reshape(H, W)
                        print(f"    t={t}: attn_sum={attn_t.sum():.4f} max={attn_t.max():.4f} "
                              f"mean={attn_t.mean():.4f} n_active(>0.01)={np.sum(attn_t>0.01)}")
                    else:
                        print(f"    t={t}: N_pixels={n_pix} (not square)")

# 可视化: sample 1, slot 0 (之前看到的大跳变)
si = 1
all_attns_per_call.clear()
sample = ds[si]
frames = sample['video'].unsqueeze(0).cuda()

with torch.no_grad():
    out = model(frames)

corrected = out['slots']['corrected']
target = out['slots']['target']
all_slots = torch.cat([corrected, target], dim=1)[0]

print(f"\nSample {si}: {len(all_attns_per_call)} attn maps")

# slot 0 的 depth
s = 0
depths = all_slots[:, s, app_dim+2].cpu().numpy()
pos_x = all_slots[:, s, app_dim].cpu().numpy()
pos_y = all_slots[:, s, app_dim+1].cpu().numpy()

# 可视化每帧的 attn map
n_attn = min(len(all_attns_per_call), 16)
H = W = 8

fig, axes = plt.subplots(3, n_attn, figsize=(3*n_attn, 8))
if n_attn == 1:
    axes = axes.reshape(3, 1)
fig.suptitle(f'Sample {si} Slot {s}: Attention Maps per Frame\nDepth: {[f"{d:.3f}" for d in depths[:n_attn]]}', fontsize=11, y=1.02)

for t in range(min(n_attn, 16)):
    # Row 0: attention map
    attn_t = all_attns_per_call[t][0, s].numpy()
    n_pix = attn_t.shape[0]
    h = w = int(np.sqrt(n_pix))
    if h*w != n_pix:
        continue
    attn_2d = attn_t.reshape(h, w)
    
    im = axes[0, t].imshow(attn_2d, cmap='hot', vmin=0, vmax=max(attn_2d.max(), 0.01))
    axes[0, t].set_title(f't={t} d={depths[t]:.4f}', fontsize=8, 
                         color='red' if t>0 and abs(depths[t]-depths[t-1])>0.005 else 'black')
    
    # Row 1: 原始帧
    frame = sample['video'][t].permute(1, 2, 0).numpy()
    axes[1, t].imshow(np.clip((frame+1)/2, 0, 1))
    axes[1, t].set_title(f't={t}', fontsize=8)
    axes[1, t].axis('off')
    
    # Row 2: 所有slot的depth
    pass

# Row 2: depth 曲线
ax_depth = fig.add_subplot(3, 1, 3)
for s2 in range(all_slots.shape[1]):
    d2 = all_slots[:, s2, app_dim+2].cpu().numpy()
    if np.mean(d2) >= depth_max:
        continue
    lw = 2.5 if s2 == s else 0.8
    ax_depth.plot(range(16), d2, '-o', markersize=3, linewidth=lw, alpha=0.8 if s2==s else 0.4, label=f'sl{s2}')
ax_depth.axvline(x=5.5, color='green', linestyle='--', alpha=0.5)
ax_depth.set_title(f'All slots depth (sl{s} bold)')
ax_depth.legend(fontsize=6, ncol=3)
ax_depth.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/attn_depth_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: pos_depth_debug/attn_depth_analysis.png")

handle.remove()
