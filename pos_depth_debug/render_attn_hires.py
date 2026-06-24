"""
高分辨率注意力热力图: 上采样到64x64，双线性插值
"""
import sys, os
sys.path.insert(0, '/autodl-fs/data/SlotHWM')
import warnings; warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
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

SI = 9
TARGET_SLOT = 0

all_attns = []
def attn_hook(module, input, output):
    if isinstance(output, tuple) and len(output) == 2:
        all_attns.append(output[1].detach().cpu())
handle = model.slot_attention.register_forward_hook(attn_hook)

sample = ds[SI]
frames = sample['video'].unsqueeze(0).cuda()

with torch.no_grad():
    out = model(frames)

handle.remove()

all_slots = torch.cat([out['slots']['corrected'], out['slots']['target']], dim=1)[0]

# 上采样注意力图 16x16 -> 64x64
def upsample_attn(attn_2d, target_size=64):
    """attn_2d: (16,16) numpy array -> (64,64) numpy array"""
    t = torch.from_numpy(attn_2d).float().unsqueeze(0).unsqueeze(0)  # (1,1,16,16)
    t_up = F.interpolate(t, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return t_up[0, 0].numpy()

# 画4x4网格，每格=视频帧(左半) + 注意力热力图(右半) 叠加在视频帧上
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
fig.suptitle(f'Sample {SI}, Slot {TARGET_SLOT}: Attention Heatmap Overlay (upsampled 16x16→64x64)\n'
             f'Green=burnin, Red=rollout. Depth values shown below frame number.', 
             fontsize=13, y=1.01)

for t in range(16):
    row = t // 4
    col = t % 4
    ax = axes[row, col]
    
    # 视频帧
    frame = sample['video'][t].permute(1, 2, 0).numpy()
    frame_vis = np.clip((frame + 1) / 2, 0, 1)
    ax.imshow(frame_vis)
    
    # 注意力热力图叠加
    if t < len(all_attns):
        attn_t = all_attns[t][0, TARGET_SLOT].numpy()  # (256,)
        attn_2d = attn_t.reshape(16, 16)
        attn_up = upsample_attn(attn_2d, 64)
        
        # 用 alpha 叠加热力图
        ax.imshow(attn_up, cmap='hot', alpha=0.5, vmin=0, vmax=attn_up.max())
    
    depth_t = all_slots[t, TARGET_SLOT, app_dim+2].item()
    pos_x_t = all_slots[t, TARGET_SLOT, app_dim].item()
    pos_y_t = all_slots[t, TARGET_SLOT, app_dim+1].item()
    
    # 标注 pos
    px = (pos_x_t + 1) / 2 * 64
    py = (pos_y_t + 1) / 2 * 64
    ax.plot(px, py, 'c+', markersize=15, markeredgewidth=3)
    
    phase = 'B' if t < 6 else 'R'
    title_color = 'black' if t < 6 else 'red'
    ax.set_title(f't={t} [{phase}] depth={depth_t:.4f}', fontsize=10, color=title_color)
    ax.axis('off')

plt.tight_layout()
out_path = '/autodl-fs/data/SlotHWM/pos_depth_debug/sample9_attn_overlay.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')

# 更清晰版: 单独的注意力热力图，每帧一个大图，用更好的 colormap
fig2, axes2 = plt.subplots(4, 4, figsize=(20, 20))
fig2.suptitle(f'Sample {SI}, Slot {TARGET_SLOT}: Pure Attention Heatmap (64x64 upsampled)\n'
              f'Cyan cross = slot position. Note shape changes between frames.', 
              fontsize=13, y=1.01)

for t in range(16):
    row = t // 4
    col = t % 4
    ax = axes2[row, col]
    
    if t < len(all_attns):
        attn_t = all_attns[t][0, TARGET_SLOT].numpy()
        attn_2d = attn_t.reshape(16, 16)
        attn_up = upsample_attn(attn_2d, 64)
        
        im = ax.imshow(attn_up, cmap='inferno', vmin=0, vmax=max(attn_up.max(), 0.001),
                       interpolation='bilinear')
        
        pos_x_t = all_slots[t, TARGET_SLOT, app_dim].item()
        pos_y_t = all_slots[t, TARGET_SLOT, app_dim+1].item()
        px = (pos_x_t + 1) / 2 * 64
        py = (pos_y_t + 1) / 2 * 64
        ax.plot(px, py, 'c+', markersize=15, markeredgewidth=3)
        
        depth_t = all_slots[t, TARGET_SLOT, app_dim+2].item()
        attn_max = attn_t.max()
        attn_entropy = -np.sum(attn_t * np.log(attn_t + 1e-10))
        
        phase = 'B' if t < 6 else 'R'
        title_color = 'black' if t < 6 else 'red'
        ax.set_title(f't={t}[{phase}] d={depth_t:.4f}\nmax={attn_max:.3f} H={attn_entropy:.2f}', 
                     fontsize=9, color=title_color)
    else:
        ax.set_title(f't={t} N/A', fontsize=9)
    
    ax.axis('off')

plt.tight_layout()
out_path2 = '/autodl-fs/data/SlotHWM/pos_depth_debug/sample9_attn_pure.png'
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path2}')

# GIF: 16帧视频 + 注意力叠加动画
from PIL import Image
gif_frames = []
for t in range(16):
    fig_gif, ax_gif = plt.subplots(1, 1, figsize=(4, 4))
    frame = sample['video'][t].permute(1, 2, 0).numpy()
    frame_vis = np.clip((frame + 1) / 2, 0, 1)
    ax_gif.imshow(frame_vis)
    
    if t < len(all_attns):
        attn_t = all_attns[t][0, TARGET_SLOT].numpy()
        attn_2d = attn_t.reshape(16, 16)
        attn_up = upsample_attn(attn_2d, 64)
        ax_gif.imshow(attn_up, cmap='hot', alpha=0.5, vmin=0, vmax=max(attn_up.max(), 0.001))
    
    depth_t = all_slots[t, TARGET_SLOT, app_dim+2].item()
    ax_gif.set_title(f't={t} depth={depth_t:.4f}', fontsize=10)
    ax_gif.axis('off')
    
    fig_gif.canvas.draw()
    img = np.frombuffer(fig_gif.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig_gif.canvas.get_width_height()[::-1] + (3,))
    gif_frames.append(Image.fromarray(img))
    plt.close(fig_gif)

gif_path = '/autodl-fs/data/SlotHWM/pos_depth_debug/sample9_attn_overlay.gif'
gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=300, loop=0)
print(f'Saved: {gif_path}')
