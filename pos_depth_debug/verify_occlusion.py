"""
验证: depth 跳变是否与遮挡相关
思路: 找一个深度跳变的帧，检查该帧的 attention map 是否有异常
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

# 获取 slot attention 的 attn map
# 需要修改 forward 来返回 attn
# 用 hook 来获取

attn_maps = {}

def get_attn_hook(name):
    def hook(module, input, output):
        # output = (slots, attn)
        if isinstance(output, tuple) and len(output) == 2:
            attn_maps[name] = output[1].detach().cpu()
    return hook

model.slot_attention.register_forward_hook(get_attn_hook('slot_attn'))

# 找有 depth 跳变的样本
sample = ds[1]  # 之前看到 s1_sl0 有大跳变
frames = sample['video'].unsqueeze(0).cuda()

# 逐帧获取 attention map
all_attns = []
all_slots_data = []

for t in range(16):
    frame_t = frames[:, t:t+1]  # (1, 1, 3, 64, 64)
    
    with torch.no_grad():
        # 编码
        enc_out = model.encoder(frame_t)  # (1, 1, N_pix, D)
        enc_out = enc_out.squeeze(1)  # (1, N_pix, D)
        
        # slot attention
        if t == 0:
            slots_init = None
        else:
            # 用上一帧的 slots 作为初始化 (GRU2 传递)
            slots_init = all_slots_data[-1].unsqueeze(0)  # (1, N_slots, D)
        
        attn_maps.clear()
        slots, attn = model.slot_attention(enc_out, slots=slots_init, num_iterations=3)
        # attn: (1, N_slots, N_pixels)
        
        all_attns.append(attn[0].cpu())  # (N_slots, N_pixels)
        all_slots_data.append(slots[0].cpu())

# 分析 depth 和 attn 的关系
N = all_slots_data[0].shape[0]
print(f"N_slots: {N}")

for s in range(N):
    depths = [all_slots_data[t][s, app_dim+2].item() for t in range(16)]
    depths = np.array(depths)
    
    if np.mean(depths) >= depth_max:
        continue
    
    # 找最大跳变
    dd = np.abs(np.diff(depths))
    max_jump_idx = dd.argmax()
    max_jump = dd[max_jump_idx]
    
    if max_jump < 0.005:
        continue
    
    print(f"\nSlot {s}: max_jump={max_jump:.4f}@t={max_jump_idx}→{max_jump_idx+1}")
    print(f"  depth: {[f'{d:.4f}' for d in depths]}")
    
    # 检查跳变前后的 attention map
    for t in [max_jump_idx, max_jump_idx+1]:
        attn_t = all_attns[t][s].numpy()  # (N_pixels,)  N_pixels = 8*8=64
        # reshape to spatial
        H = W = 8
        attn_2d = attn_t.reshape(H, W)
        print(f"  t={t}: attn sum={attn_t.sum():.4f} max={attn_t.max():.4f} entropy={-np.sum(attn_t*np.log(attn_t+1e-10)):.4f}")
        print(f"         attn spatial range: [{attn_t.min():.4f}, {attn_t.max():.4f}]")

# 可视化: 选一个有代表性跳变的slot，画 attention map 的逐帧变化
# 找 slot 0 (之前看到的大跳变)
s = 0
depths_s = np.array([all_slots_data[t][s, app_dim+2].item() for t in range(16)])

fig, axes = plt.subplots(3, 6, figsize=(24, 10))
fig.suptitle(f'Slot {s} Attention Map Evolution (depth jump analysis)\nDepth: {[f"{d:.3f}" for d in depths_s]}', fontsize=12, y=1.02)

H = W = 8
for t in range(16):
    row = t // 6
    col = t % 6
    if row >= 3:
        break
    ax = axes[row, col]
    attn_t = all_attns[t][s].numpy().reshape(H, W)
    im = ax.imshow(attn_t, cmap='hot', vmin=0, vmax=attn_t.max()+0.01)
    ax.set_title(f't={t} d={depths_s[t]:.4f}', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 如果是跳变帧，标红框
    if t > 0 and abs(depths_s[t] - depths_s[t-1]) > 0.005:
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)

plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/attn_depth_jump.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: pos_depth_debug/attn_depth_jump.png")

# 更详细: 对比跳变前后两帧的 attn diff
if max_jump_idx >= 0:
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    fig2.suptitle(f'Slot {s}: Attention change at depth jump (t={max_jump_idx}→{max_jump_idx+1})', fontsize=12)
    
    for idx, t in enumerate([max_jump_idx, max_jump_idx+1]):
        attn_t = all_attns[t][s].numpy().reshape(H, W)
        im = axes2[idx, 0].imshow(attn_t, cmap='hot')
        axes2[idx, 0].set_title(f't={t} attention')
        plt.colorbar(im, ax=axes2[idx, 0])
    
    # diff
    attn_before = all_attns[max_jump_idx][s].numpy().reshape(H, W)
    attn_after = all_attns[max_jump_idx+1][s].numpy().reshape(H, W)
    diff = attn_after - attn_before
    im = axes2[0, 1].imshow(diff, cmap='RdBu_r', vmin=-abs(diff).max(), vmax=abs(diff).max())
    axes2[0, 1].set_title(f'Attn diff (t{max_jump_idx+1}-t{max_jump_idx})')
    plt.colorbar(im, ax=axes2[0, 1])
    
    # 所有slot的depth
    for s2 in range(N):
        d2 = np.array([all_slots_data[t][s2, app_dim+2].item() for t in range(16)])
        if np.mean(d2) >= depth_max:
            continue
        lw = 2.5 if s2 == s else 0.8
        axes2[1, 1].plot(range(16), d2, '-o', markersize=3, linewidth=lw, label=f'sl{s2}')
    axes2[1, 1].axvline(x=max_jump_idx+0.5, color='red', linestyle='--')
    axes2[1, 1].set_title('All slots depth')
    axes2[1, 1].legend(fontsize=6)
    
    # slot masks
    for idx, t in enumerate([max_jump_idx, max_jump_idx+1]):
        slots_frame = all_slots_data[t].unsqueeze(0)
        # 需要cuda
        slots_frame_cuda = slots_frame.cuda()
        with torch.no_grad():
            _, alphas = model.decoder(slots_frame_cuda, return_alphas=True)
        mask = alphas[0, s, 0].cpu().numpy()
        axes2[1, 2+idx if 2+idx<3 else 2].imshow(mask, cmap='gray')
        # 标题放对位置
    
    plt.tight_layout()
    plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/attn_jump_detail.png', dpi=150, bbox_inches='tight')
    print(f"Saved: pos_depth_debug/attn_jump_detail.png")
