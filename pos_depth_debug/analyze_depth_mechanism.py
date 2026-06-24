"""
问题1: 碰撞时 depth 变化的机制分析
碰撞 = 两个物体的 slot 在同一帧竞争同一批像素
当 slot A 的注意力被 slot B 抢走部分像素时:
  - attn 分布从紧凑变分散(或反之)
  - pos = Σ attn × grid 偏移
  - depth = sqrt(Σ attn × spread) 变化

问题2: 预设 depth vs 实际 depth
slot_attention 输入: slots[..., -1:] = depth_init (来自上一帧或 slot_depth)
slot_attention 输出: depth = sqrt(Σ attn × spread) (重新计算)
差距大 → depth 初始化几乎没用 → 可能需要 depth loss
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')
import warnings; warnings.filterwarnings('ignore')
import torch
import numpy as np
import yaml
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

# Hook: 收集 slot_attention 的输入 slots 和输出 slots
hook_data = []
def hook_fn(module, input, output):
    if isinstance(output, tuple) and len(output) == 2:
        # input[0] = features, input[1] = slots (if provided)
        slots_in = input[1] if len(input) > 1 and input[1] is not None else None
        slots_out = output[0]
        attn_out = output[1]
        hook_data.append({
            'slots_in': slots_in.detach().cpu() if slots_in is not None else None,
            'slots_out': slots_out.detach().cpu(),
            'attn': attn_out.detach().cpu(),
        })

handle = model.slot_attention.register_forward_hook(hook_fn)

# ========== 问题2: 预设 depth vs 实际 depth ==========
print("=" * 60)
print("问题2: 预设 depth (输入) vs 实际 depth (输出)")
print("=" * 60)

# 对多个样本统计
depth_init_list = []
depth_out_list = []
depth_diff_list = []

for si in range(50):
    hook_data.clear()
    sample = ds[si]
    frames = sample['video'].unsqueeze(0).cuda()
    
    with torch.no_grad():
        out = model(frames)
    
    # hook_data 有 16 个（每帧一次 slot_attention 调用）
    for t, hd in enumerate(hook_data):
        slots_in = hd['slots_in']   # (1, N, D) 或 None
        slots_out = hd['slots_out']  # (1, N, D)
        attn = hd['attn']           # (1, N, N_pix)
        
        if slots_in is None:
            # 第一帧，用 slot_mu 初始化，没有预设 depth
            continue
        
        N = slots_in.shape[1]
        for s in range(N):
            d_in = slots_in[0, s, -1].item()   # 输入的 depth (上一帧输出)
            d_out = slots_out[0, s, -1].item()  # 输出的 depth (重新计算)
            
            if d_out >= depth_max:
                continue  # 跳过背景
            
            depth_init_list.append(d_in)
            depth_out_list.append(d_out)
            depth_diff_list.append(abs(d_out - d_in))

depth_init = np.array(depth_init_list)
depth_out = np.array(depth_out_list)
depth_diff = np.array(depth_diff_list)

print(f"\n总计 {len(depth_diff)} 个 (slot, frame) 对")
print(f"预设 depth (输入):  mean={depth_init.mean():.4f} std={depth_init.std():.4f}")
print(f"实际 depth (输出):  mean={depth_out.mean():.4f} std={depth_out.std():.4f}")
print(f"|depth_out - depth_in|: mean={depth_diff.mean():.6f} median={np.median(depth_diff):.6f}")
print(f"  90th percentile: {np.percentile(depth_diff, 90):.6f}")
print(f"  95th percentile: {np.percentile(depth_diff, 95):.6f}")
print(f"  max: {depth_diff.max():.6f}")

# Correlation
corr = np.corrcoef(depth_init, depth_out)[0, 1]
print(f"  Correlation(input_depth, output_depth): {corr:.4f}")

# 分 FG/BG 看
fg_mask = depth_out < depth_max
if fg_mask.sum() > 0:
    fg_diff = depth_diff[fg_mask]
    print(f"\n  FG only (depth < {depth_max}):")
    print(f"    |Δ|: mean={fg_diff.mean():.6f} median={np.median(fg_diff):.6f} max={fg_diff.max():.6f}")

# 逐帧分析
print(f"\n逐帧 |depth_out - depth_in| (FG only):")
frame_diffs = {}
idx = 0
for si in range(50):
    for t, hd in enumerate(hook_data if si == 0 else []):
        pass  # 太复杂，简化

# 简化: 对第一个样本做详细分析
print(f"\n{'='*60}")
print(f"详细分析 Sample 9")
print(f"{'='*60}")

hook_data.clear()
sample = ds[9]
frames = sample['video'].unsqueeze(0).cuda()

with torch.no_grad():
    feat_with_grid = model._encode_features(frames)
    out = model(frames)

all_slots = torch.cat([out['slots']['corrected'], out['slots']['target']], dim=1)[0]
grid = feat_with_grid[0, 0, :, -2:].cpu().numpy()  # (256, 2), [-1,1]

N = all_slots.shape[1]
for t in range(min(16, len(hook_data))):
    hd = hook_data[t]
    slots_in = hd['slots_in']
    slots_out = hd['slots_out']
    attn = hd['attn']
    
    if slots_in is None:
        print(f"\nt={t}: No input slots (first frame)")
        continue
    
    print(f"\nt={t}:")
    for s in range(N):
        d_in = slots_in[0, s, -1].item()
        d_out = slots_out[0, s, -1].item()
        
        if d_out >= depth_max and d_in >= depth_max:
            continue
        
        pos_in_x = slots_in[0, s, -3].item()
        pos_in_y = slots_in[0, s, -2].item()
        pos_out_x = slots_out[0, s, -3].item()
        pos_out_y = slots_out[0, s, -2].item()
        
        attn_s = attn[0, s].numpy()  # (256,)
        attn_max = attn_s.max()
        attn_entropy = -np.sum(attn_s * np.log(attn_s + 1e-10))
        
        # 手动计算 depth
        pos_out = np.array([pos_out_x, pos_out_y])
        spread = np.sum((grid - pos_out)**2, axis=-1)
        depth_manual = np.sqrt(np.sum(attn_s * spread))
        
        # 用预设 pos 计算 depth (隔离 pos 变化的影响)
        pos_in = np.array([pos_in_x, pos_in_y])
        spread_in = np.sum((grid - pos_in)**2, axis=-1)
        depth_at_in_pos = np.sqrt(np.sum(attn_s * spread_in))
        
        delta_d = d_out - d_in
        delta_pos = np.sqrt((pos_out_x - pos_in_x)**2 + (pos_out_y - pos_in_y)**2)
        
        print(f"  sl{s}: d_in={d_in:.4f} → d_out={d_out:.4f} (Δ={delta_d:+.4f}) | "
              f"pos_in=({pos_in_x:.3f},{pos_in_y:.3f})→({pos_out_x:.3f},{pos_out_y:.3f}) Δpos={delta_pos:.4f} | "
              f"d_manual={depth_manual:.4f} d@in_pos={depth_at_in_pos:.4f} | "
              f"attn_max={attn_max:.3f} H={attn_entropy:.2f}")

handle.remove()

# ========== 问题1 补充: 碰撞帧分析 ==========
print(f"\n{'='*60}")
print(f"问题1: 碰撞时两个 slot 的注意力重叠度")
print(f"{'='*60}")

# 找同一帧中两个 FG slot 的注意力重叠
hook_data.clear()
sample = ds[9]
frames = sample['video'].unsqueeze(0).cuda()

with torch.no_grad():
    out = model(frames)

for t in range(min(16, len(hook_data))):
    attn = hook_data[t]['attn'][0]  # (N, 256)
    fg_slots = []
    for s in range(attn.shape[0]):
        d = hook_data[t]['slots_out'][0, s, -1].item()
        if d < depth_max:
            fg_slots.append(s)
    
    if len(fg_slots) < 2:
        continue
    
    # 计算每对 FG slot 的注意力重叠
    overlaps = []
    for i in range(len(fg_slots)):
        for j in range(i+1, len(fg_slots)):
            si, sj = fg_slots[i], fg_slots[j]
            a_i = attn[si].numpy()
            a_j = attn[sj].numpy()
            # 重叠度 = min(attn_i, attn_j).sum() / max(attn_i, attn_j).sum()
            overlap = np.minimum(a_i, a_j).sum() / np.maximum(a_i, a_j).sum()
            overlaps.append((si, sj, overlap))
    
    if max(o[2] for o in overlaps) > 0.1:  # 只显示有显著重叠的帧
        max_overlap = max(overlaps, key=lambda x: x[2])
        print(f"  t={t}: max overlap sl{max_overlap[0]}-sl{max_overlap[1]} = {max_overlap[2]:.4f}")
        for si, sj, ov in overlaps:
            if ov > 0.05:
                di = hook_data[t]['slots_out'][0, si, -1].item()
                dj = hook_data[t]['slots_out'][0, sj, -1].item()
                print(f"    sl{si}(d={di:.4f})-sl{sj}(d={dj:.4f}): overlap={ov:.4f}")
