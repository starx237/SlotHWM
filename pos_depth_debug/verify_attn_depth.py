"""
精确验证: depth 跳变 vs 注意力分布变化
对画面内物体，手动计算 depth = sqrt(sum(attn * spread))
看注意力分布的哪些变化导致了 depth 跳变
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
bnd_thresh = 0.7

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

# 收集 hook 数据
all_hook = []
def hook_fn(module, input, output):
    if isinstance(output, tuple) and len(output) == 2:
        all_hook.append({
            'attn': output[1].detach().cpu(),
        })
handle = model.slot_attention.register_forward_hook(hook_fn)

# 对多个画面内物体，分析 depth 变化的来源
results = []

for si in range(30):
    all_hook.clear()
    sample = ds[si]
    frames = sample['video'].unsqueeze(0).cuda()
    
    with torch.no_grad():
        feat_with_grid = model._encode_features(frames)
        out = model(frames)
    
    all_slots = torch.cat([out['slots']['corrected'], out['slots']['target']], dim=1)[0]
    
    if len(all_hook) < 16:
        continue
    
    # grid 坐标
    grid = feat_with_grid[0, 0, :, -2:].cpu().numpy()  # (256, 2), 范围 [-1,1]
    
    N = all_slots.shape[1]
    for s in range(N):
        depth = all_slots[:, s, app_dim+2].cpu().numpy()
        pos_x = all_slots[:, s, app_dim].cpu().numpy()
        pos_y = all_slots[:, s, app_dim+1].cpu().numpy()
        
        if np.mean(depth) >= depth_max:
            continue
        
        total_move = np.sqrt((pos_x[-1]-pos_x[0])**2 + (pos_y[-1]-pos_y[0])**2)
        if total_move < 0.1:
            continue
        
        # 严格在画面内
        if np.abs(pos_x).max() >= bnd_thresh or np.abs(pos_y).max() >= bnd_thresh:
            continue
        
        # 对每帧手动计算 depth
        for t in range(16):
            attn_t = all_hook[t]['attn'][0, s].numpy()  # (256,)
            pos_t = np.array([pos_x[t], pos_y[t]])
            
            spread = np.sum((grid - pos_t)**2, axis=-1)
            depth_computed = np.sqrt(np.sum(attn_t * spread))
            
            # 分解: depth² = sum(attn * spread)
            # spread = ||grid - pos||² 依赖于 pos
            # 如果 pos 不变，depth 变化纯粹由 attn 变化导致
            
            results.append({
                'sample': si, 'slot': s, 't': t,
                'depth_actual': depth[t],
                'depth_computed': depth_computed,
                'attn_entropy': -np.sum(attn_t * np.log(attn_t + 1e-10)),
                'attn_max': attn_t.max(),
                'attn_concentration': attn_t.max() / (attn_t.mean() + 1e-10),
            })

print(f"Total data points: {len(results)}")

# 验证 computed vs actual
depths_actual = np.array([r['depth_actual'] for r in results])
depths_computed = np.array([r['depth_computed'] for r in results])
corr = np.corrcoef(depths_actual, depths_computed)[0, 1]
print(f"Computed vs actual depth correlation: {corr:.6f}")
print(f"Computed range: [{depths_computed.min():.4f}, {depths_computed.max():.4f}]")
print(f"Actual range: [{depths_actual.min():.4f}, {depths_actual.max():.4f}]")

# depth 变化 vs attn 变化
# 对每个 (sample, slot)，计算相邻帧的 depth 变化和 attn 变化
from itertools import groupby
slot_groups = {}
for r in results:
    key = (r['sample'], r['slot'])
    if key not in slot_groups:
        slot_groups[key] = []
    slot_groups[key].append(r)

depth_deltas = []
attn_entropy_deltas = []
attn_max_deltas = []

for key, group in slot_groups.items():
    group.sort(key=lambda r: r['t'])
    for i in range(len(group)-1):
        dd = group[i+1]['depth_actual'] - group[i]['depth_actual']
        de = group[i+1]['attn_entropy'] - group[i]['attn_entropy']
        dm = group[i+1]['attn_max'] - group[i]['attn_max']
        depth_deltas.append(abs(dd))
        attn_entropy_deltas.append(abs(de))
        attn_max_deltas.append(abs(dm))

# Remove the duplicate code block below

depth_deltas = np.array(depth_deltas)
attn_entropy_deltas = np.array(attn_entropy_deltas)
attn_max_deltas = np.array(attn_max_deltas)

print(f"\nDepth delta vs attention delta (in-frame only):")
print(f"  |Δdepth| mean={depth_deltas.mean():.6f} median={np.median(depth_deltas):.6f}")
print(f"  |Δentropy| mean={attn_entropy_deltas.mean():.6f}")
print(f"  |Δattn_max| mean={attn_max_deltas.mean():.6f}")

corr1 = np.corrcoef(depth_deltas, attn_entropy_deltas)[0, 1]
corr2 = np.corrcoef(depth_deltas, attn_max_deltas)[0, 1]
print(f"\n  Corr(|Δdepth|, |Δentropy|): {corr1:.4f}")
print(f"  Corr(|Δdepth|, |Δattn_max|): {corr2:.4f}")

# 控制变量: 如果 pos 完全不变，depth 变化纯粹来自 attn 变化
# 用第一帧的 pos 重新计算所有帧的 depth
print("\n=== Controlled: fixed pos from frame 0 ===")
for key, group in list(slot_groups.items())[:5]:
    group.sort(key=lambda r: r['t'])
    si, sl = key
    
    # 用 frame 0 的 pos
    pos_0 = np.array([group[0]['depth_actual']])  # placeholder
    
    # 重新获取 grid 和 attn
    pos_x_0 = all_slots[group[0]['t'], sl, app_dim].item()
    pos_y_0 = all_slots[group[0]['t'], sl, app_dim+1].item()
    pos_fixed = np.array([pos_x_0, pos_y_0])
    
    for r in group:
        attn_t = all_hook[r['t']]['attn'][0, sl].numpy()
        spread_fixed = np.sum((grid - pos_fixed)**2, axis=-1)
        depth_fixed = np.sqrt(np.sum(attn_t * spread_fixed))
        r['depth_fixed_pos'] = depth_fixed
    
    actual_depths = [r['depth_actual'] for r in group]
    fixed_depths = [r['depth_fixed_pos'] for r in group]
    
    print(f"  s{si}_sl{sl}: actual={[f'{d:.4f}' for d in actual_depths[:8]]}")
    print(f"           fixed ={[f'{d:.4f}' for d in fixed_depths[:8]]}")
    diffs = [f'{a-f:.4f}' for a,f in zip(actual_depths[:8], fixed_depths[:8])]
    print(f"           diff  ={diffs}")

handle.remove()
