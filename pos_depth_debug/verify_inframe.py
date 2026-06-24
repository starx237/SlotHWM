"""
直接验证: depth 跳变是否由注意力分布变化导致
depth = sqrt(sum(attn * spread))
当注意力从集中变分散，depth 变大；反之变小

关键问题: 即使物体完全在画面内，depth 是否也会因为注意力分布微变而跳变？
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

# Hook: 收集每次 slot_attention 调用的 attn 和 slots
all_hook_data = []

def hook_fn(module, input, output):
    if isinstance(output, tuple) and len(output) == 2:
        slots, attn = output
        all_hook_data.append({
            'attn': attn.detach().cpu(),      # (B, N_slots, N_pixels)
            'slots': slots.detach().cpu(),     # (B, N_slots, D)
        })

handle = model.slot_attention.register_forward_hook(hook_fn)

# 扫描多个样本，找完全在画面内、无遮挡的物体
# 严格筛选: 物体始终在画面内，pos 线性，且 depth 变化小
results = []

for si in range(100):
    all_hook_data.clear()
    
    sample = ds[si]
    frames = sample['video'].unsqueeze(0).cuda()
    
    with torch.no_grad():
        out = model(frames)
    
    corrected = out['slots']['corrected']
    target = out['slots']['target']
    all_slots = torch.cat([corrected, target], dim=1)[0]
    
    if len(all_hook_data) < 16:
        continue
    
    N_slots = all_slots.shape[1]
    
    for s in range(N_slots):
        depth = all_slots[:, s, app_dim+2].cpu().numpy()
        pos_x = all_slots[:, s, app_dim].cpu().numpy()
        pos_y = all_slots[:, s, app_dim+1].cpu().numpy()
        
        if np.mean(depth) >= depth_max:
            continue
        
        total_move = np.sqrt((pos_x[-1]-pos_x[0])**2 + (pos_y[-1]-pos_y[0])**2)
        if total_move < 0.05:
            continue
        
        # 严格: 始终在画面内
        ever_out = (np.abs(pos_x).max() >= bnd_thresh) or (np.abs(pos_y).max() >= bnd_thresh)
        
        # depth 变化
        dd = np.abs(np.diff(depth))
        max_jump = dd.max()
        max_jump_idx = dd.argmax()
        depth_range = depth.max() - depth.min()
        
        # pos 线性度
        fr = np.arange(16)
        r2_px = 1 - np.sum((pos_x-np.polyval(np.polyfit(fr,pos_x,1),fr))**2)/(np.sum((pos_x-pos_x.mean())**2)+1e-8) if pos_x.std()>1e-6 else 1.0
        r2_py = 1 - np.sum((pos_y-np.polyval(np.polyfit(fr,pos_y,1),fr))**2)/(np.sum((pos_y-pos_y.mean())**2)+1e-8) if pos_y.std()>1e-6 else 1.0
        
        results.append({
            'sample': si, 'slot': s,
            'pos_r2': min(r2_px, r2_py),
            'max_jump': max_jump, 'max_jump_idx': max_jump_idx,
            'depth_range': depth_range, 'total_move': total_move,
            'ever_out': ever_out,
            'depth': depth,
        })

handle.remove()

# 分析
in_frame = [r for r in results if not r['ever_out']]
out_frame = [r for r in results if r['ever_out']]

print(f"Total: {len(results)}, in-frame: {len(in_frame)}, out-frame: {len(out_frame)}")

# 对比 in-frame vs out-frame 的 depth 稳定性
if in_frame:
    in_jumps = [r['max_jump'] for r in in_frame]
    in_ranges = [r['depth_range'] for r in in_frame]
    print(f"\nIn-frame objects (n={len(in_frame)}):")
    print(f"  max_jump: mean={np.mean(in_jumps):.4f} median={np.median(in_jumps):.4f} max={np.max(in_jumps):.4f}")
    print(f"  depth_range: mean={np.mean(in_ranges):.4f} median={np.median(in_ranges):.4f}")

if out_frame:
    out_jumps = [r['max_jump'] for r in out_frame]
    out_ranges = [r['depth_range'] for r in out_frame]
    print(f"\nOut-frame objects (n={len(out_frame)}):")
    print(f"  max_jump: mean={np.mean(out_jumps):.4f} median={np.median(out_jumps):.4f} max={np.max(out_jumps):.4f}")
    print(f"  depth_range: mean={np.mean(out_ranges):.4f} median={np.median(out_ranges):.4f}")

# 再看: 严格在画面内 且 pos 线性的物体，depth 是否仍然不稳定？
strict_in = [r for r in in_frame if r['pos_r2'] > 0.85 and r['total_move'] > 0.15]
print(f"\nStrict in-frame (pos_R²>0.85, move>0.15, n={len(strict_in)}):")
if strict_in:
    sj = [r['max_jump'] for r in strict_in]
    sr = [r['depth_range'] for r in strict_in]
    print(f"  max_jump: mean={np.mean(sj):.4f} median={np.median(sj):.4f}")
    print(f"  depth_range: mean={np.mean(sr):.4f} median={np.median(sr):.4f}")
    for r in sorted(strict_in, key=lambda x: -x['max_jump'])[:5]:
        print(f"    s{r['sample']}_sl{r['slot']}: jump={r['max_jump']:.4f}@t={r['max_jump_idx']} range={r['depth_range']:.4f} move={r['total_move']:.2f} depth={[f'{d:.4f}' for d in r['depth']]}")

# 核心问题: 即使物体完全在画面内，depth 仍然跳变吗？
# 如果是，那问题就是 ISA 编码器本身的 depth 不稳定性，而非遮挡
