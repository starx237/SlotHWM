"""
找到 pos 匀速但 depth 震荡/非单调的物体，可视化 slot 分解
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
rollout = cfg.rollout_frames

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
loader = ds.get_dataloader(batch_size=1, shuffle=False, num_workers=0)

# 收集数据
candidates = []
N_SAMPLES = 200

for i in range(N_SAMPLES):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    with torch.no_grad():
        out = model(frames)
    
    corrected = out['slots']['corrected']
    target = out['slots']['target']
    all_slots = torch.cat([corrected, target], dim=1)[0]  # (16, N, D)
    
    N = all_slots.shape[1]
    for s in range(N):
        depth = all_slots[:, s, app_dim+2].cpu().numpy()
        pos_x = all_slots[:, s, app_dim].cpu().numpy()
        pos_y = all_slots[:, s, app_dim+1].cpu().numpy()
        
        # 跳过背景
        if np.mean(depth) >= depth_max:
            continue
        
        # 跳过静止物体
        total_move = np.sqrt((pos_x[-1]-pos_x[0])**2 + (pos_y[-1]-pos_y[0])**2)
        if total_move < 0.15:
            continue
        
        # pos 线性度
        fr = np.arange(16)
        if pos_x.std() < 1e-6 and pos_y.std() < 1e-6:
            continue
        
        r2_px = 1 - np.sum((pos_x-np.polyval(np.polyfit(fr,pos_x,1),fr))**2)/(np.sum((pos_x-pos_x.mean())**2)+1e-8) if pos_x.std()>1e-6 else 1.0
        r2_py = 1 - np.sum((pos_y-np.polyval(np.polyfit(fr,pos_y,1),fr))**2)/(np.sum((pos_y-pos_y.mean())**2)+1e-8) if pos_y.std()>1e-6 else 1.0
        pos_r2 = min(r2_px, r2_py)
        
        # depth 单调性分析
        d_delta = np.diff(depth)
        n_sign_changes = np.sum(np.diff(np.sign(d_delta)) != 0)
        sign_change_ratio = n_sign_changes / max(len(d_delta)-1, 1)
        
        # depth 是否单调或单峰
        # 单调: delta 始终同号
        is_monotone = np.all(d_delta >= -1e-5) or np.all(d_delta <= 1e-5)
        
        # 单峰: 先增后减或先减后增，最多1个极值点
        extrema = 0
        for k in range(1, len(d_delta)):
            if d_delta[k] * d_delta[k-1] < 0:
                extrema += 1
        is_unimodal = extrema <= 1
        
        # depth 线性度
        r2_depth = 1 - np.sum((depth-np.polyval(np.polyfit(fr,depth,1),fr))**2)/(np.sum((depth-depth.mean())**2)+1e-8) if depth.std()>1e-6 else 1.0
        
        # 筛选: pos 线性好(R²>0.9), depth 非单调非单峰 或 depth R²很低
        if pos_r2 > 0.9 and (not is_monotone and not is_unimodal or r2_depth < 0.3):
            candidates.append({
                'sample': i,
                'slot': s,
                'pos_r2': pos_r2,
                'depth_r2': r2_depth,
                'is_monotone': is_monotone,
                'is_unimodal': is_unimodal,
                'n_extrema': extrema,
                'sign_change_ratio': sign_change_ratio,
                'total_move': total_move,
                'pos_x': pos_x.copy(),
                'pos_y': pos_y.copy(),
                'depth': depth.copy(),
            })

print(f"Found {len(candidates)} candidates (pos linear, depth complex)")

# 按depth复杂度排序（sign_change多、R2低的排前面）
candidates.sort(key=lambda c: (-c['n_extrema'], c['depth_r2']))

# 选 top 8 做可视化
selected = candidates[:8]
print(f"Selected top {len(selected)} for visualization:")
for j, c in enumerate(selected):
    print(f"  [{j}] s{c['sample']}_sl{c['slot']}: pos_R²={c['pos_r2']:.3f} depth_R²={c['depth_r2']:.3f} extrema={c['n_extrema']} monotone={c['is_monotone']} unimodal={c['is_unimodal']}")

# 可视化: 每个 candidate 一行，左=pos轨迹+depth曲线，右=slot重建图
fig, axes = plt.subplots(len(selected), 4, figsize=(24, 4*len(selected)))
if len(selected) == 1:
    axes = axes.reshape(1, -1)
fig.suptitle('Objects with Linear Pos but Complex Depth', fontsize=14, y=1.02)

for j, c in enumerate(selected):
    si = c['sample']
    sl = c['slot']
    
    # 重新获取该样本
    torch.manual_seed(0)  # 确保 DataLoader 顺序一致
    # 直接从 dataset 获取
    sample = ds[si]
    frames = sample['video'].unsqueeze(0).cuda()
    
    with torch.no_grad():
        out = model(frames)
    
    corrected = out['slots']['corrected']
    target = out['slots']['target']
    all_slots = torch.cat([corrected, target], dim=1)[0]  # (16, N, D)
    
    # 重建每帧的 slot mask
    recon = out['recon_rollout'] if 'recon_rollout' in out else None
    
    # pos_x + pos_y 轨迹
    fr = np.arange(16)
    axes[j, 0].plot(fr, c['pos_x'], 'r-o', markersize=3, label='pos_x')
    axes[j, 0].plot(fr, c['pos_y'], 'b-o', markersize=3, label='pos_y')
    axes[j, 0].axvline(x=5.5, color='gray', linestyle='--', alpha=0.5, label='burnin|rollout')
    axes[j, 0].set_title(f's{si}_sl{sl} pos (R²={c["pos_r2"]:.2f})')
    axes[j, 0].legend(fontsize=7)
    axes[j, 0].grid(True, alpha=0.3)
    
    # depth 曲线
    axes[j, 1].plot(fr, c['depth'], 'g-o', markersize=3, label='depth')
    # 标注极值点
    d = c['depth']
    for k in range(1, len(d)-1):
        if (d[k] > d[k-1] and d[k] > d[k+1]) or (d[k] < d[k-1] and d[k] < d[k+1]):
            axes[j, 1].plot(k, d[k], 'rv', markersize=8)
    axes[j, 1].axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)
    axes[j, 1].set_title(f'depth (R²={c["depth_r2"]:.2f}, extrema={c["n_extrema"]})')
    axes[j, 1].grid(True, alpha=0.3)
    
    # Slot alpha mask 可视化 - 4帧 (0, 5, 10, 15)
    # 用 decoder 重建每帧的单个 slot
    slot_data = all_slots  # (16, N, D)
    decoder = model.decoder
    
    for k, t_idx in enumerate([0, 5, 10, 15]):
        if t_idx >= 16:
            continue
        ax = axes[j, 2+k] if 2+k < 4 else axes[j, 3]
        
        # 获取该帧该 slot 的 alpha mask
        # 需要解码单个 slot
        s_data = slot_data[t_idx, sl:sl+1]  # (1, D)
        with torch.no_grad():
            # decoder 需要 (1, N, D) 输入
            # 创建只有目标 slot 的输入
            single_slot = s_data.unsqueeze(0)  # (1, 1, D)
            recon_single = decoder(single_slot)  # (1, 1, 3, H, W) or similar
            
            # 获取 alpha
            if hasattr(decoder, 'decode_slots'):
                pass  # 使用下面的方法
            
            # 直接用 decoder 的 forward 获取 mask
            # ISASpatialBroadcastDecoder forward 输出 (B, N, 3+1, H, W)
            # 尝试直接调用
            try:
                out_dec = decoder(single_slot)
                if out_dec.shape[-1] == 64:  # (B, N, 4, H, W)
                    alpha = out_dec[0, 0, 3].cpu().numpy()  # alpha mask
                    rgb = out_dec[0, 0, :3].cpu().numpy()  # RGB
                    # 显示 RGB * alpha
                    img = rgb.transpose(1, 2, 0) * alpha[:,:,np.newaxis]
                    img = np.clip(img, 0, 1)
                    ax.imshow(img)
                    ax.set_title(f't={t_idx} sl{sl}', fontsize=8)
                else:
                    ax.text(0.5, 0.5, f't={t_idx}', ha='center')
            except:
                ax.text(0.5, 0.5, f't={t_idx} decode err', ha='center', fontsize=8)
        
        ax.axis('off')

plt.tight_layout()
plt.savefig('/autodl-fs/data/SlotHWM/pos_depth_debug/pos_linear_depth_complex.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: pos_depth_debug/pos_linear_depth_complex.png")
