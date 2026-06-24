"""
实验4: Slot Permutation 问题
核心假设: target slots 在 rollout 期间可能和 burnin/predicted 的 slot 排列不一致
因为 target 是重新跑 ISA 得到的，ISA 的 slot ordering 可能每帧都不同
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

with torch.no_grad():
    out = model(frames)

app_dim = cfg.appearance_dim
rollout = cfg.rollout_frames
burnin = cfg.burnin_frames

# 提取各帧的 appearance（用来判断 slot identity）
corrected = out['slots']['corrected']  # (B, burnin, N, D)
target = out['slots']['target']  # (B, rollout, N, D)

# Burnin 最后一帧和 target 第一帧应该对应相同的场景
# 如果 slot ordering 一致，那么 burnin[-1] 和 target[0] 的对应 slot 应该相似
burnin_last = corrected[:, -1]  # (B, N, D)
target_first = target[:, 0]  # (B, N, D)

print("=== Slot identity check: burnin_last vs target_frame0 ===")
# 用 appearance 的 cosine similarity 判断是否同一个 slot
for b in range(1):
    app_burnin = burnin_last[b, :, :app_dim]  # (N, 64)
    app_target = target_first[b, :, :app_dim]  # (N, 64)
    
    # 计算两两之间的 cosine similarity
    app_b_norm = app_burnin / (app_burnin.norm(dim=-1, keepdim=True) + 1e-8)
    app_t_norm = app_target / (app_target.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = torch.mm(app_b_norm, app_t_norm.t())  # (N, N)
    
    print(f"\nBatch {b}: Cosine similarity matrix (burnin_last x target_frame0):")
    for i in range(cos_sim.shape[0]):
        row = "  ".join(f"{cos_sim[i,j]:.3f}" for j in range(cos_sim.shape[1]))
        print(f"  burnin_slot{i}: [{row}]")
    
    # 找最佳匹配
    for i in range(cos_sim.shape[0]):
        best_j = cos_sim[i].argmax().item()
        best_sim = cos_sim[i, best_j].item()
        print(f"  burnin_slot{i} → target_slot{best_j} (cos={best_sim:.3f})")

# 更详细：检查连续 rollout 帧之间的 slot ordering 是否一致
print("\n=== Slot ordering consistency across rollout frames ===")
for b in range(1):
    for t in range(min(3, rollout-1)):
        app_t0 = target[b, t, :, :app_dim]
        app_t1 = target[b, t+1, :, :app_dim]
        a0_norm = app_t0 / (app_t0.norm(dim=-1, keepdim=True) + 1e-8)
        a1_norm = app_t1 / (app_t1.norm(dim=-1, keepdim=True) + 1e-8)
        cos = torch.mm(a0_norm, a1_norm.t())
        
        # 看 diagonal 是否最大
        diag_is_max = True
        for i in range(cos.shape[0]):
            if cos[i].argmax() != i:
                diag_is_max = False
                break
        
        perm_count = sum(1 for i in range(cos.shape[0]) if cos[i].argmax() != i)
        print(f"  frame {t}→{t+1}: diagonal_max={diag_is_max}, permuted_slots={perm_count}/{cos.shape[0]}")
        for i in range(cos.shape[0]):
            best_j = cos[i].argmax().item()
            if best_j != i:
                print(f"    slot{i} → slot{best_j} (SWAP! cos={cos[i,best_j]:.3f})")

# 关键指标: 如果有 permutation，计算 "最优匹配后" 的 MSE vs "直接 MSE"
print("\n=== MSE comparison: direct vs Hungarian-matched ===")
from scipy.optimize import linear_sum_assignment
import numpy as np

for b in range(1):
    # 对 rollout 的每一帧，计算 direct MSE 和 Hungarian-matched MSE
    direct_depth_mse = []
    matched_depth_mse = []
    direct_pos_mse = []
    matched_pos_mse = []
    
    for t in range(rollout):
        pred_dyn = out['slots']['predicted'][b, t, :, app_dim:]  # (N, 3) - 当前就是复制burnin_last
        target_dyn = target[b, t, :, app_dim:]  # (N, 3)
        
        # direct MSE
        d_mse = ((pred_dyn[:, 2] - target_dyn[:, 2])**2).mean()
        p_mse = ((pred_dyn[:, :2] - target_dyn[:, :2])**2).mean()
        direct_depth_mse.append(d_mse.item())
        direct_pos_mse.append(p_mse.item())
        
        # Hungarian matching based on appearance similarity
        pred_app = out['slots']['predicted'][b, t, :, :app_dim]
        target_app = target[b, t, :, :app_dim]
        p_norm = pred_app / (pred_app.norm(dim=-1, keepdim=True) + 1e-8)
        t_norm = target_app / (target_app.norm(dim=-1, keepdim=True) + 1e-8)
        cost = -torch.mm(p_norm, t_norm.t()).cpu().numpy()  # negative cos = cost
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # matched MSE
        matched_target_dyn = target_dyn[col_ind]
        m_d_mse = ((pred_dyn[:, 2] - matched_target_dyn[:, 2])**2).mean()
        m_p_mse = ((pred_dyn[:, :2] - matched_target_dyn[:, :2])**2).mean()
        matched_depth_mse.append(m_d_mse.item())
        matched_pos_mse.append(m_p_mse.item())
    
    print(f"  Direct  depth MSE per frame: {['%.6f' % x for x in direct_depth_mse]}")
    print(f"  Matched depth MSE per frame: {['%.6f' % x for x in matched_depth_mse]}")
    print(f"  Direct  pos MSE per frame:   {['%.6f' % x for x in direct_pos_mse]}")
    print(f"  Matched pos MSE per frame:   {['%.6f' % x for x in matched_pos_mse]}")
    print(f"  Direct  depth total: {sum(direct_depth_mse)/len(direct_depth_mse):.6f}")
    print(f"  Matched depth total: {sum(matched_depth_mse)/len(matched_depth_mse):.6f}")
    print(f"  Direct  pos total:   {sum(direct_pos_mse)/len(direct_pos_mse):.6f}")
    print(f"  Matched pos total:   {sum(matched_pos_mse)/len(matched_pos_mse):.6f}")
