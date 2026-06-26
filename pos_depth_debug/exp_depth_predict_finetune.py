"""
Depth->Alpha Spread Predictor 微调实验

核心思路:
- 训练一个只看 depth 的轻量 predictor: depth -> predicted_spread
- Loss: predicted_spread vs actual alpha spread (from decoder)
- 梯度同时流过 predictor 和 ISA (depth 有梯度)
- ISA 被迫把大小信息编码到 depth 中, 因为 predictor 只看 depth

从 isa_single_poscosloss_40000.pt 开始, 单帧微调
所有模块(除 predictor 外)正常接收梯度
burnin=1, detach_cospos=False, continue_pretrain=False

监控: burnin_loss, pos_loss, cos_loss, depth_pred_loss, R²(depth vs spread)
"""
import os, sys, warnings, copy
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset
from models.misc import create_coordinate_grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CKPT_PATH = 'good_checkpoints/isa_single_poscosloss_40000.pt'
SAVE_DIR = 'pos_depth_debug/depth_predict_finetune'
os.makedirs(SAVE_DIR, exist_ok=True)

with open('config/pretrain_obj3d.yaml') as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))

# 单帧, 不冻结, 不detach
cfg.continue_pretrain = False
cfg.freeze_slot = False
cfg.burnin_frames = 1
cfg.detach_cospos = False

app_dim = cfg.appearance_dim
depth_max = getattr(cfg, 'depth_max', 0.30)
bnd_threshold = getattr(cfg, 'bnd_threshold', 0.75)


class DepthSpreadPredictor(nn.Module):
    """只看 depth 预测 alpha spread 的轻量网络
    目的: 迫使 ISA 把大小信息编码到 depth 中
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # 初始化: 输出接近 linear(depth)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, depth):
        return self.net(depth.unsqueeze(-1)).squeeze(-1)


def compute_alpha_spread_batched(alpha_2d):
    """alpha_2d: (B, N, H, W) -> (B, N) spread"""
    B, N, H, W = alpha_2d.shape
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    gx = gx.unsqueeze(0).unsqueeze(0).expand(B, N, H, W).to(alpha_2d.device)
    gy = gy.unsqueeze(0).unsqueeze(0).expand(B, N, H, W).to(alpha_2d.device)
    a_sum = alpha_2d.sum(dim=[-2, -1], keepdim=True) + 1e-8
    a_norm = alpha_2d / a_sum
    cx = (a_norm * gx).sum(dim=[-2, -1])
    cy = (a_norm * gy).sum(dim=[-2, -1])
    spread = torch.sqrt((a_norm * ((gx - cx.unsqueeze(-1).unsqueeze(-1)) ** 2 +
                                    (gy - cy.unsqueeze(-1).unsqueeze(-1)) ** 2)).sum(dim=[-2, -1]))
    return spread


def eval_r2(m, ds, n_samples=10):
    all_depths = []; all_spreads = []
    m.eval()
    with torch.no_grad():
        for si in range(min(n_samples, len(ds))):
            sample = ds[si]
            frames = sample['video'].unsqueeze(0).cuda()
            feat = m._encode_features(frames)
            slots = None; gru2_hidden = None; prev_app = None
            for t in range(16):
                feat_t = feat[:, t]
                if t > 0 and slots is not None:
                    new_app, gru2_hidden = m._gru2_step(prev_app, gru2_hidden)
                    slots = torch.cat([new_app, slots[:, :, -3:-1].contiguous(),
                                       slots[:, :, -1:].contiguous()], dim=-1)
                slots, _ = m._sa(feat_t, slots, t)
                prev_app = slots[:, :, :-3].detach()
                if t == 0:
                    BN = prev_app.shape[0] * prev_app.shape[1]
                    gru2_hidden = torch.zeros(BN, m.gru2_hidden_dim, device=frames.device)
                    gru2_hidden = m.gru2(prev_app.reshape(-1, m.appearance_dim),
                                          gru2_hidden.reshape(-1, m.gru2_hidden_dim))
                recon, alpha, _ = m.decoder(slots, return_rgb=True)
                for s in range(6):
                    d = slots[0, s, app_dim + 2].item()
                    px = slots[0, s, app_dim].item(); py = slots[0, s, app_dim + 1].item()
                    if d >= depth_max: continue
                    if abs(px) >= bnd_threshold or abs(py) >= bnd_threshold: continue
                    a = alpha[0, s, 0]
                    if a.sum() < 1.0: continue
                    a_norm = a / a.sum()
                    gy, gx = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing='ij')
                    gx = gx.to(a.device); gy = gy.to(a.device)
                    cx = (a_norm * gx).sum(); cy = (a_norm * gy).sum()
                    sp = torch.sqrt((a_norm * ((gx - cx) ** 2 + (gy - cy) ** 2)).sum()).item()
                    if sp > 0.01:
                        all_depths.append(d); all_spreads.append(sp)
    all_depths = np.array(all_depths); all_spreads = np.array(all_spreads)
    mask = all_depths > 0.04
    if mask.sum() < 10: return 0.0
    coef = np.polyfit(all_depths[mask], all_spreads[mask], 1)
    y_pred = np.polyval(coef, all_depths[mask])
    r2 = 1 - ((all_spreads[mask] - y_pred) ** 2).sum() / (
            (all_spreads[mask] - all_spreads[mask].mean()) ** 2).sum()
    return r2


def eval_predictor(m, predictor, ds, n_samples=10):
    all_pred = []; all_true = []
    m.eval(); predictor.eval()
    with torch.no_grad():
        for si in range(min(n_samples, len(ds))):
            sample = ds[si]; frames = sample['video'].unsqueeze(0).cuda()
            feat = m._encode_features(frames)
            slots = None; gru2_hidden = None; prev_app = None
            for t in range(16):
                feat_t = feat[:, t]
                if t > 0 and slots is not None:
                    new_app, gru2_hidden = m._gru2_step(prev_app, gru2_hidden)
                    slots = torch.cat([new_app, slots[:, :, -3:-1].contiguous(),
                                       slots[:, :, -1:].contiguous()], dim=-1)
                slots, _ = m._sa(feat_t, slots, t)
                prev_app = slots[:, :, :-3].detach()
                if t == 0:
                    BN = prev_app.shape[0] * prev_app.shape[1]
                    gru2_hidden = torch.zeros(BN, m.gru2_hidden_dim, device=frames.device)
                    gru2_hidden = m.gru2(prev_app.reshape(-1, m.appearance_dim),
                                          gru2_hidden.reshape(-1, m.gru2_hidden_dim))
                recon, alpha, _ = m.decoder(slots, return_rgb=True)
                alpha_2d = alpha.squeeze(2)
                true_spread = compute_alpha_spread_batched(alpha_2d)
                depth = slots[0, :, app_dim + 2]
                pred_spread = predictor(depth.unsqueeze(0))
                for s in range(6):
                    d = depth[s].item()
                    px = slots[0, s, app_dim].item(); py = slots[0, s, app_dim + 1].item()
                    if d >= depth_max: continue
                    if abs(px) >= bnd_threshold or abs(py) >= bnd_threshold: continue
                    sp = true_spread[0, s].item()
                    if sp < 0.01: continue
                    all_pred.append(pred_spread[0, s].item())
                    all_true.append(sp)
    all_pred = np.array(all_pred); all_true = np.array(all_true)
    if len(all_pred) < 10: return 0.0
    ss_tot = ((all_true - all_true.mean()) ** 2).sum()
    ss_res = ((all_true - all_pred) ** 2).sum()
    return 1 - ss_res / ss_tot


def train():
    ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                                      num_workers=2, pin_memory=True)

    # Load model
    m = SlotDynamicsModel(cfg).cuda()
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    sd = m.state_dict(); ld = {}
    for mk in sd:
        mc = mk.replace('_orig_mod.', '')
        for ck in ckpt['model']:
            cc = ck.replace('_orig_mod.', '')
            if cc == mc and ckpt['model'][ck].shape == sd[mk].shape:
                ld[mk] = ckpt['model'][ck]; break
    m.load_state_dict(ld, strict=False)

    # Freeze predictor module (dynamics predictor, not our depth predictor)
    for p in m.predictor.parameters():
        p.requires_grad_(False)

    predictor = DepthSpreadPredictor(hidden=32).cuda()

    # Phase 1: 预训练 predictor (freeze ISA)
    print("=== Phase 1: Pretrain depth predictor (ISA frozen) ===")
    for p in m.parameters():
        p.requires_grad_(False)
    opt_pred = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    for step in range(500):
        batch = next(iter(dl))
        frames = batch['video'].cuda()
        m.eval(); predictor.train(); opt_pred.zero_grad()

        with torch.no_grad():
            out = m(frames)
        B = out["slots"]["corrected"].shape[0]; N = out["slots"]["corrected"].shape[2]
        burnin_T = out["slots"]["corrected"].shape[1]

        total_loss = 0; n_pairs = 0
        for t in range(burnin_T):
            slots_t = out["slots"]["corrected"][:, t].detach()
            alpha_t = out["alpha"][:, :, t]
            if alpha_t.dim() == 5: alpha_2d = alpha_t.squeeze(2)
            else: alpha_2d = alpha_t
            true_spread = compute_alpha_spread_batched(alpha_2d).detach()
            depth = slots_t[:, :, app_dim + 2]
            pred_spread = predictor(depth)
            px = slots_t[:, :, app_dim]; py = slots_t[:, :, app_dim + 1]
            fg = (depth < depth_max); in_bnd = (px.abs() < bnd_threshold) & (py.abs() < bnd_threshold)
            mask = fg & in_bnd
            if mask.any():
                total_loss = total_loss + F.mse_loss(pred_spread[mask], true_spread[mask])
                n_pairs += mask.sum().item()

        if n_pairs > 0:
            total_loss.backward(); opt_pred.step()
        if step % 100 == 0:
            print(f"  Step {step}: loss={total_loss.item():.6f}")

    r2_pred_init = eval_predictor(m, predictor, ds, n_samples=5)
    print(f"  Predictor R² after pretrain: {r2_pred_init:.4f}")

    # 解冻 ISA
    for p in m.parameters():
        if not any(p is pp for pp in m.predictor.parameters()):
            p.requires_grad_(True)
    for p in m.predictor.parameters():
        p.requires_grad_(True)

    # Phase 2: 联合训练
    print("\n=== Phase 2: Joint training (ISA + predictor) ===")
    opt_main = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=cfg.learning_rate)
    opt_pred2 = torch.optim.Adam(predictor.parameters(), lr=1e-4)

    r2_base = eval_r2(m, ds)
    print(f"Baseline R²(depth→spread): {r2_base:.4f}")

    depth_weight = 0.5
    n_steps = 2000
    log_every = 50; eval_every = 200
    history = {'step': [], 'recon': [], 'pos': [], 'cos': [], 'depth_pred': [], 'r2': [], 'r2_pred': []}

    step = 0
    for epoch in range(100):
        for batch in dl:
            if step >= n_steps: break
            frames = batch['video'].cuda()
            m.train(); predictor.train()
            opt_main.zero_grad(); opt_pred2.zero_grad()

            out = m(frames)

            # Recon
            dec_size = out["outputs"]["video_burnin"].shape[-1]
            target_size = frames.shape[-1]
            video_burnin = out["outputs"]["video_burnin"]
            if dec_size != target_size:
                b = frames.shape[0]
                video_burnin = F.interpolate(video_burnin.reshape(-1, 3, dec_size, dec_size),
                                              size=target_size, mode='bilinear').reshape(b, -1, 3, target_size, target_size)
            target_burnin = frames[:, :cfg.burnin_frames]
            recon_loss = F.mse_loss(video_burnin, target_burnin)

            # Pos + Cos
            B = out["slots"]["corrected"].shape[0]; N = out["slots"]["corrected"].shape[2]
            burnin_T = out["slots"]["corrected"].shape[1]
            loss_pos_list = []; loss_cos_list = []
            for t in range(burnin_T):
                slots_t = out["slots"]["corrected"][:, t]
                if cfg.lambda_cos > 0:
                    attn_t = out["attn"][:, t]
                    attn_dot = torch.bmm(attn_t, attn_t.transpose(1, 2))
                    diag = torch.eye(N, device=slots_t.device)
                    loss_cos_t = (attn_dot * (1 - diag.unsqueeze(0))).sum(dim=[-2, -1]).mean() / (N * (N - 1))
                    loss_cos_list.append(loss_cos_t)
                if cfg.lambda_pos > 0:
                    alpha_t = out["alpha"][:, :, t]
                    Sp = slots_t[:, :, -3:-1]
                    H, W = alpha_t.shape[-2:]
                    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H, device=slots_t.device),
                                             torch.linspace(-1, 1, W, device=slots_t.device), indexing='ij')
                    a = alpha_t.squeeze(2); denom = a.sum(dim=[-2, -1]) + 1e-8
                    cx = (a * gx).sum(dim=[-2, -1]) / denom; cy = (a * gy).sum(dim=[-2, -1]) / denom
                    centroid = torch.stack([cx, cy], dim=-1)
                    dominant = a.argmax(dim=1)
                    owned = torch.stack([(dominant == j).sum(dim=[-2, -1]) for j in range(N)], dim=-1).float()
                    fg_mask = (owned > 20) & (owned < 0.6 * H * W)
                    loss_pos_t = F.mse_loss(centroid[fg_mask], Sp[fg_mask]) if fg_mask.any() else torch.tensor(
                        0.0, device=slots_t.device)
                    loss_pos_list.append(loss_pos_t)
            loss_pos = torch.stack(loss_pos_list).mean() if loss_pos_list else torch.tensor(0.0)
            loss_cos = torch.stack(loss_cos_list).mean() if loss_cos_list else torch.tensor(0.0)

            # Depth prediction loss: predictor(depth) ≈ actual_spread
            depth_pred_loss = torch.tensor(0.0, device=frames.device)
            for t in range(burnin_T):
                slots_t = out["slots"]["corrected"][:, t]
                alpha_t = out["alpha"][:, :, t]
                if alpha_t.dim() == 5: alpha_2d = alpha_t.squeeze(2)
                else: alpha_2d = alpha_t
                true_spread = compute_alpha_spread_batched(alpha_2d).detach()
                depth = slots_t[:, :, app_dim + 2]
                pred_spread = predictor(depth)
                px = slots_t[:, :, app_dim]; py = slots_t[:, :, app_dim + 1]
                fg = (depth < depth_max); in_bnd = (px.abs() < bnd_threshold) & (py.abs() < bnd_threshold)
                mask = fg.detach() & in_bnd.detach()
                if mask.any():
                    depth_pred_loss = depth_pred_loss + F.mse_loss(pred_spread[mask], true_spread[mask])

            total_loss = (recon_loss + cfg.lambda_pos * loss_pos + cfg.lambda_cos * loss_cos +
                          depth_weight * depth_pred_loss)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            opt_main.step(); opt_pred2.step()

            if step % log_every == 0:
                r_pos = (cfg.lambda_pos * loss_pos).item(); r_cos = (cfg.lambda_cos * loss_cos).item()
                r_dp = (depth_weight * depth_pred_loss).item()
                print(f"Step {step:>5d}: recon={recon_loss.item():.6f} pos={r_pos:.6f} "
                      f"cos={r_cos:.6f} depth_p={r_dp:.6f}")
                history['step'].append(step)
                history['recon'].append(recon_loss.item()); history['pos'].append(r_pos)
                history['cos'].append(r_cos); history['depth_pred'].append(r_dp)

            if step > 0 and step % eval_every == 0:
                r2 = eval_r2(m, ds, n_samples=10)
                r2p = eval_predictor(m, predictor, ds, n_samples=5)
                history['r2'].append((step, r2)); history['r2_pred'].append((step, r2p))
                print(f"  >>> R²(depth→spread)={r2:.4f}  R²(predictor)={r2p:.4f}")

            step += 1
        if step >= n_steps: break

    # Save checkpoint
    save_ckpt = {
        'step': n_steps,
        'model': {k.replace('_orig_mod.', ''): v for k, v in m.state_dict().items()},
        'predictor': predictor.state_dict(),
        'config': {'depth_weight': depth_weight, 'n_steps': n_steps, 'hidden': 32,
                   'source_ckpt': CKPT_PATH, 'burnin_frames': 1},
    }
    save_path = os.path.join(SAVE_DIR, f'depth_predict_{n_steps}.pt')
    torch.save(save_ckpt, save_path)
    print(f"\nCheckpoint saved: {save_path}")

    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
    axes[0, 0].plot(history['step'], history['recon']); axes[0, 0].set_title('Recon Loss'); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(history['step'], history['pos']); axes[0, 1].set_title('Pos Loss'); axes[0, 1].grid(True, alpha=0.3)
    axes[0, 2].plot(history['step'], history['cos']); axes[0, 2].set_title('Cos Loss'); axes[0, 2].grid(True, alpha=0.3)
    axes[1, 0].plot(history['step'], history['depth_pred']); axes[1, 0].set_title('Depth Pred Loss'); axes[1, 0].grid(True, alpha=0.3)
    if history['r2']:
        steps, r2s = zip(*history['r2'])
        axes[1, 1].plot([0] + list(steps), [r2_base] + list(r2s), 'o-')
        axes[1, 1].axhline(y=r2_base, color='r', ls='--', alpha=0.5, label=f'baseline={r2_base:.4f}')
        axes[1, 1].set_title('R² (depth → spread)'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    if history['r2_pred']:
        steps, r2s = zip(*history['r2_pred'])
        axes[1, 2].plot(list(steps), list(r2s), 'o-', color='green')
        axes[1, 2].set_title('R² (predictor)'); axes[1, 2].grid(True, alpha=0.3)
    plt.suptitle(f'Depth→Spread Predictor Finetune (w={depth_weight}, burnin=1)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    final_r2 = history['r2'][-1][1] if history['r2'] else 0
    final_r2p = history['r2_pred'][-1][1] if history['r2_pred'] else 0
    print(f"\nR²(depth→spread): {r2_base:.4f} -> {final_r2:.4f}")
    print(f"R²(predictor): {r2_pred_init:.4f} -> {final_r2p:.4f}")


if __name__ == '__main__':
    train()
