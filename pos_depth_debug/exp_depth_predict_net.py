"""
Depth Consistency via Predictive Network
从 (app, depth) 预测 alpha_spread, 最小化预测误差
梯度只流过 depth（app detach），鼓励 depth 调整使得映射关系更紧密

训练流程:
1. 正常 forward, 计算 recon + pos + cos loss
2. 从 slots 取 (app_detach, depth) 送入预测网络, 预测 alpha_spread
3. 预测 target = 实际计算的 alpha_spread (detach)
4. depth_consist_loss = MSE(predicted, target)
5. 梯度: 预测网络参数 + ISA depth (通过 depth 梯度)

从 isa_single_poscosloss_40000.pt 开始, burnin=1 (单帧训练)
"""
import os, sys, warnings
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
SAVE_DIR = 'pos_depth_debug/depth_predict_ckpt'
os.makedirs(SAVE_DIR, exist_ok=True)

with open('config/pretrain_obj3d.yaml') as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))
cfg.continue_pretrain = False
cfg.freeze_slot = False
cfg.burnin_frames = 1

app_dim = cfg.appearance_dim
depth_max = getattr(cfg, 'depth_max', 0.30)
bnd_threshold = getattr(cfg, 'bnd_threshold', 0.75)


class SpreadPredictor(nn.Module):
    """从 (app, depth) 预测 alpha_spread"""
    def __init__(self, app_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(app_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, appearance, depth):
        # appearance: (..., app_dim), depth: (..., 1)
        x = torch.cat([appearance, depth], dim=-1)
        return self.net(x).squeeze(-1)


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
    all_depths = []
    all_spreads = []
    m.eval()
    with torch.no_grad():
        for si in range(min(n_samples, len(ds))):
            sample = ds[si]
            frames = sample['video'].unsqueeze(0).cuda()
            feat = m._encode_features(frames)
            slots = None
            gru2_hidden = None
            prev_app = None
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
                for s in range(6):
                    d = slots[0, s, app_dim + 2].item()
                    px = slots[0, s, app_dim].item()
                    py = slots[0, s, app_dim + 1].item()
                    if d >= depth_max:
                        continue
                    if abs(px) >= bnd_threshold or abs(py) >= bnd_threshold:
                        continue
                    recon, alpha, _ = m.decoder(slots, return_rgb=True)
                    a = alpha[0, s, 0]
                    if a.sum() < 1.0:
                        continue
                    a_norm = a / a.sum()
                    gy, gx = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64),
                                             indexing='ij')
                    gx = gx.to(a.device)
                    gy = gy.to(a.device)
                    cx = (a_norm * gx).sum()
                    cy = (a_norm * gy).sum()
                    sp = torch.sqrt((a_norm * ((gx - cx) ** 2 + (gy - cy) ** 2)).sum()).item()
                    if sp > 0.01:
                        all_depths.append(d)
                        all_spreads.append(sp)
    all_depths = np.array(all_depths)
    all_spreads = np.array(all_spreads)
    mask = all_depths > 0.04
    if mask.sum() < 10:
        return 0.0
    coef = np.polyfit(all_depths[mask], all_spreads[mask], 1)
    y_pred = np.polyval(coef, all_depths[mask])
    r2 = 1 - ((all_spreads[mask] - y_pred) ** 2).sum() / (
            (all_spreads[mask] - all_spreads[mask].mean()) ** 2).sum()
    return r2


def eval_predictor_r2(m, predictor, ds, n_samples=10):
    """评估预测网络的 R²"""
    all_pred = []
    all_true = []
    m.eval()
    predictor.eval()
    with torch.no_grad():
        for si in range(min(n_samples, len(ds))):
            sample = ds[si]
            frames = sample['video'].unsqueeze(0).cuda()
            feat = m._encode_features(frames)
            slots = None
            gru2_hidden = None
            prev_app = None
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

                for s in range(6):
                    d = slots[0, s, app_dim + 2].item()
                    px = slots[0, s, app_dim].item()
                    py = slots[0, s, app_dim + 1].item()
                    if d >= depth_max:
                        continue
                    if abs(px) >= bnd_threshold or abs(py) >= bnd_threshold:
                        continue
                    app = slots[0, s, :app_dim]
                    depth = slots[0, s, app_dim + 2]
                    pred = predictor(app.unsqueeze(0), depth.unsqueeze(0).unsqueeze(0))
                    all_pred.append(pred.item())
                    all_true.append(true_spread[0, s].item())
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    if len(all_pred) < 10:
        return 0.0
    ss_tot = ((all_true - all_true.mean()) ** 2).sum()
    ss_res = ((all_true - all_pred) ** 2).sum()
    return 1 - ss_res / ss_tot


def train():
    ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                                      num_workers=2, pin_memory=True)

    m = SlotDynamicsModel(cfg).cuda()
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    sd = m.state_dict()
    ld = {}
    for mk in sd:
        mc = mk.replace('_orig_mod.', '')
        for ck in ckpt['model']:
            cc = ck.replace('_orig_mod.', '')
            if cc == mc and ckpt['model'][ck].shape == sd[mk].shape:
                ld[mk] = ckpt['model'][ck]
                break
    m.load_state_dict(ld, strict=False)
    for p in m.predictor.parameters():
        p.requires_grad_(False)

    predictor = SpreadPredictor(app_dim, hidden=64).cuda()
    opt_main = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=cfg.learning_rate)
    opt_pred = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    n_steps = 2000
    depth_weight = 0.1
    log_every = 50
    eval_every = 200

    history = {'step': [], 'recon': [], 'pos': [], 'cos': [], 'depth_consist': [],
               'r2_depth_spread': [], 'r2_predictor': []}

    r2_base = eval_r2(m, ds)
    r2_pred_base = eval_predictor_r2(m, predictor, ds)
    print(f"Baseline: R²(depth→spread)={r2_base:.4f}, R²(predictor)={r2_pred_base:.4f}\n")

    step = 0
    for epoch in range(100):
        for batch in dl:
            if step >= n_steps:
                break
            frames = batch['video'].cuda()
            m.train()
            predictor.train()
            opt_main.zero_grad()
            opt_pred.zero_grad()

            out = m(frames)

            # Recon loss
            dec_size = out["outputs"]["video_burnin"].shape[-1]
            target_size = frames.shape[-1]
            video_burnin = out["outputs"]["video_burnin"]
            if dec_size != target_size:
                b = frames.shape[0]
                video_burnin = F.interpolate(
                    video_burnin.reshape(-1, 3, dec_size, dec_size),
                    size=target_size, mode='bilinear'
                ).reshape(b, -1, 3, target_size, target_size)
            target_burnin = frames[:, :cfg.burnin_frames]
            recon_loss = F.mse_loss(video_burnin, target_burnin)

            # Pos + Cos loss
            B = out["slots"]["corrected"].shape[0]
            N = out["slots"]["corrected"].shape[2]
            burnin_T = out["slots"]["corrected"].shape[1]
            loss_pos_list = []
            loss_cos_list = []

            for t in range(burnin_T):
                slots_t = out["slots"]["corrected"][:, t]
                if cfg.lambda_cos > 0:
                    attn_t = out["attn"][:, t].detach()
                    attn_dot = torch.bmm(attn_t, attn_t.transpose(1, 2))
                    diag = torch.eye(N, device=slots_t.device)
                    loss_cos_t = (attn_dot * (1 - diag.unsqueeze(0))).sum(dim=[-2, -1]).mean() / (N * (N - 1))
                    loss_cos_list.append(loss_cos_t)
                if cfg.lambda_pos > 0:
                    alpha_t = out["alpha"][:, :, t].detach()
                    Sp = slots_t[:, :, -3:-1]
                    H, W = alpha_t.shape[-2:]
                    gy, gx = torch.meshgrid(
                        torch.linspace(-1, 1, H, device=slots_t.device),
                        torch.linspace(-1, 1, W, device=slots_t.device), indexing='ij')
                    a = alpha_t.squeeze(2)
                    denom = a.sum(dim=[-2, -1]) + 1e-8
                    cx = (a * gx).sum(dim=[-2, -1]) / denom
                    cy = (a * gy).sum(dim=[-2, -1]) / denom
                    centroid = torch.stack([cx, cy], dim=-1)
                    dominant = a.argmax(dim=1)
                    owned = torch.stack([(dominant == j).sum(dim=[-2, -1]) for j in range(N)], dim=-1).float()
                    fg_mask = (owned > 20) & (owned < 0.6 * H * W)
                    loss_pos_t = F.mse_loss(centroid[fg_mask], Sp[fg_mask]) if fg_mask.any() else torch.tensor(
                        0.0, device=slots_t.device)
                    loss_pos_list.append(loss_pos_t)

            loss_pos = torch.stack(loss_pos_list).mean() if loss_pos_list else torch.tensor(0.0)
            loss_cos = torch.stack(loss_cos_list).mean() if loss_cos_list else torch.tensor(0.0)

            # Depth Consistency Loss: 预测网络
            depth_consist_loss = torch.tensor(0.0, device=frames.device)
            for t in range(burnin_T):
                slots_t = out["slots"]["corrected"][:, t]  # (B, N, D)
                alpha_t = out["alpha"][:, :, t]
                if alpha_t.dim() == 5:
                    alpha_2d = alpha_t.squeeze(2)
                else:
                    alpha_2d = alpha_t

                # 计算 alpha spread (detach, 作为 target)
                true_spread = compute_alpha_spread_batched(alpha_2d).detach()  # (B, N)

                # 预测: app detach, depth 有梯度
                app_input = slots_t[:, :, :app_dim].detach()  # (B, N, app_dim)
                depth_input = slots_t[:, :, app_dim + 2]  # (B, N), 有梯度

                pred_spread = predictor(app_input, depth_input.unsqueeze(-1))  # (B, N)

                # Mask: FG + bnd
                px = slots_t[:, :, app_dim]
                py = slots_t[:, :, app_dim + 1]
                fg = (depth_input < depth_max)
                in_bnd = (px.abs() < bnd_threshold) & (py.abs() < bnd_threshold)
                # detach mask 里的 depth 相关部分, 避免梯度通过 mask
                mask = fg.detach() & in_bnd.detach()

                if mask.any():
                    depth_consist_loss = depth_consist_loss + F.mse_loss(pred_spread[mask], true_spread[mask])

            total_loss = (recon_loss +
                          cfg.lambda_pos * loss_pos +
                          cfg.lambda_cos * loss_cos +
                          depth_weight * depth_consist_loss)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            opt_main.step()
            opt_pred.step()

            if step % log_every == 0:
                r_pos = (cfg.lambda_pos * loss_pos).item()
                r_cos = (cfg.lambda_cos * loss_cos).item()
                r_dc = (depth_weight * depth_consist_loss).item()
                print(f"Step {step:>5d}: recon={recon_loss.item():.6f} pos={r_pos:.6f} "
                      f"cos={r_cos:.6f} depth_c={r_dc:.6f}")
                history['step'].append(step)
                history['recon'].append(recon_loss.item())
                history['pos'].append(r_pos)
                history['cos'].append(r_cos)
                history['depth_consist'].append(r_dc)

            if step > 0 and step % eval_every == 0:
                r2_ds = eval_r2(m, ds, n_samples=10)
                r2_pr = eval_predictor_r2(m, predictor, ds, n_samples=10)
                history['r2_depth_spread'].append((step, r2_ds))
                history['r2_predictor'].append((step, r2_pr))
                print(f"  >>> R²(depth→spread)={r2_ds:.4f}  R²(predictor)={r2_pr:.4f}")

            step += 1
        if step >= n_steps:
            break

    # Save
    torch.save({
        'step': n_steps,
        'model': {k.replace('_orig_mod.', ''): v for k, v in m.state_dict().items()},
        'predictor': predictor.state_dict(),
    }, os.path.join(SAVE_DIR, f'depth_predict_{n_steps}.pt'))

    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
    axes[0, 0].plot(history['step'], history['recon'])
    axes[0, 0].set_title('Recon Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(history['step'], history['pos'])
    axes[0, 1].set_title('Pos Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 2].plot(history['step'], history['cos'])
    axes[0, 2].set_title('Cos Loss')
    axes[0, 2].grid(True, alpha=0.3)
    axes[1, 0].plot(history['step'], history['depth_consist'])
    axes[1, 0].set_title('Depth Consistency Loss')
    axes[1, 0].grid(True, alpha=0.3)

    if history['r2_depth_spread']:
        steps, r2s = zip(*history['r2_depth_spread'])
        axes[1, 1].plot([0] + list(steps), [r2_base] + list(r2s), 'o-')
        axes[1, 1].axhline(y=r2_base, color='r', linestyle='--', alpha=0.5, label=f'baseline={r2_base:.4f}')
        axes[1, 1].set_title('R² (ISA depth vs Alpha spread)')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    if history['r2_predictor']:
        steps, r2s = zip(*history['r2_predictor'])
        axes[1, 2].plot([0] + list(steps), [r2_pred_base] + list(r2s), 'o-', color='green')
        axes[1, 2].set_title('R² (Predictor)')
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'Depth Predictive Network (w={depth_weight}, burnin=1)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    final_r2 = history['r2_depth_spread'][-1][1] if history['r2_depth_spread'] else 0
    final_r2p = history['r2_predictor'][-1][1] if history['r2_predictor'] else 0
    print(f"\nR²(depth→spread): {r2_base:.4f} -> {final_r2:.4f}")
    print(f"R²(predictor):    {r2_pred_base:.4f} -> {final_r2p:.4f}")
    print(f"Checkpoint: {SAVE_DIR}/depth_predict_{n_steps}.pt")


if __name__ == '__main__':
    train()
