"""
Depth Consistency Loss 训练测试
从 isa_single_poscosloss_40000.pt 开始，加入帧间 depth consistency loss
观察: burnin loss, pos loss, cos loss, depth loss, R² (ISA depth vs decoder alpha spread)

Depth consistency: 同一 FG slot 在相邻帧的 depth 应该平滑变化
loss_depth = ((depth_t - depth_{t-1})^2).mean() for FG slots within bnd_mask
"""
import os, sys, copy, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
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
SAVE_DIR = 'pos_depth_debug/depth_consist_ckpt'
os.makedirs(SAVE_DIR, exist_ok=True)

with open('config/pretrain_obj3d.yaml') as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))

# 不冻结 ISA
cfg.continue_pretrain = False
cfg.freeze_slot = False

app_dim = cfg.appearance_dim
depth_max = getattr(cfg, 'depth_max', 0.30)
bnd_threshold = getattr(cfg, 'bnd_threshold', 0.75)


def compute_normed_spread(weight_map, H=64, W=64):
    a = weight_map.reshape(-1)
    gy, gx = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W), indexing='ij')
    gx = gx.reshape(-1).to(a.device); gy = gy.reshape(-1).to(a.device)
    a_sum = a.sum()
    if a_sum < 1e-8: return 0.0
    a_norm = a / a_sum
    cx = (a_norm * gx).sum(); cy = (a_norm * gy).sum()
    return torch.sqrt((a_norm * ((gx-cx)**2 + (gy-cy)**2)).sum()).item()


def single_slot_sigmoid(decoder, slot):
    with torch.no_grad():
        dl = slot.unsqueeze(0).unsqueeze(0)
        B,N,D = dl.shape
        appearance = dl[...,:-3]; positions = dl[...,-3:-1]; depth_t = dl[...,-1:]
        S = decoder.broadcast_size
        broadcast = appearance.reshape(B*N, decoder.appearance_dim, 1, 1).expand(-1,-1,S,S)
        grid = create_coordinate_grid(S,S,dl.device).unsqueeze(0).unsqueeze(0).expand(B,N,S,S,2)
        relative_grid = grid - positions.view(B,N,1,1,2)
        relative_grid = relative_grid / decoder.scales_factor
        relative_grid = relative_grid / (depth_t.view(B,N,1,1,1)+1e-8)
        pos_emb = decoder.grid_proj(relative_grid).permute(0,1,4,2,3)
        x = decoder.backbone(broadcast + pos_emb.reshape(B*N, decoder.appearance_dim, S, S))
        out = decoder.out_conv(x).reshape(B,N,-1,64,64)
        alpha = torch.sigmoid(out[0,0,-1])
    return alpha


def eval_r2(m, ds, n_samples=20):
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
                    slots = torch.cat([new_app, slots[:, :, -3:-1].contiguous(), slots[:, :, -1:].contiguous()], dim=-1)
                slots, _ = m._sa(feat_t, slots, t)
                prev_app = slots[:, :, :-3].detach()
                if t == 0:
                    BN = prev_app.shape[0] * prev_app.shape[1]
                    gru2_hidden = torch.zeros(BN, m.gru2_hidden_dim, device=frames.device)
                    gru2_hidden = m.gru2(prev_app.reshape(-1, m.appearance_dim), gru2_hidden.reshape(-1, m.gru2_hidden_dim))
                for s in range(6):
                    d = slots[0, s, app_dim+2].item()
                    px = slots[0, s, app_dim].item(); py = slots[0, s, app_dim+1].item()
                    if d >= depth_max: continue
                    if abs(px) >= bnd_threshold or abs(py) >= bnd_threshold: continue
                    alpha = single_slot_sigmoid(m.decoder, slots[0, s])
                    sp = compute_normed_spread(alpha)
                    if sp > 0.01:
                        all_depths.append(d); all_spreads.append(sp)
    all_depths = np.array(all_depths); all_spreads = np.array(all_spreads)
    mask = all_depths > 0.04
    if mask.sum() < 10: return 0.0
    coef = np.polyfit(all_depths[mask], all_spreads[mask], 1)
    y_pred = np.polyval(coef, all_depths[mask])
    r2 = 1 - ((all_spreads[mask]-y_pred)**2).sum()/((all_spreads[mask]-all_spreads[mask].mean())**2).sum()
    return r2


def train():
    ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Load model
    m = SlotDynamicsModel(cfg).cuda()
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    sd = m.state_dict()
    ld = {}
    for mk in sd:
        mc = mk.replace('_orig_mod.','')
        for ck in ckpt['model']:
            cc = ck.replace('_orig_mod.','')
            if cc==mc and ckpt['model'][ck].shape==sd[mk].shape:
                ld[mk]=ckpt['model'][ck]; break
    m.load_state_dict(ld, strict=False)

    # Freeze predictor
    for p in m.predictor.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=cfg.learning_rate)

    n_steps = 1000
    depth_weight = 1.0  # depth consistency loss weight
    log_every = 50
    eval_every = 200

    history = {'step': [], 'recon': [], 'pos': [], 'cos': [], 'depth_consist': [], 'r2': []}

    step = 0
    for epoch in range(100):
        for batch in dl:
            if step >= n_steps: break
            frames = batch['video'].cuda()

            m.train()
            optimizer.zero_grad()

            out = m(frames)

            # Recon loss (same as trainer)
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

            # Pos loss (same as trainer)
            B = out["slots"]["corrected"].shape[0]
            N = out["slots"]["corrected"].shape[2]
            burnin_T = out["slots"]["corrected"].shape[1]
            loss_pos_list = []
            loss_cos_list = []

            for t in range(burnin_T):
                slots_t = out["slots"]["corrected"][:, t]

                # Cos loss
                if cfg.lambda_cos > 0:
                    attn_t = out["attn"][:, t].detach()
                    attn_dot = torch.bmm(attn_t, attn_t.transpose(1, 2))
                    diag = torch.eye(N, device=slots_t.device)
                    off_diag = (attn_dot * (1 - diag.unsqueeze(0))).sum(dim=[-2, -1])
                    loss_cos_t = off_diag.mean() / (N * (N - 1))
                    loss_cos_list.append(loss_cos_t)

                # Pos loss
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
                    noise_floor = 20
                    bg_threshold = 0.6 * H * W
                    fg_mask = (owned > noise_floor) & (owned < bg_threshold)
                    if fg_mask.any():
                        loss_pos_t = F.mse_loss(centroid[fg_mask], Sp[fg_mask])
                    else:
                        loss_pos_t = torch.tensor(0.0, device=slots_t.device)
                    loss_pos_list.append(loss_pos_t)

            loss_pos = torch.stack(loss_pos_list).mean() if loss_pos_list else torch.tensor(0.0)
            loss_cos = torch.stack(loss_cos_list).mean() if loss_cos_list else torch.tensor(0.0)

            # Depth consistency loss: 帧间 depth 平滑
            depth_consist_loss = torch.tensor(0.0, device=frames.device)
            if burnin_T > 1:
                for t in range(1, burnin_T):
                    depth_prev = out["slots"]["corrected"][:, t-1, :, app_dim+2]
                    depth_cur = out["slots"]["corrected"][:, t, :, app_dim+2]
                    # 只对 FG slots 计算 (depth < depth_max)
                    fg = (depth_cur < depth_max) & (depth_prev < depth_max)
                    # bnd mask
                    px_cur = out["slots"]["corrected"][:, t, :, app_dim]
                    py_cur = out["slots"]["corrected"][:, t, :, app_dim+1]
                    px_prev = out["slots"]["corrected"][:, t-1, :, app_dim]
                    py_prev = out["slots"]["corrected"][:, t-1, :, app_dim+1]
                    in_bnd = (px_cur.abs() < bnd_threshold) & (py_cur.abs() < bnd_threshold) & \
                             (px_prev.abs() < bnd_threshold) & (py_prev.abs() < bnd_threshold)
                    mask = fg & in_bnd
                    if mask.any():
                        depth_consist_loss = depth_consist_loss + ((depth_cur[mask] - depth_prev[mask])**2).mean()

            total_loss = (recon_loss +
                         cfg.lambda_pos * loss_pos +
                         cfg.lambda_cos * loss_cos +
                         depth_weight * depth_consist_loss)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], cfg.max_grad_norm)
            optimizer.step()

            if step % log_every == 0:
                r_pos = (cfg.lambda_pos * loss_pos).item()
                r_cos = (cfg.lambda_cos * loss_cos).item()
                r_dc = (depth_weight * depth_consist_loss).item()
                print(f"Step {step:>5d}: recon={recon_loss.item():.6f} pos={r_pos:.6f} "
                      f"cos={r_cos:.6f} depth_c={r_dc:.6f} total={total_loss.item():.6f}")
                history['step'].append(step)
                history['recon'].append(recon_loss.item())
                history['pos'].append(r_pos)
                history['cos'].append(r_cos)
                history['depth_consist'].append(r_dc)

            if step % eval_every == 0 and step > 0:
                r2 = eval_r2(m, ds, n_samples=10)
                history['r2'].append((step, r2))
                print(f"  >>> R² = {r2:.4f}")

            step += 1
        if step >= n_steps: break

    # Save checkpoint
    torch.save({
        'step': n_steps,
        'model': {k.replace('_orig_mod.',''): v for k, v in m.state_dict().items()},
    }, os.path.join(SAVE_DIR, f'depth_consist_{n_steps}.pt'))
    print(f"\nCheckpoint saved to {SAVE_DIR}/depth_consist_{n_steps}.pt")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axes[0,0].plot(history['step'], history['recon']); axes[0,0].set_title('Recon Loss'); axes[0,0].grid(True, alpha=0.3)
    axes[0,1].plot(history['step'], history['pos']); axes[0,1].set_title('Pos Loss'); axes[0,1].grid(True, alpha=0.3)
    axes[1,0].plot(history['step'], history['cos']); axes[1,0].set_title('Cos Loss'); axes[1,0].grid(True, alpha=0.3)
    axes[1,1].plot(history['step'], history['depth_consist']); axes[1,1].set_title('Depth Consistency Loss'); axes[1,1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # R² plot
    if history['r2']:
        steps, r2s = zip(*history['r2'])
        fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=150)
        ax2.plot(steps, r2s, 'o-'); ax2.set_xlabel('Step'); ax2.set_ylabel('R²')
        ax2.set_title('ISA depth vs Decoder alpha spread R²'); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'r2_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nFinal R²: {history['r2'][-1][1]:.4f}" if history['r2'] else "No R² evaluations")
    print("Done!")


if __name__ == '__main__':
    train()
