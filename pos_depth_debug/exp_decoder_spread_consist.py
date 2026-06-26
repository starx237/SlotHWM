"""
Decoder Spread Consistency Loss

核心思路: 同一 slot 在相邻帧, 如果物体大小不变, 那么即使用不同 appearance 解码,
decoder 输出的 alpha spread 也应该一致。

Loss: 对 FG slot t, 用 (app_t, pos_t, depth_t) decode → spread_t
              用 (app_{t-1}, pos_t, depth_t) decode → spread_ref
      要求 ||spread_t - spread_ref||² → 0

梯度通过 depth_t, 鼓励 ISA 调整 depth 使 decoder 输出一致

实现关键: 需要对每个 slot 单独 decode (因为要替换 appearance)
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
SAVE_DIR = 'pos_depth_debug/decoder_spread_consist'
os.makedirs(SAVE_DIR, exist_ok=True)

with open('config/pretrain_obj3d.yaml') as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))
cfg.continue_pretrain = False
cfg.freeze_slot = False
cfg.burnin_frames = 6

app_dim = cfg.appearance_dim
depth_max = getattr(cfg, 'depth_max', 0.30)
bnd_threshold = getattr(cfg, 'bnd_threshold', 0.75)


def decode_single_slot_spread(decoder, app, pos, depth):
    """
    对单个 slot decode 并计算 alpha spread (归一化).
    app: (app_dim,), pos: (2,), depth: scalar
    返回: spread (scalar tensor, 有梯度)
    """
    slot = torch.cat([app, pos, depth.unsqueeze(-1)], dim=-1)
    dl = slot.unsqueeze(0).unsqueeze(0)
    B, N, D = dl.shape
    appearance = dl[..., :-3]
    positions = dl[..., -3:-1]
    depth_t = dl[..., -1:]
    S = decoder.broadcast_size
    broadcast = appearance.reshape(B * N, decoder.appearance_dim, 1, 1).expand(-1, -1, S, S)
    grid = create_coordinate_grid(S, S, dl.device).unsqueeze(0).unsqueeze(0).expand(B, N, S, S, 2)
    relative_grid = grid - positions.view(B, N, 1, 1, 2)
    relative_grid = relative_grid / decoder.scales_factor
    relative_grid = relative_grid / (depth_t.view(B, N, 1, 1, 1) + 1e-8)
    pos_emb = decoder.grid_proj(relative_grid).permute(0, 1, 4, 2, 3)
    x = decoder.backbone(broadcast + pos_emb.reshape(B * N, decoder.appearance_dim, S, S))
    out = decoder.out_conv(x).reshape(B, N, -1, 64, 64)
    alpha = torch.sigmoid(out[0, 0, -1])
    a_sum = alpha.sum()
    if a_sum < 0.5:
        return torch.tensor(0.0, device=alpha.device, requires_grad=True)
    a_norm = alpha / a_sum
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing='ij')
    gx = gx.to(alpha.device); gy = gy.to(alpha.device)
    cx = (a_norm * gx).sum(); cy = (a_norm * gy).sum()
    spread = torch.sqrt((a_norm * ((gx - cx) ** 2 + (gy - cy) ** 2)).sum())
    return spread


def eval_r2(m, ds, n_samples=10):
    all_depths = []; all_spreads = []
    m.eval()
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
                for s in range(6):
                    d = slots[0, s, app_dim + 2].item()
                    px = slots[0, s, app_dim].item(); py = slots[0, s, app_dim + 1].item()
                    if d >= depth_max: continue
                    if abs(px) >= bnd_threshold or abs(py) >= bnd_threshold: continue
                    a = alpha[0, s, 0]
                    if a.sum() < 1.0: continue
                    a_norm = a / a.sum()
                    gy, gx = torch.meshgrid(torch.linspace(-1,1,64), torch.linspace(-1,1,64), indexing='ij')
                    gx = gx.to(a.device); gy = gy.to(a.device)
                    cx = (a_norm*gx).sum(); cy = (a_norm*gy).sum()
                    sp = torch.sqrt((a_norm*((gx-cx)**2+(gy-cy)**2)).sum()).item()
                    if sp > 0.01: all_depths.append(d); all_spreads.append(sp)
    all_depths = np.array(all_depths); all_spreads = np.array(all_spreads)
    mask = all_depths > 0.04
    if mask.sum() < 10: return 0.0
    coef = np.polyfit(all_depths[mask], all_spreads[mask], 1)
    y_pred = np.polyval(coef, all_depths[mask])
    r2 = 1 - ((all_spreads[mask] - y_pred) ** 2).sum() / ((all_spreads[mask] - all_spreads[mask].mean()) ** 2).sum()
    return r2


def train():
    ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                                      num_workers=2, pin_memory=True)

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
    for p in m.predictor.parameters(): p.requires_grad_(False)

    optimizer = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=cfg.learning_rate)

    depth_weight = 0.05
    n_steps = 2000
    log_every = 50; eval_every = 200
    history = {'step': [], 'recon': [], 'pos': [], 'cos': [], 'depth_consist': [], 'r2': []}

    r2_base = eval_r2(m, ds)
    print(f"Baseline R²={r2_base:.4f}\n")

    step = 0
    for epoch in range(100):
        for batch in dl:
            if step >= n_steps: break
            frames = batch['video'].cuda()
            m.train(); optimizer.zero_grad()
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
                    attn_t = out["attn"][:, t].detach()
                    attn_dot = torch.bmm(attn_t, attn_t.transpose(1, 2))
                    diag = torch.eye(N, device=slots_t.device)
                    loss_cos_t = (attn_dot * (1 - diag.unsqueeze(0))).sum(dim=[-2, -1]).mean() / (N * (N - 1))
                    loss_cos_list.append(loss_cos_t)
                if cfg.lambda_pos > 0:
                    alpha_t = out["alpha"][:, :, t].detach()
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

            # Decoder Spread Consistency Loss
            depth_consist_loss = torch.tensor(0.0, device=frames.device)
            n_pairs = 0
            for t in range(1, burnin_T):
                slots_cur = out["slots"]["corrected"][:, t]   # (B, N, D)
                slots_prev = out["slots"]["corrected"][:, t - 1]

                for s in range(N):
                    # 只处理 FG + bnd slots
                    depth_cur = slots_cur[:, s, app_dim + 2]   # (B,)
                    depth_prev = slots_prev[:, s, app_dim + 2]  # (B,)
                    px = slots_cur[:, s, app_dim]; py = slots_cur[:, s, app_dim + 1]

                    # bnd mask: detach 避免梯度通过 mask
                    fg = (depth_cur < depth_max) & (depth_prev < depth_max)
                    in_bnd = (px.abs() < bnd_threshold) & (py.abs() < bnd_threshold)
                    mask = fg & in_bnd
                    mask = mask.detach()

                    if not mask.any(): continue

                    for b_idx in range(B):
                        if not mask[b_idx]: continue

                        app_cur = slots_cur[b_idx, s, :app_dim].detach()
                        app_prev = slots_prev[b_idx, s, :app_dim].detach()
                        pos_cur = slots_cur[b_idx, s, app_dim:app_dim + 2].detach()
                        depth_val = slots_cur[b_idx, s, app_dim + 2]  # 有梯度

                        # spread_cur: 用当前帧的 app+pos+depth
                        sp_cur = decode_single_slot_spread(m.decoder, app_cur, pos_cur, depth_val)

                        # spread_ref: 用上一帧的 app+当前pos+当前depth
                        sp_ref = decode_single_slot_spread(m.decoder, app_prev, pos_cur, depth_val)

                        if sp_cur.item() > 0.01 and sp_ref.item() > 0.01:
                            depth_consist_loss = depth_consist_loss + (sp_cur - sp_ref) ** 2
                            n_pairs += 1

            if n_pairs > 0:
                depth_consist_loss = depth_consist_loss / n_pairs

            total_loss = (recon_loss + cfg.lambda_pos * loss_pos + cfg.lambda_cos * loss_cos +
                          depth_weight * depth_consist_loss)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], cfg.max_grad_norm)
            optimizer.step()

            if step % log_every == 0:
                r_pos = (cfg.lambda_pos * loss_pos).item()
                r_cos = (cfg.lambda_cos * loss_cos).item()
                r_dc = (depth_weight * depth_consist_loss).item()
                print(f"Step {step:>5d}: recon={recon_loss.item():.6f} pos={r_pos:.6f} "
                      f"cos={r_cos:.6f} depth_c={r_dc:.6f} npairs={n_pairs}")
                history['step'].append(step)
                history['recon'].append(recon_loss.item())
                history['pos'].append(r_pos); history['cos'].append(r_cos)
                history['depth_consist'].append(r_dc)

            if step > 0 and step % eval_every == 0:
                r2 = eval_r2(m, ds, n_samples=10)
                history['r2'].append((step, r2))
                print(f"  >>> R²={r2:.4f}")

            step += 1
        if step >= n_steps: break

    # Save
    torch.save({'step': n_steps,
                'model': {k.replace('_orig_mod.', ''): v for k, v in m.state_dict().items()},
                'depth_weight': depth_weight}, os.path.join(SAVE_DIR, f'decoder_spread_consist_{n_steps}.pt'))

    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
    axes[0,0].plot(history['step'], history['recon']); axes[0,0].set_title('Recon Loss'); axes[0,0].grid(True, alpha=0.3)
    axes[0,1].plot(history['step'], history['pos']); axes[0,1].set_title('Pos Loss'); axes[0,1].grid(True, alpha=0.3)
    axes[0,2].plot(history['step'], history['cos']); axes[0,2].set_title('Cos Loss'); axes[0,2].grid(True, alpha=0.3)
    axes[1,0].plot(history['step'], history['depth_consist']); axes[1,0].set_title('Decoder Spread Consist Loss'); axes[1,0].grid(True, alpha=0.3)
    if history['r2']:
        steps, r2s = zip(*history['r2'])
        axes[1,1].plot([0]+list(steps), [r2_base]+list(r2s), 'o-')
        axes[1,1].axhline(y=r2_base, color='r', ls='--', alpha=0.5, label=f'baseline={r2_base:.4f}')
        axes[1,1].set_title('R² (ISA depth vs Alpha spread)'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)
    axes[1,2].axis('off')
    plt.suptitle(f'Decoder Spread Consistency (w={depth_weight}, burnin={cfg.burnin_frames})', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    final_r2 = history['r2'][-1][1] if history['r2'] else 0
    print(f"\nR²: {r2_base:.4f} -> {final_r2:.4f}")
    print(f"Checkpoint: {SAVE_DIR}/decoder_spread_consist_{n_steps}.pt")


if __name__ == '__main__':
    train()
