#!/usr/bin/env python3
"""
v14: 
1. MLP depth predictor: (sharp_acov, app, pos) → depth
2. App predictor: encoder features + (pos, depth) → app
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np, yaml, time, math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from models.encoder import CNNEncoder
from train import Trainer, create_optimizer
from train.trainer import WandBLogger

OUT = os.path.dirname(os.path.abspath(__file__))
TAU = 0.05


def load_model(cfg_dict, ckpt_path):
    cfg = SimpleNamespace(**cfg_dict)
    model = SlotDynamicsModel(cfg)
    opt, sch = create_optimizer((p for p in model.parameters() if p.requires_grad), cfg)
    wb = WandBLogger(enabled=False)
    trainer = Trainer(model, opt, sch, cfg, wandb_logger=wb)
    trainer.load_checkpoint(ckpt_path)
    return model


def sharpen_alpha(alpha, tau=TAU):
    return torch.sigmoid((alpha - 0.5) / tau)


class DepthPredictorMLP(nn.Module):
    def __init__(self, app_dim=64, pos_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + app_dim + pos_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, sharp_acov, app, pos):
        x = torch.cat([sharp_acov, app, pos], dim=-1)
        return self.net(x).squeeze(-1)


class AppPredictor(nn.Module):
    def __init__(self, encoder, app_dim=64, feat_dim=64, hidden=128):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.feat_dim = feat_dim
        self.app_dim = app_dim
        self.read = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, app_dim)
        )

    def forward(self, frames, pos, depth):
        """
        frames: (B, T, 3, H, W)
        pos: (B, T, 2)
        depth: (B, T, 1)
        Returns: (B, T, app_dim)
        """
        B, T = frames.shape[:2]
        with torch.no_grad():
            feats = self.encoder(frames)  # (B, T, N, D) where N=256, D=64
        N = feats.shape[2]
        H_feat = int(math.sqrt(N))  # 16

        # bilinear interpolation at pos
        # pos is in [-1, 1] coordinate system, map to [0, H_feat-1]
        px = (pos[..., 0:1] + 1) / 2 * (H_feat - 1)  # (B, T, 1)
        py = (pos[..., 1:2] + 1) / 2 * (H_feat - 1)

        # reshape feats for grid_sample: (B*T, D, H_feat, H_feat)
        feats_2d = feats.reshape(B * T, H_feat, H_feat, self.feat_dim)
        feats_2d = feats_2d.permute(0, 3, 1, 2)  # (B*T, D, H, W)

        # grid_sample needs (B*T, 1, 1, 2) grid in [-1, 1]
        grid_x = pos[..., 0:1].reshape(B * T, 1, 1, 1)  # already in [-1, 1]
        grid_y = pos[..., 1:2].reshape(B * T, 1, 1, 1)
        grid = torch.cat([grid_x, grid_y], dim=-1)  # (B*T, 1, 1, 2)

        sampled = F.grid_sample(feats_2d, grid, align_corners=True)  # (B*T, D, 1, 1)
        sampled = sampled.reshape(B, T, self.feat_dim)  # (B, T, D)

        x = torch.cat([sampled, depth], dim=-1)  # (B, T, D+1)
        app = self.read(x)  # (B, T, app_dim)
        return app


def generate_depth_training_data(model, slots_c, alpha_out, fg_slots, bg_slots,
                                 app_dim, n_depth_samples=20):
    """为每个 (BestApp, pos, bg_slots) 配置采样多个 depth，生成训练数据"""
    T = slots_c.shape[1]
    per_frame_cov = {s: [] for s in fg_slots}
    for s in fg_slots:
        for t in range(T):
            alpha_t = alpha_out[0, s, t]
            a2 = alpha_t.squeeze(0) if alpha_t.dim() == 3 else alpha_t
            per_frame_cov[s].append(a2.sum().item() / (64 * 64))

    all_data = []
    for s in fg_slots:
        best_t = max(range(T), key=lambda t: per_frame_cov[s][t])
        best_app = slots_c[0, best_t, s, :app_dim].clone().detach()

        for t in range(T):
            slots_t = slots_c[0, t]
            pos_t = slots_t[s, app_dim:app_dim+2].detach()
            d_orig = slots_t[s, app_dim+2].item()
            bg_slots_t = slots_t[bg_slots].detach()

            # 采样多个 depth 值
            d_samples = torch.linspace(max(0.02, d_orig * 0.3), min(0.45, d_orig * 2.5), n_depth_samples)

            for d_val in d_samples:
                slot_vec = torch.cat([best_app, pos_t, torch.tensor([d_val], device='cuda')])
                with torch.no_grad():
                    slots_batch = torch.cat([
                        slot_vec.reshape(1, 1, -1),
                        bg_slots_t.reshape(1, -1, bg_slots_t.shape[-1])
                    ], dim=1)
                    _, alpha, _ = model.decoder(slots_batch, return_rgb=True)
                    alpha_sharp = sharpen_alpha(alpha[0, 0, 0], TAU)
                    sharp_acov = alpha_sharp.sum().item() / (64 * 64)

                all_data.append({
                    'sharp_acov': sharp_acov,
                    'app': best_app.cpu(),
                    'pos': pos_t.cpu(),
                    'depth': d_val.item()
                })

    sharp_acovs = torch.tensor([d['sharp_acov'] for d in all_data]).unsqueeze(1)
    apps = torch.stack([d['app'] for d in all_data])
    poss = torch.stack([d['pos'] for d in all_data])
    depths = torch.tensor([d['depth'] for d in all_data])

    return sharp_acovs, apps, poss, depths


def main():
    with open('config/pretrain_phase3.yaml') as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict['burnin_frames'] = 16
    ckpt_path = 'experiments/phase3_gru2_full/checkpoints/best.pt'

    print("Loading model...")
    model = load_model(cfg_dict, ckpt_path)
    model.eval().cuda()
    app_dim = model.appearance_dim

    sample = torch.load('data/sample9/sample9.pt')
    frames = sample['video'].unsqueeze(0).cuda()

    with torch.no_grad():
        out = model(frames)
    slots_c = out['slots']['corrected'] if isinstance(out['slots'], dict) else out['slots']
    alpha_out = out['alpha']
    T = slots_c.shape[1]
    N = slots_c.shape[2]

    fg_slots, bg_slots_list = [], []
    for s in range(N):
        mean_amax = np.mean([alpha_out[0, s, t].amax().item() for t in range(T)])
        max_depth = max(slots_c[0, t, s, app_dim + 2].item() for t in range(T))
        if mean_amax > 0.3 and max_depth < 0.4:
            fg_slots.append(s)
        else:
            bg_slots_list.append(s)
    print(f"FG={fg_slots}, BG={bg_slots_list}")

    # ============================================================
    # Part 1: Depth Predictor MLP
    # ============================================================
    print("\n=== Part 1: Depth Predictor MLP ===")
    print("Generating training data...")
    t0 = time.time()
    sharp_acovs, apps, poss, depths = generate_depth_training_data(
        model, slots_c, alpha_out, fg_slots, bg_slots_list, app_dim, n_depth_samples=30)
    print(f"  Data: {len(depths)} samples, {time.time()-t0:.1f}s")
    print(f"  sharp_acov range: [{sharp_acovs.min():.6f}, {sharp_acovs.max():.6f}]")
    print(f"  depth range: [{depths.min():.4f}, {depths.max():.4f}]")

    device = 'cuda'
    sharp_acovs = sharp_acovs.to(device)
    apps = apps.to(device)
    poss = poss.to(device)
    depths = depths.to(device)

    # Split train/val
    n = len(depths)
    idx = torch.randperm(n)
    n_train = int(0.8 * n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    depth_mlp = DepthPredictorMLP(app_dim=app_dim).to(device)
    optimizer = torch.optim.Adam(depth_mlp.parameters(), lr=1e-3)

    train_losses, val_losses = [], []
    for epoch in range(300):
        depth_mlp.train()
        pred = depth_mlp(sharp_acovs[train_idx], apps[train_idx], poss[train_idx])
        loss = F.mse_loss(pred, depths[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        depth_mlp.eval()
        with torch.no_grad():
            pred_val = depth_mlp(sharp_acovs[val_idx], apps[val_idx], poss[val_idx])
            val_loss = F.mse_loss(pred_val, depths[val_idx])
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: train_loss={loss.item():.6f}, val_loss={val_loss.item():.6f}")

    # Evaluate on sample9
    print("\n  Evaluating depth predictor on sample9...")
    per_frame_cov = {s: [] for s in fg_slots}
    nofg_sharp_acovs = {s: [] for s in fg_slots}
    for s in fg_slots:
        for t in range(T):
            alpha_t = alpha_out[0, s, t]
            a2 = alpha_t.squeeze(0) if alpha_t.dim() == 3 else alpha_t
            per_frame_cov[s].append(a2.sum().item() / (64 * 64))
            keep = [s] + bg_slots_list
            slots_nofg = slots_c[0, t][keep].unsqueeze(0)
            with torch.no_grad():
                _, alpha_nofg, _ = model.decoder(slots_nofg, return_rgb=True)
            alpha_sharp = sharpen_alpha(alpha_nofg[0, 0, 0], TAU)
            nofg_sharp_acovs[s].append(alpha_sharp.sum().item() / (64 * 64))

    # Reference: GD solved depths (20 steps)
    print("  Computing reference (GD 20-step) depths...")
    ref_d = {}
    for s in fg_slots:
        best_t = max(range(T), key=lambda t: per_frame_cov[s][t])
        best_app = slots_c[0, best_t, s, :app_dim].clone().detach()
        ref_d[s] = []
        for t in range(T):
            slots_t = slots_c[0, t]
            pos_t = slots_t[s, app_dim:app_dim+2].detach()
            d_orig = slots_t[s, app_dim+2].item()
            bg_slots_t = slots_t[bg_slots_list].detach()
            d_var = torch.tensor([d_orig], device='cuda', requires_grad=True)
            opt_d = torch.optim.Adam([d_var], lr=0.01)
            for _ in range(20):
                slot_v = torch.cat([best_app.detach(), pos_t.detach(), d_var.reshape(1)])
                sl = torch.cat([slot_v.reshape(1,1,-1),
                               bg_slots_t.reshape(1,-1,bg_slots_t.shape[-1])], dim=1)
                _, alpha, _ = model.decoder(sl, return_rgb=True)
                sa = sharpen_alpha(alpha[0,0,0], TAU).sum() / (64*64)
                l = (sa - nofg_sharp_acovs[s][t])**2
                opt_d.zero_grad(); l.backward(); opt_d.step()
                with torch.no_grad(): d_var.data.clamp_(0.01, 0.5)
            ref_d[s].append(d_var.item())

    # MLP prediction
    print("  MLP predictions:")
    depth_mlp.eval()
    mlp_d = {}
    for s in fg_slots:
        best_t = max(range(T), key=lambda t: per_frame_cov[s][t])
        best_app = slots_c[0, best_t, s, :app_dim].clone().detach()
        mlp_d[s] = []
        for t in range(T):
            pos_t = slots_c[0, t, s, app_dim:app_dim+2].detach()
            with torch.no_grad():
                sa = torch.tensor([[nofg_sharp_acovs[s][t]]], device='cuda')
                d_pred = depth_mlp(sa, best_app.unsqueeze(0), pos_t.unsqueeze(0))
                mlp_d[s].append(d_pred.item())
            d_orig = slots_c[0, t, s, app_dim+2].item()
            d_ref = ref_d[s][t]
            d_mlp = mlp_d[s][-1]

    # Summary
    print(f"\n  Depth Predictor Results:")
    for s in fg_slots:
        best_t = max(range(T), key=lambda t: per_frame_cov[s][t])
        mlp_errs, orig_errs = [], []
        print(f"  Slot {s} [best_t={best_t}]:")
        for t in range(T):
            d_orig = slots_c[0, t, s, app_dim+2].item()
            d_ref = ref_d[s][t]
            d_mlp = mlp_d[s][t]
            mlp_err = abs(d_mlp - d_ref)
            orig_err = abs(d_orig - d_ref)
            mlp_errs.append(mlp_err)
            orig_errs.append(orig_err)
            if t < 3 or t == best_t or t >= T-2:
                print(f"    t={t:>2}: d_orig={d_orig:.4f}, d_ref={d_ref:.4f}, "
                      f"d_mlp={d_mlp:.4f} (err={mlp_err:.4f} vs orig_err={orig_err:.4f})")
        print(f"    Mean error: MLP={np.mean(mlp_errs):.4f}, orig={np.mean(orig_errs):.4f}")

    # ============================================================
    # Part 2: App Predictor
    # ============================================================
    print("\n=== Part 2: App Predictor ===")

    # 准备训练数据: (frames, pos, depth) → app (solved)
    # 用 GD 50步求解的 app 作为 target
    print("Generating app training data (GD 50-step)...")

    # 先算 reference d_t (复用上面的)
    # 然后对每帧每 slot，GD 50步求 a_t
    app_targets = {s: [] for s in fg_slots}
    for s in fg_slots:
        best_t = max(range(T), key=lambda t: per_frame_cov[s][t])
        best_app_slot = slots_c[0, best_t, s, :app_dim].clone().detach()
        for t in range(T):
            slots_t = slots_c[0, t]
            pos_t = slots_t[s, app_dim:app_dim+2].detach()
            d_solved = ref_d[s][t]
            d_t_tensor = torch.tensor([d_solved], device='cuda')
            a_orig = slots_t[s, :app_dim].detach()
            all_slots_t = slots_t.unsqueeze(0).clone().detach()
            with torch.no_grad():
                dec_full, _, _ = model.decoder(slots_t.unsqueeze(0), return_rgb=True)
            target_img = dec_full.detach()

            a_t = a_orig.clone().detach().requires_grad_(True)
            opt_a = torch.optim.Adam([a_t], lr=0.02)
            for _ in range(50):
                target_slot = torch.cat([a_t, pos_t.detach(), d_t_tensor])
                sl = all_slots_t.clone()
                sl[0, s] = target_slot
                recon, _, _ = model.decoder(sl, return_rgb=True)
                l = F.mse_loss(recon, target_img)
                opt_a.zero_grad(); l.backward(); opt_a.step()
            app_targets[s].append(a_t.detach())

    # Build app predictor training data
    all_frames_data = frames.repeat(1, 1, 1, 1, 1)  # (1, T, 3, 64, 64)
    train_pos, train_depth, train_app = [], [], []
    for s in fg_slots:
        for t in range(T):
            pos_t = slots_c[0, t, s, app_dim:app_dim+2].cpu()
            d_t_val = torch.tensor([ref_d[s][t]])
            a_t = app_targets[s][t].cpu()
            train_pos.append(pos_t)
            train_depth.append(d_t_val)
            train_app.append(a_t)

    train_pos = torch.stack(train_pos)
    train_depth = torch.stack(train_depth)
    train_app = torch.stack(train_app)

    # Create app predictor
    encoder = model.encoder
    app_predictor = AppPredictor(encoder, app_dim=app_dim, feat_dim=64, hidden=128).to(device)

    # Train
    opt_app = torch.optim.Adam(app_predictor.read.parameters(), lr=1e-3)
    frames_input = frames  # (1, T, 3, 64, 64)
    pos_input = train_pos.to(device)  # (N, 2)
    depth_input = train_depth.to(device)  # (N, 1)
    app_target = train_app.to(device)  # (N, 64)

    # Need to reshape for batch processing
    # frames: expand to (N, 1, 3, 64, 64) where N = len(fg_slots) * T
    n_total = len(fg_slots) * T
    frames_expanded = frames.expand(n_total, -1, -1, -1, -1)  # (N, T, 3, H, W)

    # But app_predictor expects (B, T, 3, H, W) with B=batch_size
    # Let's just use B=1 and iterate, or batch properly

    # Simpler: for each (s, t), the input is frame at time t
    # We need to construct proper batch
    # frames: (1, T, 3, 64, 64)
    # For each (s, t), input is frames[0, t:t+1] → (1, 1, 3, 64, 64)

    # Let me batch all samples: create frames_batch of shape (n_total, 1, 3, 64, 64)
    frame_list = []
    for s in fg_slots:
        for t in range(T):
            frame_list.append(frames[0, t:t+1])  # (1, 3, 64, 64)
    frames_batch = torch.stack(frame_list)  # (n_total, 1, 3, 64, 64)

    app_losses = []
    for epoch in range(200):
        app_predictor.train()
        pred_app = app_predictor(frames_batch, pos_input.unsqueeze(1), depth_input.unsqueeze(1))
        pred_app = pred_app.squeeze(1)  # (n_total, 64)
        loss = F.mse_loss(pred_app, app_target)
        opt_app.zero_grad()
        loss.backward()
        opt_app.step()
        app_losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f}")

    # Evaluate
    print("\n  App Predictor Results:")
    app_predictor.eval()
    with torch.no_grad():
        pred_app = app_predictor(frames_batch, pos_input.unsqueeze(1), depth_input.unsqueeze(1))
        pred_app = pred_app.squeeze(1)

    for s in fg_slots:
        cos_sims, mses = [], []
        for t in range(T):
            idx = fg_slots.index(s) * T + t
            pred = pred_app[idx]
            target = app_target[idx]
            cos_sim = F.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0)).item()
            mse = F.mse_loss(pred, target).item()
            cos_sims.append(cos_sim)
            mses.append(mse)
        print(f"  Slot {s}: cos_sim={np.mean(cos_sims):.4f}, MSE={np.mean(mses):.6f}")

    # ============================================================
    # Plotting
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Depth MLP training curve
    axes[0][0].plot(train_losses, label='train', alpha=0.7)
    axes[0][0].plot(val_losses, label='val', alpha=0.7)
    axes[0][0].set_xlabel('Epoch'); axes[0][0].set_ylabel('MSE')
    axes[0][0].set_title('Depth MLP training'); axes[0][0].legend()
    axes[0][0].grid(True, alpha=0.3)

    # Depth comparison: orig vs ref vs MLP
    colors = plt.cm.tab10(np.linspace(0, 1, len(fg_slots)))
    for idx_s, s in enumerate(fg_slots):
        d_origs = [slots_c[0, t, s, app_dim+2].item() for t in range(T)]
        d_refs = ref_d[s]
        d_mlps = mlp_d[s]
        fr = np.arange(T)
        axes[0][1].plot(fr, d_origs, '--', color=colors[idx_s], alpha=0.3)
        axes[0][1].plot(fr, d_refs, 'o', color=colors[idx_s], markersize=4, alpha=0.6)
        axes[0][1].plot(fr, d_mlps, 's', color=colors[idx_s], markersize=3)
    axes[0][1].set_xlabel('Frame'); axes[0][1].set_ylabel('Depth')
    axes[0][1].set_title('Depth: orig(dash) vs ref(circle) vs MLP(square)')
    axes[0][1].grid(True, alpha=0.3)

    # App predictor training curve
    axes[1][0].plot(app_losses)
    axes[1][0].set_xlabel('Epoch'); axes[1][0].set_ylabel('MSE')
    axes[1][0].set_title('App Predictor training')
    axes[1][0].grid(True, alpha=0.3)

    # App predictor: cos_sim per slot per frame
    for idx_s, s in enumerate(fg_slots):
        cos_sims = []
        for t in range(T):
            idx = fg_slots.index(s) * T + t
            pred = pred_app[idx]
            target = app_target[idx]
            cos_sim = F.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0)).item()
            cos_sims.append(cos_sim)
        axes[1][1].plot(np.arange(T), cos_sims, '-o', markersize=3, label=f'Slot {s}')
    axes[1][1].set_xlabel('Frame'); axes[1][1].set_ylabel('Cosine Similarity')
    axes[1][1].set_title('App Predictor: cos_sim with target')
    axes[1][1].legend(); axes[1][1].grid(True, alpha=0.3)

    plt.suptitle('v14: Depth MLP + App Predictor', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUT}/v14_predictors.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: v14_predictors.png")

    # Inference speed test
    print("\n=== Speed Test ===")
    depth_mlp.eval()
    # Depth MLP
    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            sa = torch.rand(1, 1, device='cuda')
            ap = torch.rand(1, app_dim, device='cuda')
            po = torch.rand(1, 2, device='cuda')
            depth_mlp(sa, ap, po)
    print(f"  Depth MLP: {(time.time()-t0)/100*1000:.3f} ms/call")

    # App predictor
    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            app_predictor(frames_batch[:1], pos_input[:1].unsqueeze(1),
                         depth_input[:1].unsqueeze(1))
    print(f"  App Predictor: {(time.time()-t0)/100*1000:.3f} ms/call")


if __name__ == '__main__':
    main()
