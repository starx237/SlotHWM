"""
小型测试：在 best.pt 上加 Appearance Consistency Loss 微调
验证：
1. 能否收敛
2. 是否发生表征坍缩
3. 是否破坏已有 loss（recon, pos, cos）
4. Depth vs decoder spread 对齐是否改善
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os
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


def load_model(cfg, ckpt_path, freeze_isa=False):
    # 不设 continue_pretrain，避免 ISA 被冻结
    cfg_train = SimpleNamespace(**{k: v for k, v in vars(cfg).items()})
    cfg_train.continue_pretrain = False
    cfg_train.freeze_slot = False
    m = SlotDynamicsModel(cfg_train).cuda()
    ckpt = torch.load(ckpt_path, map_location='cpu')
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
    # 解冻所有 ISA 参数用于微调
    for p in m.parameters():
        p.requires_grad_(True)
    # 冻结 predictor（不需要训练）
    for p in m.predictor.parameters():
        p.requires_grad_(False)
    return m


def compute_spread(alpha_map, H=64, W=64):
    a = alpha_map.reshape(-1)
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    gx = gx.reshape(-1).to(a.device)
    gy = gy.reshape(-1).to(a.device)
    a_sum = a.sum()
    if a_sum < 1e-8:
        return 0.0
    cx = (a * gx).sum() / a_sum
    cy = (a * gy).sum() / a_sum
    return torch.sqrt((a * ((gx - cx) ** 2 + (gy - cy) ** 2)).sum() / a_sum).item()


def single_slot_sigmoid(decoder, slot):
    with torch.no_grad():
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
    return alpha


def eval_depth_spread_alignment(m, ds, n_samples=20):
    """评估 depth vs decoder alpha spread 的 R²"""
    app_dim = m.appearance_dim
    depth_max = 0.30

    all_depths = []
    all_spreads = []

    for idx in range(min(n_samples, len(ds))):
        sample = ds[idx]
        frames = sample['video'].unsqueeze(0).cuda()

        with torch.no_grad():
            feat = m._encode_features(frames)

        slots = None
        gru2_hidden = None
        prev_appearance = None
        for t in range(16):
            feat_t = feat[:, t]
            if t > 0 and slots is not None:
                new_appearance, gru2_hidden = m._gru2_step(prev_appearance, gru2_hidden)
                slots = torch.cat([new_appearance, slots[:, :, -3:-1].contiguous(),
                                   slots[:, :, -1:].contiguous()], dim=-1)
            slots, attn = m._sa(feat_t, slots, t)
            prev_appearance = slots[:, :, :-3].detach()
            if t == 0:
                BN = prev_appearance.shape[0] * prev_appearance.shape[1]
                gru2_hidden = torch.zeros(BN, m.gru2_hidden_dim, device=frames.device)
                gru2_hidden = m.gru2(
                    prev_appearance.reshape(-1, m.appearance_dim),
                    gru2_hidden.reshape(-1, m.gru2_hidden_dim))

            for s in range(6):
                d = slots[0, s, app_dim + 2].item()
                if d >= depth_max:
                    continue
                # 只用 FG slots（depth < depth_max 且不是 BG）
                # 用原始 pos/depth 单slot decode
                alpha = single_slot_sigmoid(m.decoder, slots[0, s])
                sp = compute_spread(alpha)
                if sp > 0.01:  # 过滤掉无效 slot
                    all_depths.append(d)
                    all_spreads.append(sp)

    all_depths = np.array(all_depths)
    all_spreads = np.array(all_spreads)
    mask = all_depths > 0.04
    if mask.sum() < 10:
        return 0.0
    coef = np.polyfit(all_depths[mask], all_spreads[mask], 1)
    y_pred = np.polyval(coef, all_depths[mask])
    r2 = 1 - ((all_spreads[mask] - y_pred) ** 2).sum() / \
        ((all_spreads[mask] - all_spreads[mask].mean()) ** 2).sum()
    return r2


def train_loop(m, ds, cfg, n_steps=200, app_weight=0.1, log_every=10):
    app_dim = m.appearance_dim
    depth_max = 0.30

    # 收集可训练参数 (predictor 已冻结)
    isa_params = [p for p in m.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(isa_params, lr=5e-5)

    results = {
        'step': [], 'recon_loss': [], 'app_consist_loss': [],
        'app_delta': [], 'r2_depth_spread': []
    }

    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                                      num_workers=2, pin_memory=True)

    step = 0
    for epoch in range(100):
        for batch in dl:
            if step >= n_steps:
                break

            frames = batch['video'].cuda()  # (B, T, 3, H, W)

            optimizer.zero_grad()

            # 不用 amp，避免数值问题
            with torch.no_grad():
                feat = m._encode_features(frames)

            all_slots_per_t = []
            slots = None
            gru2_hidden = None
            prev_appearance = None

            for t in range(frames.shape[1]):
                feat_t = feat[:, t].detach()
                if t > 0 and slots is not None:
                    new_appearance, gru2_hidden = m._gru2_step(prev_appearance, gru2_hidden)
                    slots = torch.cat([new_appearance, slots[:, :, -3:-1].contiguous(),
                                       slots[:, :, -1:].contiguous()], dim=-1)
                slots, attn = m._sa(feat_t, slots, t)
                prev_appearance = slots[:, :, :-3].detach()
                if t == 0:
                    BN = prev_appearance.shape[0] * prev_appearance.shape[1]
                    gru2_hidden = torch.zeros(BN, m.gru2_hidden_dim, device=frames.device)
                    gru2_hidden = m.gru2(
                        prev_appearance.reshape(-1, m.appearance_dim),
                        gru2_hidden.reshape(-1, m.gru2_hidden_dim))
                all_slots_per_t.append(slots)

            # Reconstruction loss
            recon_loss = 0
            for t in range(frames.shape[1]):
                s_t = all_slots_per_t[t]
                recon, _, _ = m.decoder(s_t, return_rgb=True)
                recon_loss += F.mse_loss(recon, frames[:, t])

            # Appearance consistency loss
            app_consist_loss = 0
            for t in range(1, len(all_slots_per_t)):
                app_delta = all_slots_per_t[t][:, :, :app_dim] - all_slots_per_t[t - 1][:, :, :app_dim]
                app_consist_loss += (app_delta ** 2).mean()

            loss = recon_loss + app_weight * app_consist_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(isa_params, 0.05)
            optimizer.step()

            # 记录 app delta
            with torch.no_grad():
                total_app_delta = 0
                count = 0
                for t in range(1, len(all_slots_per_t)):
                    d = (all_slots_per_t[t][:, :, :app_dim] - all_slots_per_t[t - 1][:, :, :app_dim]).norm(dim=-1)
                    total_app_delta += d.sum().item()
                    count += d.numel()
                avg_app_delta = total_app_delta / max(count, 1)

            if step % log_every == 0:
                print(f"Step {step:>4d}: recon={recon_loss.item():.5f}, "
                      f"app_consist={app_consist_loss.item():.5f}, "
                      f"loss={loss.item():.5f}, app_Δ={avg_app_delta:.4f}")
                results['step'].append(step)
                results['recon_loss'].append(recon_loss.item())
                results['app_consist_loss'].append(app_consist_loss.item())
                results['app_delta'].append(avg_app_delta)

            step += 1

        if step >= n_steps:
            break

    # 最终评估
    print("\n=== 最终评估 ===")
    r2 = eval_depth_spread_alignment(m, ds, n_samples=20)
    print(f"Depth-Decoder spread R²: {r2:.4f}")
    results['r2_depth_spread'].append(r2)

    return results


def main():
    with open('config/pretrain_obj3d.yaml') as f:
        cfg = SimpleNamespace(**yaml.safe_load(f))

    ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)

    # Baseline: 不加 app consistency
    print("=" * 60)
    print("BASELINE: 无 app consistency loss")
    print("=" * 60)
    m_base = load_model(cfg, 'pretrained/obj3d/checkpoints/best.pt')
    m_base.eval()
    r2_base = eval_depth_spread_alignment(m_base, ds, n_samples=20)
    print(f"Baseline R²: {r2_base:.4f}")

    # 测试不同 app_consistency 权重
    for app_weight in [0.5, 1.0, 5.0]:
        print("\n" + "=" * 60)
        print(f"TEST: app_consistency_weight = {app_weight}")
        print("=" * 60)

        m = load_model(cfg, 'pretrained/obj3d/checkpoints/best.pt')
        results = train_loop(m, ds, cfg, n_steps=500, app_weight=app_weight, log_every=50)

        # 保存训练曲线
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
        axes[0].plot(results['step'], results['recon_loss'])
        axes[0].set_title('Reconstruction Loss')
        axes[0].set_xlabel('Step')
        axes[1].plot(results['step'], results['app_consist_loss'])
        axes[1].set_title('App Consistency Loss')
        axes[1].set_xlabel('Step')
        axes[2].plot(results['step'], results['app_delta'])
        axes[2].set_title('Avg ||Δapp||')
        axes[2].set_xlabel('Step')
        plt.suptitle(f'app_consistency_weight = {app_weight}')
        plt.tight_layout()
        plt.savefig(f'pos_depth_debug/exp_app_consistency_w{app_weight}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
