#!/usr/bin/env python3
"""
诊断 Slot 0 (帧4-6) 和 Slot 2 (帧8-12) 的 depth↑ cov↓ 异常
分析: alpha mask 可视化 + attention entropy + pos 变化
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings('ignore')
import torch, numpy as np, yaml
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset
from train import Trainer, create_optimizer
from train.trainer import WandBLogger


def load_model(cfg_dict, ckpt_path):
    cfg = SimpleNamespace(**cfg_dict)
    model = SlotDynamicsModel(cfg)
    opt, sch = create_optimizer((p for p in model.parameters() if p.requires_grad), cfg)
    wb = WandBLogger(enabled=False)
    trainer = Trainer(model, opt, sch, cfg, wandb_logger=wb)
    trainer.load_checkpoint(ckpt_path)
    return model


def main():
    with open('config/pretrain_phase3.yaml') as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict['burnin_frames'] = 16

    model = load_model(cfg_dict, 'experiments/phase3_gru2_full/checkpoints/best.pt')
    model.eval().cuda()
    app_dim = model.appearance_dim

    ds = OBJ3DDataset(data_path='data/obj3d', num_frames=16, stride=4, subsample=2)
    sample = ds[9]
    frames = sample['video'].unsqueeze(0).cuda()

    with torch.no_grad():
        out = model(frames)
    slots_c = out['slots']['corrected'] if isinstance(out['slots'], dict) else out['slots']
    alpha = out['alpha']
    T = slots_c.shape[1]
    N = slots_c.shape[2]

    # 收集所有帧的详细指标
    all_data = []
    for t in range(T):
        slots_t = slots_c[0, t]
        alpha_t = alpha[0, :, t]
        if alpha_t.dim() == 4:
            alpha_2d = alpha_t.squeeze(1)
        else:
            alpha_2d = alpha_t
        H, W = alpha_2d.shape[-2], alpha_2d.shape[-1]

        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, H, device='cuda'),
            torch.linspace(-1, 1, W, device='cuda'), indexing='ij')
        a_sum = alpha_2d.sum(dim=[-2, -1], keepdim=True) + 1e-8
        a_norm = alpha_2d / a_sum
        cx = (a_norm * gx.unsqueeze(0)).sum(dim=[-2, -1])
        cy = (a_norm * gy.unsqueeze(0)).sum(dim=[-2, -1])
        spread = torch.sqrt((a_norm * ((gx.unsqueeze(0) - cx.unsqueeze(-1).unsqueeze(-1))**2 +
                                        (gy.unsqueeze(0) - cy.unsqueeze(-1).unsqueeze(-1))**2)).sum(dim=[-2, -1]))
        depth = slots_t[:, app_dim + 2].cpu().numpy()
        pos_x = slots_t[:, app_dim].cpu().numpy()
        pos_y = slots_t[:, app_dim + 1].cpu().numpy()
        cov = alpha_2d.sum(dim=[-2, -1]).cpu().numpy()
        a_max = alpha_2d.amax(dim=[-2, -1]).cpu().numpy()

        # attention entropy per slot
        entropy = -(a_norm * (a_norm + 1e-10).log()).sum(dim=[-2, -1]).cpu().numpy()

        dominant = alpha_2d.argmax(dim=0)
        pixcov = torch.stack([(dominant == s).sum() for s in range(N)]).float().cpu().numpy()

        all_data.append({
            'depth': depth, 'spread': spread.cpu().numpy(), 'cov': cov / (H * W),
            'pixcov': pixcov / (H * W), 'a_max': a_max, 'entropy': entropy,
            'pos_x': pos_x, 'pos_y': pos_y, 'alpha_2d': alpha_2d.cpu().numpy()
        })

    # === 绘图1: Slot 0 和 Slot 2 的详细指标对比 ===
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle('Phase3 Sample9: Slot 0 (frames 4-6) & Slot 2 (frames 8-12) Anomaly\n'
                 'depth↑ spread↑ but cov↓ → attention diffusion artifact', fontsize=13)
    fr = np.arange(T)
    burnin = cfg_dict.get('burnin_frames', 16)

    metrics = [('depth', 'Depth (=Scale)'), ('spread', 'Alpha Spread'),
               ('cov', 'Alpha Coverage'), ('entropy', 'Attn Entropy')]
    for row, (key, label) in enumerate(metrics):
        for col, s in enumerate([0, 2]):
            ax = axes[row][col]
            vals = [all_data[t][key][s] for t in range(T)]
            ax.plot(fr, vals, '-o', markersize=4, linewidth=1.5)
            # 标注异常区间
            if s == 0:
                ax.axvspan(4, 6, alpha=0.15, color='red', label='anomaly')
            else:
                ax.axvspan(8, 12, alpha=0.15, color='red', label='anomaly')
            ax.set_xlabel('Frame')
            ax.set_ylabel(label)
            ax.set_title(f'Slot {s}: {label}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pos_depth_debug/phase3_sample9_anomaly_metrics.png', dpi=150, bbox_inches='tight')
    print("Saved: phase3_sample9_anomaly_metrics.png")

    # === 绘图2: alpha mask 可视化 (异常帧 vs 正常帧) ===
    slot0_frames = [3, 4, 5, 6, 7]  # 异常前、中、后
    slot2_frames = [7, 8, 9, 10, 11, 12, 13]

    fig, axes = plt.subplots(2, max(len(slot0_frames), len(slot2_frames)), figsize=(20, 8))

    for i, t in enumerate(slot0_frames):
        ax = axes[0][i]
        mask = all_data[t]['alpha_2d'][0]
        ax.imshow(mask, cmap='Reds', vmin=0, vmax=1)
        d = all_data[t]['depth'][0]
        ac = all_data[t]['cov'][0]
        ent = all_data[t]['entropy'][0]
        ax.set_title(f'Slot0 t={t}\nd={d:.3f} cov={ac:.3f} H={ent:.2f}', fontsize=9)
        ax.axis('off')

    for i, t in enumerate(slot2_frames):
        ax = axes[1][i]
        mask = all_data[t]['alpha_2d'][2]
        ax.imshow(mask, cmap='Blues', vmin=0, vmax=1)
        d = all_data[t]['depth'][2]
        ac = all_data[t]['cov'][2]
        ent = all_data[t]['entropy'][2]
        ax.set_title(f'Slot2 t={t}\nd={d:.3f} cov={ac:.3f} H={ent:.2f}', fontsize=9)
        ax.axis('off')

    plt.suptitle('Alpha Masks: Slot0 (top) frames 3-7, Slot2 (bottom) frames 7-13', fontsize=12)
    plt.tight_layout()
    plt.savefig('pos_depth_debug/phase3_sample9_anomaly_masks.png', dpi=150, bbox_inches='tight')
    print("Saved: phase3_sample9_anomaly_masks.png")

    # === 绘图3: 原始帧 + slot mask 叠加 ===
    ncols = max(len(slot0_frames), len(slot2_frames))
    fig, axes = plt.subplots(4, ncols, figsize=(4 * ncols, 16))
    for i in range(4):
        for j in range(ncols):
            axes[i][j].axis('off')

    for i, t in enumerate(slot0_frames):
        frame = sample['video'][t].permute(1, 2, 0).numpy()
        frame = np.clip((frame + 1) / 2, 0, 1)
        ax = axes[0][i]
        ax.imshow(frame); ax.set_title(f'Frame {t}', fontsize=9); ax.axis('off')
        ax = axes[1][i]
        mask0 = all_data[t]['alpha_2d'][0]
        overlay = frame.copy()
        m = mask0 > 0.2
        overlay[m] = overlay[m] * 0.5 + np.array([1, 0.2, 0]) * 0.5
        ax.imshow(overlay)
        ax.set_title(f'Slot0 t={t} d={all_data[t]["depth"][0]:.3f} cov={all_data[t]["cov"][0]:.3f}', fontsize=9)
        ax.axis('off')

    for i, t in enumerate(slot2_frames):
        frame = sample['video'][t].permute(1, 2, 0).numpy()
        frame = np.clip((frame + 1) / 2, 0, 1)
        ax = axes[2][i]
        ax.imshow(frame); ax.set_title(f'Frame {t}', fontsize=9); ax.axis('off')
        ax = axes[3][i]
        mask2 = all_data[t]['alpha_2d'][2]
        overlay = frame.copy()
        m = mask2 > 0.2
        overlay[m] = overlay[m] * 0.5 + np.array([0.2, 0.2, 1]) * 0.5
        ax.imshow(overlay)
        ax.set_title(f'Slot2 t={t} d={all_data[t]["depth"][2]:.3f} cov={all_data[t]["cov"][2]:.3f}', fontsize=9)
        ax.axis('off')

    plt.suptitle('Frame + Slot Mask Overlay (Slot0 red, Slot2 blue)', fontsize=12)
    plt.tight_layout()
    plt.savefig('pos_depth_debug/phase3_sample9_anomaly_overlay.png', dpi=150, bbox_inches='tight')
    print("Saved: phase3_sample9_anomaly_overlay.png")

    # 数值诊断
    print("\n=== Slot 0 帧间变化 ===")
    for t in range(1, T):
        dd = all_data[t]['depth'][0] - all_data[t-1]['depth'][0]
        ds_ = all_data[t]['spread'][0] - all_data[t-1]['spread'][0]
        dc = all_data[t]['cov'][0] - all_data[t-1]['cov'][0]
        dent = all_data[t]['entropy'][0] - all_data[t-1]['entropy'][0]
        flag = " ← ANOMALY" if (t >= 4 and t <= 6) else ""
        print(f"  t={t-1}→{t}: Δdepth={dd:+.4f} Δspread={ds_:+.4f} Δcov={dc:+.4f} Δentropy={dent:+.4f}{flag}")

    print("\n=== Slot 2 帧间变化 ===")
    for t in range(1, T):
        dd = all_data[t]['depth'][2] - all_data[t-1]['depth'][2]
        ds_ = all_data[t]['spread'][2] - all_data[t-1]['spread'][2]
        dc = all_data[t]['cov'][2] - all_data[t-1]['cov'][2]
        dent = all_data[t]['entropy'][2] - all_data[t-1]['entropy'][2]
        flag = " ← ANOMALY" if (t >= 8 and t <= 12) else ""
        print(f"  t={t-1}→{t}: Δdepth={dd:+.4f} Δspread={ds_:+.4f} Δcov={dc:+.4f} Δentropy={dent:+.4f}{flag}")


if __name__ == '__main__':
    main()
