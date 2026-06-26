#!/usr/bin/env python3
"""
Phase3 best.pt → sample9 各 FG slot 的 depth/spread/alpha_cov/pixcov 帧间变化曲线
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

    per_slot = {s: {'depth': [], 'spread': [], 'acov': [], 'pixcov': []} for s in range(N)}
    fg_slots = set()

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
        depth = slots_t[:, app_dim + 2]
        cov = alpha_2d.sum(dim=[-2, -1])
        a_max = alpha_2d.amax(dim=[-2, -1])
        dominant = alpha_2d.argmax(dim=0)
        pixcov = torch.stack([(dominant == s).sum() for s in range(N)]).float()

        for s in range(N):
            per_slot[s]['depth'].append(depth[s].item())
            per_slot[s]['spread'].append(spread[s].item())
            per_slot[s]['acov'].append(cov[s].item() / (H * W))
            per_slot[s]['pixcov'].append(pixcov[s].item() / (H * W))
            if (cov[s] > 20) & (a_max[s] > 0.7) & (depth[s] < 0.3):
                fg_slots.add(s)

    fg_slots = sorted(fg_slots)
    bg_slots = sorted(set(range(N)) - set(fg_slots))
    print(f"FG slots: {fg_slots}, BG slots: {bg_slots}")

    fr = np.arange(T)
    metrics = ['depth', 'spread', 'acov', 'pixcov']
    titles = ['Depth (=Scale)', 'Alpha Spread', 'Alpha Coverage (norm)', 'Pixel Coverage (norm)']
    colors = plt.cm.tab10(np.linspace(0, 1, N))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Phase3 Sample9: Slot-wise Metrics over {T} Frames\nFG={fg_slots}, BG={bg_slots}', fontsize=13)

    for m_idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[m_idx // 2][m_idx % 2]
        for s in fg_slots:
            vals = per_slot[s][metric]
            ax.plot(fr, vals, '-o', markersize=3, color=colors[s], label=f'Slot {s}', linewidth=1.5)
        ax.axvline(x=cfg_dict.get('burnin_frames', 6) - 0.5, color='gray', linestyle=':', alpha=0.5, label='burnin end')
        ax.set_xlabel('Frame')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = 'pos_depth_debug/phase3_sample9_slot_metrics.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {save_path}")

    # 数值输出
    print(f"\n{'Slot':>4} | {'mean_depth':>10} | {'mean_spread':>11} | {'mean_acov':>10} | {'mean_pixcov':>12}")
    print('-' * 60)
    for s in fg_slots:
        d = np.mean(per_slot[s]['depth'])
        sp = np.mean(per_slot[s]['spread'])
        ac = np.mean(per_slot[s]['acov'])
        pc = np.mean(per_slot[s]['pixcov'])
        print(f"{s:>4} | {d:>10.4f} | {sp:>11.4f} | {ac:>10.4f} | {pc:>12.4f}")


if __name__ == '__main__':
    main()
