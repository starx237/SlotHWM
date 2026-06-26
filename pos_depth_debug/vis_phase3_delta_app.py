#!/usr/bin/env python3
"""
Phase3 Sample9: 帧间 Δapp 大小分析
对每个 FG slot，计算相邻帧的 appearance 向量 L2 距离
与 Δdepth/Δcov 对比，看遮挡时 app 是否也剧烈变化
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

    # 收集每帧每 slot 的 app 向量 + 其他指标
    per_frame = []
    for t in range(T):
        slots_t = slots_c[0, t]
        alpha_t = alpha[0, :, t]
        if alpha_t.dim() == 4:
            alpha_2d = alpha_t.squeeze(1)
        else:
            alpha_2d = alpha_t
        H, W = alpha_2d.shape[-2], alpha_2d.shape[-1]
        app = slots_t[:, :app_dim].cpu().numpy()  # (N, app_dim)
        depth = slots_t[:, app_dim + 2].cpu().numpy()
        pos_x = slots_t[:, app_dim].cpu().numpy()
        pos_y = slots_t[:, app_dim + 1].cpu().numpy()
        cov = alpha_2d.sum(dim=[-2, -1]).cpu().numpy() / (H * W)
        a_max = alpha_2d.amax(dim=[-2, -1]).cpu().numpy()
        per_frame.append({'app': app, 'depth': depth, 'cov': cov, 'a_max': a_max,
                          'pos_x': pos_x, 'pos_y': pos_y})

    fg_slots = []
    for s in range(N):
        if np.mean([per_frame[t]['a_max'][s] for t in range(T)]) > 0.5:
            fg_slots.append(s)

    # 计算帧间 Δapp (L2) 和其他变化量
    delta_app = {s: [] for s in fg_slots}
    delta_depth = {s: [] for s in fg_slots}
    delta_cov = {s: [] for s in fg_slots}
    delta_pos = {s: [] for s in fg_slots}

    for t in range(1, T):
        for s in fg_slots:
            d_app = np.linalg.norm(per_frame[t]['app'][s] - per_frame[t-1]['app'][s])
            d_depth = per_frame[t]['depth'][s] - per_frame[t-1]['depth'][s]
            d_cov = per_frame[t]['cov'][s] - per_frame[t-1]['cov'][s]
            d_pos = np.sqrt((per_frame[t]['pos_x'][s] - per_frame[t-1]['pos_x'][s])**2 +
                            (per_frame[t]['pos_y'][s] - per_frame[t-1]['pos_y'][s])**2)
            delta_app[s].append(d_app)
            delta_depth[s].append(d_depth)
            delta_cov[s].append(d_cov)
            delta_pos[s].append(d_pos)

    fr_delta = np.arange(1, T)

    # === 绘图: Δapp, Δdepth, Δcov, Δpos ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Phase3 Sample9 FG Slots: Frame-to-Frame Changes\nFG={fg_slots}', fontsize=13)
    colors = plt.cm.tab10(np.linspace(0, 1, len(fg_slots)))

    for idx, s in enumerate(fg_slots):
        axes[0][0].plot(fr_delta, delta_app[s], '-o', markersize=4, color=colors[idx],
                        label=f'Slot {s}', linewidth=1.5)
        axes[0][1].plot(fr_delta, delta_depth[s], '-o', markersize=4, color=colors[idx],
                        label=f'Slot {s}', linewidth=1.5)
        axes[1][0].plot(fr_delta, delta_cov[s], '-o', markersize=4, color=colors[idx],
                        label=f'Slot {s}', linewidth=1.5)
        axes[1][1].plot(fr_delta, delta_pos[s], '-o', markersize=4, color=colors[idx],
                        label=f'Slot {s}', linewidth=1.5)

    # 标注异常区间
    for ax in axes.flat:
        ax.axvspan(4, 6, alpha=0.08, color='red')
        ax.axvspan(8, 12, alpha=0.08, color='blue')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Frame')

    axes[0][0].set_ylabel('Δapp (L2)')
    axes[0][0].set_title('Δapp (L2 distance)')
    axes[0][0].legend(fontsize=8)
    axes[0][1].set_ylabel('Δdepth')
    axes[0][1].set_title('Δdepth')
    axes[1][0].set_ylabel('Δcov (norm)')
    axes[1][0].set_title('Δcoverage')
    axes[1][1].set_ylabel('Δpos (L2)')
    axes[1][1].set_title('Δposition')

    plt.tight_layout()
    plt.savefig('pos_depth_debug/phase3_sample9_delta_app.png', dpi=150, bbox_inches='tight')
    print("Saved: phase3_sample9_delta_app.png")

    # === 数值输出 ===
    print(f"\nFG slots: {fg_slots}")
    print(f"\n=== Δapp (L2) per frame ===")
    header = f"{'t':>3}" + "".join(f"  Slot{s:>2}" for s in fg_slots)
    print(header)
    for i, t in enumerate(fr_delta):
        row = f"{t:>3}" + "".join(f"  {delta_app[s][i]:>6.4f}" for s in fg_slots)
        print(row)

    print(f"\n=== Mean Δapp per slot ===")
    for s in fg_slots:
        mean_da = np.mean(delta_app[s])
        max_da = np.max(delta_app[s])
        # 异常帧 vs 正常帧
        normal = [delta_app[s][i] for i in range(T-1) if not (4 <= i+1 <= 6 or 8 <= i+1 <= 12)]
        anomaly_frames = []
        if s == 0:
            anomaly_frames = [delta_app[s][i] for i in range(T-1) if 4 <= i+1 <= 6]
        elif s == 2:
            anomaly_frames = [delta_app[s][i] for i in range(T-1) if 8 <= i+1 <= 12]
        anom_str = f", anomaly={np.mean(anomaly_frames):.4f}" if anomaly_frames else ""
        print(f"  Slot {s}: mean={mean_da:.4f}, max={max_da:.4f}{anom_str}")

    # app L2 norm per frame (绝对大小)
    print(f"\n=== App L2 norm per frame ===")
    header = f"{'t':>3}" + "".join(f"  Slot{s:>2}" for s in fg_slots)
    print(header)
    for t in range(T):
        row = f"{t:>3}" + "".join(f"  {np.linalg.norm(per_frame[t]['app'][s]):>6.3f}" for s in fg_slots)
        print(row)

    # 相关性: Δapp vs |Δdepth|, Δapp vs |Δcov|
    print(f"\n=== Correlation: Δapp vs other deltas (across all FG slots & frames) ===")
    all_da, all_dd, all_dc = [], [], []
    for s in fg_slots:
        all_da.extend(delta_app[s])
        all_dd.extend([abs(d) for d in delta_depth[s]])
        all_dc.extend([abs(d) for d in delta_cov[s]])
    all_da = np.array(all_da)
    all_dd = np.array(all_dd)
    all_dc = np.array(all_dc)
    r_app_depth = np.corrcoef(all_da, all_dd)[0, 1]
    r_app_cov = np.corrcoef(all_da, all_dc)[0, 1]
    print(f"  corr(Δapp, |Δdepth|) = {r_app_depth:.4f}")
    print(f"  corr(Δapp, |Δcov|) = {r_app_cov:.4f}")


if __name__ == '__main__':
    main()
