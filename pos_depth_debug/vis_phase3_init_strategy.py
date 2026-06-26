#!/usr/bin/env python3
"""
验证 GRU2 的真正价值：slot-object assignment 一致性
对比三种 init 策略:
  A. GRU2 预测 (正常模式)
  B. 直接用上一帧 ISA 输出 (无 GRU2)
  C. 随机初始化 (baseline)
指标: 相邻帧间同 slot 的 attention mask IoU
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


def compute_iou(mask1, mask2):
    inter = (mask1 & mask2).sum().item()
    union = (mask1 | mask2).sum().item()
    return inter / max(union, 1)


def run_with_strategy(model, frames, strategy='gru2'):
    """手动逐步执行 _forward_pretrain, 可替换 init 策略"""
    app_dim = model.appearance_dim
    burnin = 16
    gru2_predict_full = model.gru2_predict_full

    with torch.no_grad():
        feat = model._encode_features(frames)

    slots = None
    gru2_hidden = None
    prev_appearance = None
    prev_posdepth = None

    all_alpha = []
    all_slots = []

    for t in range(burnin):
        if t > 0 and slots is not None:
            if strategy == 'gru2':
                new_appearance, new_posdepth, gru2_hidden = model._gru2_step(
                    prev_appearance, gru2_hidden, prev_posdepth)
                slots = torch.cat([new_appearance, new_posdepth], dim=-1)
            elif strategy == 'prev_frame':
                # 直接用上一帧 ISA 输出, 不经 GRU2
                slots = slots.detach().clone()
                gru2_hidden = model.gru2(
                    prev_appearance.reshape(-1, app_dim),
                    gru2_hidden.reshape(-1, model.gru2_hidden_dim),
                ) if gru2_hidden is not None else torch.zeros(
                    prev_appearance.shape[0] * prev_appearance.shape[1],
                    model.gru2_hidden_dim, device=frames.device)
            elif strategy == 'random':
                slots = torch.randn_like(slots) * 0.1
                gru2_hidden = torch.zeros(
                    slots.shape[0] * slots.shape[1],
                    model.gru2_hidden_dim, device=frames.device)

        slots, attn = model._sa(feat[:, t], slots, t)
        dec_rgb, dec_alpha, _ = model.decoder(slots, return_rgb=True)
        all_alpha.append(dec_alpha[0].cpu())
        all_slots.append(slots[0, :, :app_dim].detach().cpu().numpy())

        if t == 0:
            prev_appearance = slots[:, :, :-3].detach()
            prev_posdepth = slots[:, :, -3:].detach()
            BN = prev_appearance.shape[0] * prev_appearance.shape[1]
            gru2_hidden = torch.zeros(BN, model.gru2_hidden_dim, device=frames.device)
            gru2_hidden = model.gru2(
                prev_appearance.reshape(-1, app_dim),
                gru2_hidden,
            )
        else:
            prev_appearance = slots[:, :, :-3].detach()
            prev_posdepth = slots[:, :, -3:].detach()

    return all_alpha, all_slots


def main():
    with open('config/pretrain_phase3.yaml') as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict['burnin_frames'] = 16

    model = load_model(cfg_dict, 'experiments/phase3_gru2_full/checkpoints/best.pt')
    model.eval().cuda()

    ds = OBJ3DDataset(data_path='data/obj3d', num_frames=16, stride=4, subsample=2)
    sample = ds[9]
    frames = sample['video'].unsqueeze(0).cuda()

    N = 6
    burnin = 16

    strategies = ['gru2', 'prev_frame', 'random']
    strategy_labels = {'gru2': 'GRU2 pred', 'prev_frame': 'Prev frame', 'random': 'Random'}
    strategy_colors = {'gru2': 'blue', 'prev_frame': 'orange', 'random': 'red'}

    results = {}
    for strat in strategies:
        print(f"Running strategy: {strat}...")
        all_alpha, all_slots = run_with_strategy(model, frames, strategy=strat)
        results[strat] = {'alpha': all_alpha, 'slots': all_slots}

    # 计算 slot-object assignment IoU (相邻帧, 同slot, threshold=0.5)
    for strat in strategies:
        ious_per_slot = {s: [] for s in range(N)}
        for t in range(1, burnin):
            for s in range(N):
                mask_prev = results[strat]['alpha'][t-1][s] > 0.5
                mask_curr = results[strat]['alpha'][t][s] > 0.5
                if mask_prev.sum() == 0 and mask_curr.sum() == 0:
                    continue
                iou = compute_iou(mask_prev, mask_curr)
                ious_per_slot[s].append(iou)
        results[strat]['ious'] = ious_per_slot

    # 识别 FG slots (用 GRU2 策略的结果)
    fg_slots = []
    for s in range(N):
        mean_cov = np.mean([results['gru2']['alpha'][t][s].sum().item() for t in range(burnin)])
        if mean_cov > 20:
            fg_slots.append(s)

    print(f"\nFG slots: {fg_slots}")

    # === 数值输出 ===
    print(f"\n=== Mean Inter-frame IoU per slot ===")
    header = f"{'Slot':>5}" + "".join(f"  {strategy_labels[s]:>12}" for s in strategies)
    print(header)
    for s in range(N):
        row = f"{s:>5}"
        for strat in strategies:
            mean_iou = np.mean(results[strat]['ious'][s]) if results[strat]['ious'][s] else 0
            row += f"  {mean_iou:>12.4f}"
        print(row)

    print(f"\n=== FG Mean IoU ===")
    for strat in strategies:
        fg_ious = [iou for s in fg_slots for iou in results[strat]['ious'][s]]
        print(f"  {strategy_labels[strat]:>12}: {np.mean(fg_ious):.4f}")

    # === 绘图: 每帧 IoU ===
    fig, axes = plt.subplots(1, len(fg_slots) + 1, figsize=(5 * (len(fg_slots) + 1), 5))
    fr = np.arange(1, burnin)

    for idx, s in enumerate(fg_slots):
        ax = axes[idx]
        for strat in strategies:
            ious = results[strat]['ious'][s]
            if len(ious) == len(fr):
                ax.plot(fr, ious, '-o', markersize=4, color=strategy_colors[strat],
                        label=strategy_labels[strat], linewidth=1.5)
        ax.axvspan(4, 6, alpha=0.08, color='red')
        ax.axvspan(8, 12, alpha=0.08, color='blue')
        ax.set_xlabel('Frame')
        ax.set_ylabel('IoU')
        ax.set_title(f'Slot {s}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    # 最后一个: FG 平均
    ax = axes[-1]
    for strat in strategies:
        mean_per_frame = []
        for t_idx in range(burnin - 1):
            frame_ious = [results[strat]['ious'][s][t_idx] for s in fg_slots
                          if t_idx < len(results[strat]['ious'][s])]
            mean_per_frame.append(np.mean(frame_ious) if frame_ious else 0)
        ax.plot(fr, mean_per_frame, '-o', markersize=4, color=strategy_colors[strat],
                label=strategy_labels[strat], linewidth=1.5)
    ax.axvspan(4, 6, alpha=0.08, color='red')
    ax.axvspan(8, 12, alpha=0.08, color='blue')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mean IoU')
    ax.set_title('FG Mean')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.suptitle('Slot-Object Assignment Consistency: Inter-frame IoU\n'
                 '(GRU2 vs Prev-frame vs Random init for ISA)', fontsize=13)
    plt.tight_layout()
    plt.savefig('pos_depth_debug/phase3_sample9_init_strategy_iou.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: phase3_sample9_init_strategy_iou.png")

    # === 额外: App L2 差异 (GRU2 vs prev_frame init) ===
    print(f"\n=== App difference: GRU2_pred vs Prev_ISA (L2) ===")
    for s in fg_slots:
        diffs = []
        for t in range(1, burnin):
            gru2_app = results['gru2']['slots'][t][s]
            # prev_frame init = 上一帧 ISA 输出, 即 results['gru2']['slots'][t-1][s]
            prev_app = results['gru2']['slots'][t-1][s]
            diffs.append(np.linalg.norm(gru2_app - prev_app))
        print(f"  Slot {s}: mean={np.mean(diffs):.4f}, max={np.max(diffs):.4f}")


if __name__ == '__main__':
    main()
