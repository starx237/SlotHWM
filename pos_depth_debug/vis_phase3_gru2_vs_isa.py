#!/usr/bin/env python3
"""
Phase3 Sample9: GRU2 预测的 appearance vs ISA 实际编码的 appearance
手动逐步执行 _forward_pretrain，提取每帧的:
  - gru2_app: GRU2 预测的 appearance (prev_app + residual)
  - isa_app: ISA 重新编码后的 appearance (corrected)
  - gru2_posdepth: GRU2 预测的 pos/depth
  - isa_posdepth: ISA 重新编码后的 pos/depth
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
        feat = model._encode_features(frames)

    burnin = 16
    gru2_predict_full = model.gru2_predict_full
    use_gru2 = True

    gru2_apps = []
    isa_apps = []
    gru2_posdepths = []
    isa_posdepths = []

    slots = None
    gru2_hidden = None
    prev_appearance = None
    prev_posdepth = None

    with torch.no_grad():
        for t in range(burnin):
            if t > 0 and use_gru2 and slots is not None:
                if gru2_predict_full:
                    new_appearance, new_posdepth, gru2_hidden = model._gru2_step(
                        prev_appearance, gru2_hidden, prev_posdepth)
                    gru2_apps.append(new_appearance[0].cpu().numpy())
                    gru2_posdepths.append(new_posdepth[0].cpu().numpy())
                    slots = torch.cat([new_appearance, new_posdepth], dim=-1)
                else:
                    new_appearance, gru2_hidden = model._gru2_step(
                        prev_appearance, gru2_hidden)
                    gru2_apps.append(new_appearance[0].cpu().numpy())
                    slots = torch.cat([
                        new_appearance,
                        slots[:, :, -3:-1].contiguous(),
                        slots[:, :, -1:].contiguous(),
                    ], dim=-1)
            else:
                gru2_apps.append(None)
                gru2_posdepths.append(None)

            slots, attn = model._sa(feat[:, t], slots, t)

            isa_apps.append(slots[0, :, :app_dim].cpu().numpy())
            isa_posdepths.append(slots[0, :, app_dim:].cpu().numpy())

            if use_gru2:
                prev_appearance = slots[:, :, :-3].detach()
                if gru2_predict_full:
                    prev_posdepth = slots[:, :, -3:].detach()
                if t == 0:
                    BN = prev_appearance.shape[0] * prev_appearance.shape[1]
                    gru2_hidden = torch.zeros(BN, model.gru2_hidden_dim, device=frames.device)
                    gru2_hidden = model.gru2(
                        prev_appearance.reshape(-1, app_dim),
                        gru2_hidden,
                    )

    N = isa_apps[0].shape[0]
    fg_slots = []
    for s in range(N):
        max_depth = max(isa_posdepths[t][s, 2] for t in range(burnin))
        if max_depth < 0.4:
            fg_slots.append(s)

    print(f"FG slots: {fg_slots}")

    # === 计算每帧 GRU2 pred vs ISA 的误差 ===
    print(f"\n=== App L2 error: GRU2 pred vs ISA (per frame, per FG slot) ===")
    header = f"{'t':>3}" + "".join(f"  Slot{s:>2}" for s in fg_slots)
    print(header)
    app_errors = {s: [] for s in fg_slots}
    for t in range(1, burnin):
        row = f"{t:>3}"
        for s in fg_slots:
            err = np.linalg.norm(gru2_apps[t][s] - isa_apps[t][s])
            app_errors[s].append(err)
            row += f"  {err:>6.4f}"
        print(row)

    # === posdepth error ===
    print(f"\n=== PosDepth L2 error: GRU2 pred vs ISA (per frame, per FG slot) ===")
    header = f"{'t':>3}" + "".join(f"  Slot{s:>2}" for s in fg_slots)
    print(header)
    posdepth_errors = {s: [] for s in fg_slots}
    depth_errors = {s: [] for s in fg_slots}
    for t in range(1, burnin):
        row = f"{t:>3}"
        for s in fg_slots:
            err = np.linalg.norm(gru2_posdepths[t][s] - isa_posdepths[t][s])
            posdepth_errors[s].append(err)
            depth_err = abs(gru2_posdepths[t][s, 2] - isa_posdepths[t][s, 2])
            depth_errors[s].append(depth_err)
            row += f"  {err:>6.4f}"
        print(row)

    # === Depth error 单独 ===
    print(f"\n=== Depth error: GRU2 pred depth - ISA depth (per frame, per FG slot) ===")
    header = f"{'t':>3}" + "".join(f"  Slot{s:>2}" for s in fg_slots)
    print(header)
    for t in range(1, burnin):
        row = f"{t:>3}"
        for s in fg_slots:
            d_pred = gru2_posdepths[t][s, 2]
            d_isa = isa_posdepths[t][s, 2]
            row += f"  {d_pred - d_isa:>+7.4f}"
        print(row)

    # === 对比: GRU2 Δapp (帧间残差) vs ISA correction (GRU2→ISA 修正量) ===
    print(f"\n=== App: GRU2 residual vs ISA correction (per FG slot) ===")
    for s in fg_slots:
        gru2_residuals = []
        isa_corrections = []
        for t in range(1, burnin):
            gru2_res = np.linalg.norm(gru2_apps[t][s] - isa_apps[t-1][s])
            isa_corr = np.linalg.norm(isa_apps[t][s] - gru2_apps[t][s])
            gru2_residuals.append(gru2_res)
            isa_corrections.append(isa_corr)
        print(f"  Slot {s}: mean GRU2_residual={np.mean(gru2_residuals):.4f}, "
              f"mean ISA_correction={np.mean(isa_corrections):.4f}, "
              f"ratio={np.mean(isa_corrections)/max(np.mean(gru2_residuals), 1e-8):.2f}")

    # === 绘图 ===
    fr = np.arange(1, burnin)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Phase3 Sample9: GRU2 Prediction vs ISA Encoding\n'
                 'GRU2 = prev_app + residual, ISA = re-encode from frame', fontsize=13)
    colors = plt.cm.tab10(np.linspace(0, 1, len(fg_slots)))

    # Row 1: App L2 error, Depth error, PosDepth L2 error
    for idx, s in enumerate(fg_slots):
        axes[0][0].plot(fr, app_errors[s], '-o', markersize=4, color=colors[idx],
                        label=f'Slot {s}', linewidth=1.5)
        axes[0][1].plot(fr, depth_errors[s], '-o', markersize=4, color=colors[idx],
                        label=f'Slot {s}', linewidth=1.5)
        axes[0][2].plot(fr, posdepth_errors[s], '-o', markersize=4, color=colors[idx],
                        label=f'Slot {s}', linewidth=1.5)

    # Row 2: GRU2 residual vs ISA correction
    for idx, s in enumerate(fg_slots):
        gru2_res = [np.linalg.norm(gru2_apps[t][s] - isa_apps[t-1][s]) for t in range(1, burnin)]
        isa_corr = [np.linalg.norm(isa_apps[t][s] - gru2_apps[t][s]) for t in range(1, burnin)]
        axes[1][0].plot(fr, gru2_res, '-o', markersize=4, color=colors[idx],
                        label=f'Slot {s}', linewidth=1.5)
        axes[1][1].plot(fr, isa_corr, '-o', markersize=4, color=colors[idx],
                        label=f'Slot {s}', linewidth=1.5)
        ratio = [c / max(r, 1e-8) for r, c in zip(gru2_res, isa_corr)]
        axes[1][2].plot(fr, ratio, '-o', markersize=4, color=colors[idx],
                        label=f'Slot {s}', linewidth=1.5)

    # 标注异常区间
    for ax in axes.flat:
        ax.axvspan(4, 6, alpha=0.08, color='red')
        ax.axvspan(8, 12, alpha=0.08, color='blue')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Frame')
        ax.legend(fontsize=7)

    axes[0][0].set_ylabel('L2 distance')
    axes[0][0].set_title('App Error: |GRU2_app - ISA_app|')
    axes[0][1].set_ylabel('|Δdepth|')
    axes[0][1].set_title('Depth Error: |GRU2_depth - ISA_depth|')
    axes[0][2].set_ylabel('L2 distance')
    axes[0][2].set_title('PosDepth Error: |GRU2_pd - ISA_pd|')
    axes[1][0].set_ylabel('L2 distance')
    axes[1][0].set_title('GRU2 Residual: |GRU2_app - prev_ISA_app|')
    axes[1][1].set_ylabel('L2 distance')
    axes[1][1].set_title('ISA Correction: |ISA_app - GRU2_app|')
    axes[1][2].set_ylabel('Ratio')
    axes[1][2].set_title('ISA Correction / GRU2 Residual')
    axes[1][2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('pos_depth_debug/phase3_sample9_gru2_vs_isa.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: phase3_sample9_gru2_vs_isa.png")


if __name__ == '__main__':
    main()
