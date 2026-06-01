#!/usr/bin/env python3
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from data.obj3d_dataset import OBJ3DDataset


def make_config():
    from types import SimpleNamespace
    d = dict(
        num_slots=7, slot_dim=128, hidden_dim=256,
        num_heads=4, qkv_size=128, mlp_size=256,
        num_layers=2, h_layers=1, out_mlp=False,
        out_hidden_layers=1, pre_norm=False, dropout_rate=0.1,
        delta_t=0.125, integrator_method="Leapfrog",
        burnin_frames=6, rollout_frames=10, buffer_len=10,
        batch_size=2, learning_rate=2e-4, weight_decay=0.0,
        max_grad_norm=0.05, warmup_steps=1000,
        lambda_slots=1.0, lambda_images=0.0, lambda_energy=0.01, lambda_static=0.001,
        dataset="obj3d", data_root="./data",
        encoder_type="cnn", in_channels=3, img_size=64,
        encoder_hidden=32, decoder_hidden=64, broadcast_size=8,
        slot_hidden=128, slot_iters=3,
        return_energy=False, use_static_context=True,
        num_spatiotemporal_blocks=2, spatiotemporal_mlp_size=256,
        spatiotemporal_pre_norm=False,
        workdir='/tmp/smoke_test', num_workers=0,
    )
    return SimpleNamespace(**d)


def test_slotpi_model():
    """SlotPiModel（物理模块）单独测试"""
    print("=" * 60)
    print("SlotPiModel Physical Module Test")
    print("=" * 60)
    from models.slotpi_model import SlotPiModel
    from train.losses import SlotPiLoss

    cfg = make_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SlotPiModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    B, T, N, D = 4, 30, cfg.num_slots, cfg.slot_dim
    slots = torch.randn(B, T, N, D, device=device)

    loss_fn = SlotPiLoss(cfg)
    burnin, rollout = cfg.burnin_frames, cfg.rollout_frames
    slots_burnin = slots[:, :burnin]
    slots_gt = slots[:, burnin:burnin + rollout]
    buffer = torch.zeros(B, cfg.buffer_len, N, D, device=device)
    for t in range(burnin):
        buffer = torch.cat([buffer[:, 1:], slots_burnin[:, t].unsqueeze(1)], dim=1)

    slots_burnin = slots_burnin.to(device)
    slots_gt = slots_gt.to(device)
    C = model.compute_C(slots_burnin) if hasattr(model, 'compute_C') else None

    pred_slots_list = []
    slots_t = slots_burnin[:, -1]
    for _ in range(rollout):
        ns = model(slots_t, buffer, C=C)
        pred_slots_list.append(ns)
        buffer = torch.cat([buffer[:, 1:], ns.unsqueeze(1)], dim=1)
        slots_t = ns
    pred_slots = torch.stack(pred_slots_list, dim=1)

    loss, aux = loss_fn(pred_slots, slots_gt,
                        slots_full_seq=torch.cat([slots_burnin, pred_slots], dim=1))
    init_loss = loss.item()
    print(f"  Loss(before): {init_loss:.4f}  slot={aux['slot_loss']:.4f}  static={aux['static_loss']:.6f}")

    p_before = list(model.parameters())[0].clone()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    optimizer.step()
    p_after = list(model.parameters())[0].clone()
    assert not torch.equal(p_before, p_after), "Parameters did not change!"
    print("  ✓ SlotPiModel PASSED\n")
    return True


def test_slotpi():
    """SlotPi 端到端模型测试"""
    print("=" * 60)
    print("SlotPi End-to-End Test")
    print("=" * 60)
    from models.slotpi import SlotPi
    from train.losses import SlotPiLoss

    cfg = make_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    ds = OBJ3DDataset(data_path='data/obj3d',
                       num_frames=cfg.burnin_frames + cfg.rollout_frames)
    loader = ds.get_dataloader(batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    frames = batch["video"].to(device)
    print(f"  Frames shape: {frames.shape}")

    model = SlotPi(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = SlotPiLoss(cfg)
    model.train()
    optimizer.zero_grad()
    out = model(frames)

    burnin, rollout = cfg.burnin_frames, cfg.rollout_frames
    dec_size = out["outputs"]["video_burnin"].shape[-1]
    if dec_size != frames.shape[-1]:
        b, t = frames.shape[:2]
        target_burnin = nn.functional.interpolate(
            frames[:, :burnin].reshape(-1, 3, 64, 64), size=dec_size, mode='bilinear'
        ).reshape(b, burnin, 3, dec_size, dec_size)
        target_rollout = nn.functional.interpolate(
            frames[:, burnin:burnin + rollout].reshape(-1, 3, 64, 64),
            size=dec_size, mode='bilinear'
        ).reshape(b, rollout, 3, dec_size, dec_size)
    else:
        target_burnin = frames[:, :burnin]
        target_rollout = frames[:, burnin:burnin + rollout]

    recon_burnin = nn.functional.mse_loss(out["outputs"]["video_burnin"], target_burnin)
    recon_rollout = nn.functional.mse_loss(out["outputs"]["video_pred"], target_rollout)
    all_slots = torch.cat([out["slots"]["corrected"], out["slots"]["predicted"]], dim=1)
    slot_loss, aux = loss_fn(out["slots"]["predicted"], out["slots"]["target"],
                             slots_full_seq=all_slots)

    total_loss = recon_burnin + recon_rollout + slot_loss
    init_loss = total_loss.item()
    print(f"  Loss(before): total={init_loss:.4f}  recon={recon_burnin.item()+recon_rollout.item():.4f}  "
          f"slot={aux['slot_loss']:.4f}  static={aux['static_loss']:.6f}")

    p_before = list(model.parameters())[0].clone()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    optimizer.step()
    p_after = list(model.parameters())[0].clone()
    assert not torch.equal(p_before, p_after), "Parameters did not change!"
    print("  ✓ SlotPi E2E PASSED\n")
    return True


if __name__ == '__main__':
    t0 = time.time()
    ok1 = test_slotpi_model()
    ok2 = test_slotpi()
    elapsed = time.time() - t0
    print(f"Total: {elapsed:.1f}s")
    if ok1 and ok2:
        print("ALL SMOKE TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)
