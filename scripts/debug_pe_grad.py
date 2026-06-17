#!/usr/bin/env python3
"""验证位置编码梯度路径：从 slot loss → PE_recon → _decode_pe_from_zd → p_pred
模拟 _forward_finetune 的 decode 阶段梯度传播。
"""
import torch

POS_EMBED_DIM = 32
static_dim = 120
dyn_core_dim = 8
pos_enc_dim = 8

# ---------- 复制 dynamics.py 中的静态方法 ----------
@staticmethod
def _reconstruct_pe(pos_yx):
    B, N = pos_yx.shape[:2]
    pos_y, pos_x = pos_yx[:, :, 0], pos_yx[:, :, 1]
    D_pos = 32
    half = D_pos // 4
    device = pos_yx.device
    freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    pe = torch.stack([
        torch.sin(pos_y.unsqueeze(-1) * freq), torch.cos(pos_y.unsqueeze(-1) * freq),
        torch.sin(pos_x.unsqueeze(-1) * freq), torch.cos(pos_x.unsqueeze(-1) * freq),
    ], dim=-1).reshape(B, N, D_pos)
    return pe

@staticmethod
def _encode_pos_to_zd(pos_yx, pos_enc_dim):
    B, N = pos_yx.shape[:2]
    pos_y, pos_x = pos_yx[:, :, 0], pos_yx[:, :, 1]
    p = torch.zeros(B, N, pos_enc_dim, device=pos_yx.device)
    p[:, :, 0] = torch.sin(pos_y)
    p[:, :, 1] = torch.cos(pos_y)
    p[:, :, 2] = torch.sin(pos_x)
    p[:, :, 3] = torch.cos(pos_x)
    n_freq = (pos_enc_dim - 4) // 4
    if n_freq > 0:
        freq = 1.0 / (10000.0 ** (torch.arange(1, n_freq + 1, device=pos_yx.device).float() / 8))
        for i in range(n_freq):
            o = 4 + i * 4
            p[:, :, o + 0] = torch.sin(pos_y * freq[i])
            p[:, :, o + 1] = torch.cos(pos_y * freq[i])
            p[:, :, o + 2] = torch.sin(pos_x * freq[i])
            p[:, :, o + 3] = torch.cos(pos_x * freq[i])
    return p

@staticmethod
def _decode_pe_from_zd(p_zd):
    B, N, pos_enc_dim = p_zd.shape
    D_pos = 32
    half = D_pos // 4
    device = p_zd.device
    n_groups = pos_enc_dim // 4
    pos_y = torch.atan2(p_zd[:, :, 0], p_zd[:, :, 1])
    pos_x = torch.atan2(p_zd[:, :, 2], p_zd[:, :, 3])
    freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    pe_parts = []
    for fi in range(half):
        if fi < n_groups:
            o = fi * 4
            pe_parts.extend([p_zd[:, :, o], p_zd[:, :, o + 1],
                             p_zd[:, :, o + 2], p_zd[:, :, o + 3]])
        else:
            pe_parts.extend([
                torch.sin(pos_y * freq[fi]), torch.cos(pos_y * freq[fi]),
                torch.sin(pos_x * freq[fi]), torch.cos(pos_x * freq[fi]),
            ])
    return torch.stack(pe_parts, dim=-1).reshape(B, N, D_pos)


def test_pe_gradient():
    B, N = 8, 7
    D = static_dim + dyn_core_dim  # 128

    # 模拟：pred_S = f_z⁻¹(Z_core) + PE_recon(p_pred)
    #       target_S = slots_target + PE_gt
    # 其中 Z_core ≈ identity, f_z ≈ identity

    # 构造随机 target S
    torch.manual_seed(42)
    pos_yx_gt = torch.rand(B, N, 2) * 2 - 1          # 随机的真正位置 [-1, 1]
    pe_gt = _reconstruct_pe(pos_yx_gt)                 # target PE (32-dim)
    slots_target = torch.randn(B, N, D)                # target SA slots
    target_S = slots_target.clone()
    target_S[:, :, -POS_EMBED_DIM:] += pe_gt

    # 构造 pred S: S_raw = f_z⁻¹(Z_core) ≈ Z_core ≈ slots  (模拟 identity)
    slots_pred = torch.randn(B, N, D)

    # p_pred 作为可学习参数 (模拟 fusion_mlp 输出)
    p_pred = torch.zeros(B, N, pos_enc_dim, requires_grad=True)
    optimizer = torch.optim.Adam([p_pred], lr=0.001)

    print("=" * 60)
    print("Gradient Test: learning p_pred to match target PE")
    print("=" * 60)

    pos_gt_sample = pos_yx_gt[0, 0]
    print(f"\nTarget position (sample): pos_y={pos_gt_sample[0]:.4f}, pos_x={pos_gt_sample[1]:.4f}")
    print(f"Target PE[0:4] (base freq): {pe_gt[0, 0, :4].detach().tolist()}")
    print(f"Init p_pred: {p_pred[0, 0].detach().tolist()}")
    print(f"Init PE_recon[0:4]: {_decode_pe_from_zd(p_pred)[0, 0, :4].detach().tolist()}")
    print()

    for step in range(2000):
        optimizer.zero_grad()
        pe_recon = _decode_pe_from_zd(p_pred)
        S_raw = slots_pred  # simulate f_z⁻¹(Z_core) ≈ Z_core
        pred_S = S_raw.clone()
        pred_S[:, :, -POS_EMBED_DIM:] += pe_recon

        slot_loss = torch.nn.functional.mse_loss(pred_S, target_S)
        slot_loss.backward()

        if p_pred.grad is None:
            print(f"  Step {step}: ERROR: p_pred.grad is None!")
            return False
        if torch.isnan(p_pred.grad).any():
            print(f"  Step {step}: ERROR: p_pred.grad contains NaN!")
            return False
        if p_pred.grad.abs().max() == 0:
            print(f"  Step {step}: ERROR: p_pred.grad is ALL ZEROS!")
            return False

        optimizer.step()

        if step % 200 == 0:
            pe_recon_val = _decode_pe_from_zd(p_pred)
            mse_pe = torch.nn.functional.mse_loss(pe_recon_val, pe_gt)
            grad_norm = p_pred.grad.norm().item()
            p_vals = p_pred[0, 0].detach().tolist()
            print(f"  Step {step:>4}: slot_loss={slot_loss.item():.6f}, "
                  f"pe_mse={mse_pe.item():.6f}, grad_norm={grad_norm:.4f}, "
                  f"p[0:4]={[f'{v:.4f}' for v in p_vals[:4]]}")

    # 验证 p_pred 是否接近 GT 编码
    p_gt = _encode_pos_to_zd(pos_yx_gt, pos_enc_dim)
    mse_p = torch.nn.functional.mse_loss(p_pred, p_gt).item()
    print(f"\nFinal p_pred vs p_gt MSE: {mse_p:.6f}")

    if mse_p < 0.01:
        print("PASS: p_pred successfully learned to match target position encoding")
        return True
    else:
        print(f"FAIL: p_pred did NOT converge (MSE={mse_p:.6f})")
        return False


def test_pe_consistency():
    """验证 encode → decode 一致性：给定 centroid，重建出的 PE 和 _reconstruct_pe 一致"""
    B, N = 8, 7
    torch.manual_seed(42)
    pos_yx = torch.rand(B, N, 2) * 2 - 1

    p = _encode_pos_to_zd(pos_yx, pos_enc_dim)
    pe_from_decode = _decode_pe_from_zd(p)
    pe_from_reconstruct = _reconstruct_pe(pos_yx)

    mse = torch.nn.functional.mse_loss(pe_from_decode, pe_from_reconstruct).item()
    print(f"\nPE consistency check:")
    print(f"  _decode_pe_from_zd vs _reconstruct_pe MSE: {mse:.8f}")

    if mse < 1e-6:
        print("  PASS: PE reconstruction is consistent")
        return True
    else:
        # 找出差异在哪里
        diff = (pe_from_decode - pe_from_reconstruct).abs()
        max_diff = diff.max().item()
        argmax = diff.argmax()
        b, n, d = argmax // (N * 32), (argmax % (N * 32)) // 32, argmax % 32
        print(f"  FAIL: max diff = {max_diff:.6f} at dim {d}")
        print(f"    decode[{b},{n},{d}] = {pe_from_decode[b,n,d]:.6f}")
        print(f"    recon[{b},{n},{d}]  = {pe_from_reconstruct[b,n,d]:.6f}")
        return False


if __name__ == "__main__":
    print("Test 1: PE consistency (encode→decode)")
    test_pe_consistency()

    print("\n" + "=" * 60)
    print("Test 2: Gradient propagation (learning p_pred)")
    result = test_pe_gradient()

    if result:
        print("\nSUCCESS: Position encoding gradient path is correct.")
    else:
        print("\nFAILED: Position encoding gradient path has issues!")
