import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


def find_foreground_slots(alpha_maps: torch.Tensor, n_slots: int = 2) -> Tuple[List[int], int]:
    """从 alpha mask 中找前景 slot 和背景 slot。
    alpha_maps: (B, N, 1, H, W), B=1
    步骤:
      1. 总 alpha 和最大的 = 背景（覆盖像素最多）
      2. 其余 slot 按空间集中度（方差）排序
      3. 取前 n_slots 个集中度最高的作为前景
    Returns: (foreground_slots, background_idx)
    """
    alpha = alpha_maps[0, :, 0]  # (N, H, W)
    N = alpha.shape[0]
    flat = alpha.view(N, -1)  # (N, H*W)
    total = flat.sum(dim=-1)  # (N,)
    bg_idx = total.argmax().item()

    scores = flat.var(dim=-1)
    candidates = [(i, scores[i].item()) for i in range(N) if i != bg_idx]
    candidates.sort(key=lambda x: x[1], reverse=True)
    fg_slots = [i for i, _ in candidates[:n_slots]]
    return fg_slots, bg_idx


def compute_slot_similarity(attn_maps: torch.Tensor) -> torch.Tensor:
    """计算 slot attention 图的余弦相似度矩阵。
    attn_maps: (1, 1, N, N_feat)
    Returns: (N, N) 相似度矩阵
    """
    attn = attn_maps[0, 0]  # (N, N_feat)
    norm = attn / (attn.norm(dim=-1, keepdim=True) + 1e-8)
    return norm @ norm.T


def pick_foreground_pairs(bg_idx: int, num_slots: int, n_pairs: int = 1,
                          sim_matrix: Optional[torch.Tensor] = None,
                          sim_threshold: float = 0.3) -> List[Tuple[int, int]]:
    """从前景 slot 中挑选可交换对。
    如果提供了 sim_matrix，只选择 attention 图差异大的 pair（cos sim < threshold），
    避免交换注意同一物体的 slot。
    """
    foreground = [i for i in range(num_slots) if i != bg_idx]
    if len(foreground) < 2:
        return []
    pairs = []
    for i in range(0, len(foreground) - 1, 2):
        if len(pairs) >= n_pairs:
            break
        a, b = foreground[i], foreground[i + 1]
        if sim_matrix is not None:
            sim = sim_matrix[a, b].item()
            if sim > sim_threshold:
                continue
        pairs.append((a, b))
    return pairs


def filter_active_slots(attn_maps: torch.Tensor, alpha_threshold: float = 0.05) -> List[int]:
    """标记活跃 slot（attention 不是均匀分布，即有实际关注的区域）。
    均匀分布的 slot 视为空，不参与交换。
    """
    attn = attn_maps[0, 0]
    variances = attn.var(dim=-1)
    bg_idx = variances.argmin().item()
    active = []
    bg_var = variances[bg_idx].item()
    for i in range(attn.shape[0]):
        if i == bg_idx:
            continue
        if variances[i].item() > bg_var * (1.0 + alpha_threshold):
            active.append(i)
    return active


@torch.no_grad()
def run_interpret_swap(model, frames, swap_pairs, burnin, rollout, device,
                       debug=True, mode='swap'):
    B = frames.shape[0]
    static_dim = getattr(model.config, 'static_dim', 128)
    dyn_core_dim = getattr(model.config, 'dynamic_dim', 128)
    pos_enc_dim = getattr(model.config, 'pos_enc_dim', 8)
    slot_dim = static_dim + dyn_core_dim + pos_enc_dim
    freeze_C = getattr(model.config, 'freeze_C', False)
    num_slots = getattr(model.config, 'num_slots', 7)
    buf_sz = getattr(model.config, 'buffer_len', burnin + rollout)
    D_pos = model.POS_EMBED_DIM

    if debug:
        print(f"\n  [debug] slot_dim={slot_dim}, freeze_C={freeze_C}, mode={mode}")

    enc_features = model.encoder(frames)
    _, _, N_feat, _ = enc_features.shape
    grid_sz = int(N_feat ** 0.5)

    Z_buffer = torch.zeros(B, buf_sz, num_slots, slot_dim, device=device)
    slots = None
    for t in range(burnin):
        feat_t = enc_features[:, t]
        slots, attn = model._sa(feat_t, slots, t)
        centroid = model._compute_slot_centroid(attn, grid_sz)
        pe_32 = model._reconstruct_pe(centroid)
        slots_pe = slots.clone()
        slots_pe[:, :, -D_pos:] = slots_pe[:, :, -D_pos:] + pe_32
        Z_core = model.f_z(slots)
        p = model._encode_pos_to_zd(centroid, pos_enc_dim)
        Z_buffer[:, t] = torch.cat([Z_core, p], dim=-1)

    burnin_Z = Z_buffer[:, :burnin]

    # 前景/背景检测（用最后一帧 burnin decode 的 alpha）
    with torch.no_grad():
        last_P = model._decode_pe_from_zd(burnin_Z[:, -1, :, static_dim + dyn_core_dim:])
        last_S_raw = model.f_z.inverse(burnin_Z[:, -1, :, :static_dim + dyn_core_dim])
        last_S = last_S_raw.clone()
        last_S[:, :, -D_pos:] = last_S[:, :, -D_pos:] + last_P
        _, last_alpha = model.decoder(last_S, return_alpha=True)
    fg_slots, bg_idx = find_foreground_slots(last_alpha, n_slots=2)

    if swap_pairs is None:
        if len(fg_slots) >= 2:
            swap_pairs = [(fg_slots[0], fg_slots[1])]
        else:
            swap_pairs = []
    elif isinstance(swap_pairs, list) and len(swap_pairs) > 0:
        pass
    else:
        swap_pairs = []

    if debug:
        print(f"  [debug] bg_idx={bg_idx}, fg_slots={fg_slots}, swap_pairs={swap_pairs}")

    global_C = model.predictor.compute_C(burnin_Z) if freeze_C else None

    if debug and freeze_C and global_C is not None:
        print(f"  [debug] global_C.shape={global_C.shape}")
        for a, b in swap_pairs:
            diff = (global_C[0, a] - global_C[0, b]).norm().item()
            print(f"  [debug] C diff between slot {a} and {b}: ||C{a}-C{b}|| = {diff:.6f}")

    def rollout_fn(c_mod_fn=None):
        pred_Z_list = []
        cur_Z = Z_buffer[:, burnin - 1]
        cur_buffer = Z_buffer.clone()
        for step in range(rollout):
            if freeze_C:
                C_use = global_C
            else:
                C_use = cur_Z[:, :, :static_dim]
            if c_mod_fn is not None:
                B_, N_ = cur_Z.shape[:2]
                C_use = c_mod_fn(C_use, cur_Z, step)
            out = model.predictor(cur_Z, cur_buffer[:, :burnin + step], C=C_use, return_energy=False)
            next_Z = out
            pred_Z_list.append(next_Z)
            if burnin + step < buf_sz:
                cur_buffer[:, burnin + step] = next_Z
            cur_Z = next_Z
        return torch.stack(pred_Z_list, dim=1)

    normal_Z = rollout_fn(c_mod_fn=None)

    def swap_fn(C_use, cur_z, step):
        swapped = C_use.clone()
        for a, b in swap_pairs:
            tmp = swapped[:, a].clone()
            swapped[:, a] = swapped[:, b]
            swapped[:, b] = tmp
        return swapped

    def ablate_fn(C_use, cur_z, step):
        ablated = C_use.clone()
        for a, _ in swap_pairs:
            ablated[:, a] = 0.0
        return ablated

    mod_fn = ablate_fn if mode == 'ablate' else swap_fn
    modified_Z = rollout_fn(c_mod_fn=mod_fn)

    # 解码：Z_core → f_z⁻¹ → S_raw, p → _decode_pe_from_zd → PE_32 → S = S_raw + PE_32
    def decode(z_tensor):
        frames = []
        alphas = []
        for t in range(rollout):
            Z_core = z_tensor[:, t, :, :static_dim + dyn_core_dim]
            p_pred = z_tensor[:, t, :, static_dim + dyn_core_dim:]
            S_raw = model.f_z.inverse(Z_core)
            P_recon = model._decode_pe_from_zd(p_pred)
            S = S_raw.clone()
            S[:, :, -D_pos:] = S[:, :, -D_pos:] + P_recon
            out, alpha = model.decoder(S, return_alpha=True)
            frames.append(out)
            alphas.append(alpha)
        return torch.stack(frames, dim=1), torch.stack(alphas, dim=1)

    dec_normal, dec_normal_alpha = decode(normal_Z)
    dec_modified, dec_modified_alpha = decode(modified_Z)

    with torch.no_grad():
        target_S_list = []
        s = slots
        for t in range(burnin, burnin + rollout):
            feat_t = enc_features[:, t]
            s, _ = model._sa(feat_t, s, t)
            s = model._add_sd_pos_encoding(s, _, grid_sz)
            target_S_list.append(s)
        target_S = torch.stack(target_S_list, dim=1)
    dec_target = torch.stack([model.decoder(target_S[:, t]) for t in range(rollout)], dim=1)

    return {
        "video_normal": dec_normal,
        "video_modified": dec_modified,
        "video_target": dec_target,
        "slot_alpha": dec_normal_alpha,
        "slot_alpha_modified": dec_modified_alpha,
        "bg_idx": bg_idx,
        "swap_pairs": swap_pairs,
        "mode": mode,
    }


def visualize_swap_result(result_dict, burnin, rollout, save_path):
    from PIL import Image, ImageDraw, ImageFont

    normal_vid = result_dict["video_normal"]
    modified_vid = result_dict["video_modified"]
    target_vid = result_dict["video_target"]
    slot_alpha = result_dict["slot_alpha"]
    slot_alpha_modified = result_dict["slot_alpha_modified"]
    bg_idx = result_dict["bg_idx"]
    swap_pairs = result_dict["swap_pairs"]
    mode = result_dict.get("mode", "swap")

    n_cols = rollout
    extra_rows = 2 if swap_pairs else 0
    n_rows = 5 + 2 * extra_rows  # normal alpha + modified alpha
    S = 64
    font = ImageFont.load_default()
    W = n_cols * S
    H = n_rows * S
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    def put_img(tensor, row, col):
        arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        im = Image.fromarray((arr * 255).astype('uint8'))
        canvas.paste(im, (col * S, row * S))

    def put_heatmap(tensor, row, col):
        arr = tensor.detach().cpu().clamp(0, 1).numpy()
        arr = (arr * 255).astype('uint8')
        im = Image.fromarray(arr, mode='L')
        canvas.paste(im, (col * S, row * S))

    for t in range(rollout):
        put_img(target_vid[0, t], 0, t)
        put_img(normal_vid[0, t], 1, t)
        put_img(modified_vid[0, t], 2, t)

        diff = (normal_vid[0, t] - modified_vid[0, t]).abs().mean(dim=0, keepdim=True)
        diff_amp = (diff * 10.0).clamp(0, 1).repeat(3, 1, 1)
        put_img(diff_amp, 3, t)

        mse = F.mse_loss(normal_vid[0, t], modified_vid[0, t]).item()
        draw.text((t * S + 2, 3 * S + 2), f"diff={mse:.6f}", fill=(0, 0, 0), font=font)

    for t in range(min(rollout, burnin)):
        draw.text((t * S + 2, 0 * S + 2), "GT", fill=(255, 255, 255), font=font)

    mode_label = f"{mode.upper()} Pred"
    labels = ['GT Rollout', 'Normal Pred', mode_label, '|N-M|×10', 'MSE vs GT',
              '', '']
    for i, label in enumerate(labels):
        draw.text((2, i * S + 2), label, fill=(0, 0, 0), font=font)

    # Alpha masks
    target_slots = []
    for a, b in swap_pairs:
        target_slots.extend([a, b])
    target_slots = target_slots[:2]

    # Normal alpha rows
    for row_idx, slot_idx in enumerate(target_slots):
        for t in range(rollout):
            alpha_t = slot_alpha[0, t, slot_idx, 0]
            alpha_t = alpha_t / (alpha_t.max() + 1e-8)
            put_heatmap(alpha_t, 5 + row_idx, t)
        draw.text((2, (5 + row_idx) * S + 2),
                  f"S{slot_idx} α (norm)", fill=(255, 0, 0), font=font)

    # Modified alpha rows
    for row_idx, slot_idx in enumerate(target_slots):
        for t in range(rollout):
            alpha_t = slot_alpha_modified[0, t, slot_idx, 0]
            alpha_t = alpha_t / (alpha_t.max() + 1e-8)
            put_heatmap(alpha_t, 5 + extra_rows + row_idx, t)
        draw.text((2, (5 + extra_rows + row_idx) * S + 2),
                  f"S{slot_idx} α ({mode})", fill=(255, 0, 0), font=font)

    for t in range(rollout):
        m_normal = F.mse_loss(normal_vid[0, t], target_vid[0, t]).item()
        m_modified = F.mse_loss(modified_vid[0, t], target_vid[0, t]).item()
        draw.text((t * S + 2, 4 * S + 2), f"N={m_normal:.4f}", fill=(0, 0, 0), font=font)
        draw.text((t * S + 2, 4 * S + S // 2 + 2), f"M={m_modified:.4f}", fill=(0, 0, 0), font=font)

    info = f"BG slot={bg_idx}, mode={mode}, target_slots={list(swap_pairs[:2])}"
    draw.text((2, H - 14), info, fill=(128, 0, 128), font=font)

    canvas.save(save_path)
    return save_path
