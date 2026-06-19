import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from models.misc import create_coordinate_grid


def find_foreground_slots(alpha_maps: torch.Tensor, n_slots: int = 2) -> Tuple[List[int], int]:
    alpha = alpha_maps[0, :, 0]
    N = alpha.shape[0]
    flat = alpha.view(N, -1)
    total = flat.sum(dim=-1)
    bg_idx = total.argmax().item()

    scores = flat.var(dim=-1)
    candidates = [(i, scores[i].item()) for i in range(N) if i != bg_idx]
    candidates.sort(key=lambda x: x[1], reverse=True)
    fg_slots = [i for i, _ in candidates[:n_slots]]
    return fg_slots, bg_idx


def compute_slot_similarity(attn_maps: torch.Tensor) -> torch.Tensor:
    attn = attn_maps[0, 0]
    norm = attn / (attn.norm(dim=-1, keepdim=True) + 1e-8)
    return norm @ norm.T


def pick_foreground_pairs(bg_idx: int, num_slots: int, n_pairs: int = 1,
                          sim_matrix: Optional[torch.Tensor] = None,
                          sim_threshold: float = 0.3) -> List[Tuple[int, int]]:
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
    static_dim = getattr(model.config, 'static_dim', 34)
    dynamic_dim = getattr(model.config, 'dynamic_dim', 33)
    slot_dim = static_dim + dynamic_dim
    appearance_dim = getattr(model.config, 'appearance_dim', 64)
    freeze_C = getattr(model.config, 'freeze_C', False)
    num_slots = getattr(model.config, 'num_slots', 6)
    buf_sz = getattr(model.config, 'buffer_len', burnin + rollout)

    # Z 中前 64 维是可交换的 appearance 部分（f_z 输出）
    swap_dim = static_dim + dynamic_dim - 3  # = appearance_dim = 64

    if debug:
        print(f"\n  [debug] slot_dim={slot_dim}, swap_dim={swap_dim}, freeze_C={freeze_C}, mode={mode}")

    feat = model.encoder(frames)
    B_e, T_e, N_feat, D_e = feat.shape
    grid_sz = int(N_feat ** 0.5)
    grid = create_coordinate_grid(grid_sz, grid_sz, frames.device)
    grid = grid.view(1, 1, N_feat, 2).expand(B_e, T_e, N_feat, 2)
    feat_with_grid = torch.cat([feat, grid], dim=-1)

    Z_buffer = torch.zeros(B, buf_sz, num_slots, slot_dim, device=device)
    slots = None
    for t in range(burnin):
        feat_t = feat_with_grid[:, t]
        slots, attn = model.slot_attention(feat_t, slots)
        Z_core = model.f_z(slots[:, :, :appearance_dim])
        Z_full = torch.cat([Z_core, slots[:, :, -3:]], dim=-1)
        Z_buffer[:, t] = Z_full

    burnin_Z = Z_buffer[:, :burnin]

    # 前景/背景检测（用最后一帧 burnin decode 的 alpha）
    with torch.no_grad():
        Z_last = burnin_Z[:, -1]
        S_appearance = model.f_z.inverse(Z_last[:, :, :appearance_dim])
        S = torch.cat([S_appearance, Z_last[:, :, appearance_dim:]], dim=-1)
        _, last_alpha = model.decoder(S, return_alphas=True)
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
        # 交换 Z 的前 swap_dim 维（appearance 部分 = f_z 输出全部）
        swapped = cur_z.clone()
        for a, b in swap_pairs:
            tmp = swapped[:, a, :swap_dim].clone()
            swapped[:, a, :swap_dim] = swapped[:, b, :swap_dim]
            swapped[:, b, :swap_dim] = tmp
        return C_use  # C_use 不变，直接在 cur_z 上修改后由调用者处理

    def ablate_fn(C_use, cur_z, step):
        ablated = C_use.clone()
        for a, _ in swap_pairs:
            ablated[:, a] = 0.0
        return ablated

    # 交换模式：修改 cur_Z 的 appearance 部分
    if mode == 'swap':
        modified_Z_list = []
        cur_Z = Z_buffer[:, burnin - 1]
        cur_buffer = Z_buffer.clone()
        for step in range(rollout):
            C_use = global_C if freeze_C else cur_Z[:, :, :static_dim]
            # 在 predictor 调用前，修改 cur_Z 的 appearance 部分
            cur_Z_modified = cur_Z.clone()
            for a, b in swap_pairs:
                tmp = cur_Z_modified[:, a, :swap_dim].clone()
                cur_Z_modified[:, a, :swap_dim] = cur_Z_modified[:, b, :swap_dim]
                cur_Z_modified[:, b, :swap_dim] = tmp
            out = model.predictor(cur_Z_modified, cur_buffer[:, :burnin + step], C=C_use, return_energy=False)
            next_Z = out
            modified_Z_list.append(next_Z)
            if burnin + step < buf_sz:
                cur_buffer[:, burnin + step] = next_Z
            cur_Z = next_Z
        modified_Z = torch.stack(modified_Z_list, dim=1)
    else:
        modified_Z = rollout_fn(c_mod_fn=ablate_fn)

    # 解码：Z → f_z.inverse(appearance_part) + pos_depth → S → decoder
    def decode(z_tensor):
        frames_out = []
        alphas = []
        for t in range(rollout):
            Z_appearance = z_tensor[:, t, :, :appearance_dim]
            pos_depth = z_tensor[:, t, :, appearance_dim:]
            S_raw = model.f_z.inverse(Z_appearance)
            S = torch.cat([S_raw, pos_depth], dim=-1)
            out_img, alpha = model.decoder(S, return_alphas=True)
            frames_out.append(out_img)
            alphas.append(alpha)
        return torch.stack(frames_out, dim=1), torch.stack(alphas, dim=1)

    dec_normal, dec_normal_alpha = decode(normal_Z)
    dec_modified, dec_modified_alpha = decode(modified_Z)

    with torch.no_grad():
        target_S_list = []
        s = slots
        for t in range(burnin, burnin + rollout):
            feat_t = feat_with_grid[:, t]
            s, attn_t = model.slot_attention(feat_t, s)
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
    n_rows = 5 + 2 * extra_rows
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

    mode_label = f"{mode.upper()} Pred"
    labels = ['GT Rollout', 'Normal Pred', mode_label, '|N-M|×10', 'MSE vs GT']
    for i, label in enumerate(labels):
        draw.text((2, i * S + 2), label, fill=(0, 0, 0), font=font)

    target_slots = []
    for a, b in swap_pairs:
        target_slots.extend([a, b])
    target_slots = target_slots[:2]

    for row_idx, slot_idx in enumerate(target_slots):
        for t in range(rollout):
            alpha_t = slot_alpha[0, t, slot_idx, 0]
            alpha_t = alpha_t / (alpha_t.max() + 1e-8)
            put_heatmap(alpha_t, 5 + row_idx, t)
        draw.text((2, (5 + row_idx) * S + 2),
                  f"S{slot_idx} α (norm)", fill=(255, 0, 0), font=font)

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
