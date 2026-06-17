import torch
import torch.nn.functional as F
from typing import List, Tuple


def find_background_slot(attn_maps: torch.Tensor) -> int:
    attn = attn_maps[0, 0]
    variances = attn.var(dim=-1)
    return variances.argmin().item()


def pick_foreground_pairs(bg_idx: int, num_slots: int, n_pairs: int = 1) -> List[Tuple[int, int]]:
    foreground = [i for i in range(num_slots) if i != bg_idx]
    if len(foreground) < 2:
        return []
    pairs = []
    for i in range(0, len(foreground) - 1, 2):
        if len(pairs) >= n_pairs:
            break
        pairs.append((foreground[i], foreground[i + 1]))
    return pairs


@torch.no_grad()
def run_interpret_swap(model, frames, swap_pairs, burnin, rollout, device, debug=True):
    B = frames.shape[0]
    static_dim = getattr(model.config, 'static_dim', 128)
    dynamic_dim = getattr(model.config, 'dynamic_dim', 128)
    slot_dim = static_dim + dynamic_dim
    freeze_C = getattr(model.config, 'freeze_C', False)
    num_slots = getattr(model.config, 'num_slots', 7)
    buf_sz = getattr(model.config, 'buffer_len', burnin + rollout)

    if debug:
        print(f"\n  [debug] static_dim={static_dim}, dynamic_dim={dynamic_dim}, freeze_C={freeze_C}")

    enc_features = model.encoder(frames)
    _, _, N_feat, _ = enc_features.shape
    grid_sz = int(N_feat ** 0.5)

    Z_buffer = torch.zeros(B, buf_sz, num_slots, slot_dim, device=device)
    burnin_attn = []
    slots = None
    for t in range(burnin):
        feat_t = enc_features[:, t]
        slots, attn = model._sa(feat_t, slots, t)
        slots = model._add_sd_pos_encoding(slots, attn, grid_sz)
        Z_t = model.f_z(slots)
        Z_buffer[:, t] = Z_t
        burnin_attn.append(attn)

    burnin_Z = Z_buffer[:, :burnin]

    last_attn = burnin_attn[-1]
    bg_idx = find_background_slot(last_attn)

    if swap_pairs is None:
        swap_pairs = pick_foreground_pairs(bg_idx, num_slots, n_pairs=1)

    global_C = model.predictor.compute_C(burnin_Z) if freeze_C else None

    if debug and freeze_C and global_C is not None:
        print(f"  [debug] global_C.shape={global_C.shape}")
        for a, b in swap_pairs:
            diff = (global_C[0, a] - global_C[0, b]).norm().item()
            print(f"  [debug] C diff between slot {a} and {b}: ||C{a}-C{b}|| = {diff:.6f}")

    def rollout_fn(c_mod_fn=None):
        pred_Z_list = []
        cur_Z = Z_t
        cur_buffer = Z_buffer.clone()
        for step in range(rollout):
            if freeze_C:
                C_use = global_C
            else:
                C_use = cur_Z[:, :, :static_dim]
            if c_mod_fn is not None:
                B_, N_ = cur_Z.shape[:2]
                C_use = c_mod_fn(C_use, cur_Z, step)
                if debug and step == 0 and freeze_C:
                    for a, b in swap_pairs:
                        cd = (C_use[0, a] - C_use[0, b]).norm().item()
                        print(f"  [debug] after swap: ||C{a}-C{b}|| = {cd:.6f}")
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
        return torch.zeros_like(swapped)

    swapped_Z = rollout_fn(c_mod_fn=swap_fn)

    z_diff = (normal_Z - swapped_Z).norm().item()
    if debug:
        print(f"  [debug] ||normal_Z - swapped_Z|| = {z_diff:.6f}")
        if z_diff < 1e-8:
            print(f"  [debug] WARNING: normal and swapped Z are identical!")

    pred_S_normal = torch.stack([model.f_z.inverse(normal_Z[:, t]) for t in range(rollout)], dim=1)
    pred_S_swapped = torch.stack([model.f_z.inverse(swapped_Z[:, t]) for t in range(rollout)], dim=1)

    dec_normal = torch.stack([model.decoder(pred_S_normal[:, t]) for t in range(rollout)], dim=1)
    dec_swapped = torch.stack([model.decoder(pred_S_swapped[:, t]) for t in range(rollout)], dim=1)

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
        "video_swapped": dec_swapped,
        "video_target": dec_target,
        "bg_idx": bg_idx,
        "swap_pairs": swap_pairs,
    }


def visualize_swap_result(result_dict, burnin, rollout, save_path):
    from PIL import Image, ImageDraw, ImageFont

    normal_vid = result_dict["video_normal"]
    swapped_vid = result_dict["video_swapped"]
    target_vid = result_dict["video_target"]
    bg_idx = result_dict["bg_idx"]
    swap_pairs = result_dict["swap_pairs"]

    n_cols = rollout
    n_rows = 5
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

    for t in range(rollout):
        put_img(target_vid[0, t], 0, t)
        put_img(normal_vid[0, t], 1, t)
        put_img(swapped_vid[0, t], 2, t)

        diff = (normal_vid[0, t] - swapped_vid[0, t]).abs().mean(dim=0, keepdim=True)
        diff_amp = (diff * 10.0).clamp(0, 1).repeat(3, 1, 1)
        put_img(diff_amp, 3, t)

        mse_swap_normal = F.mse_loss(normal_vid[0, t], swapped_vid[0, t]).item()
        draw.text((t * S + 2, 3 * S + 2), f"diff={mse_swap_normal:.6f}", fill=(0, 0, 0), font=font)

    for t in range(min(rollout, burnin)):
        draw.text((t * S + 2, 0 * S + 2), "GT", fill=(255, 255, 255), font=font)

    labels = ['GT Rollout', 'Normal Pred', 'Swapped Pred', '|N-S|×10', 'MSE vs GT']
    for i, label in enumerate(labels):
        draw.text((2, i * S + 2), label, fill=(0, 0, 0), font=font)

    for t in range(rollout):
        mn = F.mse_loss(normal_vid[0, t], target_vid[0, t]).item()
        ms = F.mse_loss(swapped_vid[0, t], target_vid[0, t]).item()
        draw.text((t * S + 2, 4 * S + 2), f"N={mn:.4f}", fill=(0, 0, 0), font=font)
        draw.text((t * S + 2, 4 * S + S // 2 + 2), f"S={ms:.4f}", fill=(0, 0, 0), font=font)

    swap_desc = f"BG slot={bg_idx}, swap={swap_pairs}"
    draw.text((2, H - 14), swap_desc, fill=(128, 0, 128), font=font)

    canvas.save(save_path)
    return save_path
