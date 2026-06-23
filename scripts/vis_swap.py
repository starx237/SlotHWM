#!/usr/bin/env python3
import sys, os, yaml
import numpy as np
from colorsys import hsv_to_rgb
import torch
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset
from PIL import Image, ImageDraw, ImageFont

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sample_idx = int(sys.argv[1])
step = int(sys.argv[2])
mode = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] in ('cswap', 'sswap') else 'cswap'
config_path = sys.argv[4] if len(sys.argv) > 4 else 'config/pretrain_obj3d.yaml'
workdir = sys.argv[5] if len(sys.argv) > 5 else None
slot_a_arg = sys.argv[6] if len(sys.argv) > 6 else 'auto'
slot_b_arg = sys.argv[7] if len(sys.argv) > 7 else 'auto'

with open(config_path) as f:
    cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)
if workdir:
    cfg.workdir = workdir

is_pretrain = getattr(cfg, 'pretrain', True)

model = SlotDynamicsModel(cfg).to(device)
ckpt_dir = os.path.join(cfg.workdir, 'checkpoints')
ckpt_path = os.path.join(ckpt_dir, f'step_{step}.pt')
if not os.path.isfile(ckpt_path):
    print(f'Checkpoint not found: {ckpt_path}')
    sys.exit(0)
sd = torch.load(ckpt_path, map_location=device)
state = sd.get('model', sd)
matched = {}
for mk in model.state_dict():
    mk_clean = mk.replace('_orig_mod.', '')
    if mk_clean in state: matched[mk] = state[mk_clean]
    elif mk in state: matched[mk] = state[mk]
model.load_state_dict(matched, strict=False)
model.eval()

burnin = getattr(cfg, 'burnin_frames', 6)
rollout = getattr(cfg, 'rollout_frames', 0) if not is_pretrain else 0
total_frames = burnin + rollout
ds = OBJ3DDataset(data_path=cfg_dict.get('data_root', './data/obj3d'), num_frames=total_frames, stride=getattr(cfg, 'slide_stride', 4), subsample=getattr(cfg, 'subsample', 2))
loader = ds.get_dataloader(batch_size=1, shuffle=True, num_workers=0)

def detect_foreground_slots(alpha, rgb):
    B, N, C, H, W = rgb.shape
    dominant = alpha.argmax(dim=1).squeeze(1)
    scores = []
    for j in range(N):
        mask = (dominant[0] == j) & (alpha[0, j, 0] > 0.3)
        if mask.sum() < 20:
            scores.append(-1.0)
            continue
        pix = rgb[0, j, :, mask]
        sat = pix.std(dim=0).mean().item()
        scores.append(sat)
    scores = np.array(scores)
    fg_idx = np.argsort(-scores).tolist()
    bg_idx = np.where(scores < 0.02)[0]
    return fg_idx, bg_idx, scores

for i, batch in enumerate(loader):
    if i != sample_idx:
        continue
    frames = batch['video'].to(device)

    with torch.no_grad():
        out = model(frames)

    if is_pretrain:
        slots_seq = out['slots']['corrected']       # (1, T, N, 67)
        recon_seq = out['outputs']['video_burnin']   # (1, T, 3, H, W)

        slots = slots_seq[:, -1]
        _, alpha, rgb = model.decoder(slots, return_rgb=True)
        fg_idxs, bg_idxs, scores = detect_foreground_slots(alpha, rgb)

        if len(fg_idxs) < 2:
            print(f'Only {len(fg_idxs)} foreground slots, skipping swap')
            continue

        if slot_a_arg != 'auto':
            swap_a = int(slot_a_arg)
            swap_b = int(slot_b_arg)
            if swap_a in bg_idxs or swap_b in bg_idxs:
                print(f'Warning: slot {swap_a if swap_a in bg_idxs else swap_b} appears to be background')
        else:
            swap_a, swap_b = fg_idxs[0], fg_idxs[1]
            print(f'Auto-selected: {swap_a} (score={scores[swap_a]:.4f}), {swap_b} (score={scores[swap_b]:.4f})')

        swapped_seq = slots_seq.clone()
        if mode == 'cswap':
            app_a = swapped_seq[:, :, swap_a, :-3].clone()
            app_b = swapped_seq[:, :, swap_b, :-3].clone()
            swapped_seq[:, :, swap_a, :-3] = app_b
            swapped_seq[:, :, swap_b, :-3] = app_a
        else:
            tmp = swapped_seq[:, :, swap_a, -1:].clone()
            swapped_seq[:, :, swap_a, -1:] = swapped_seq[:, :, swap_b, -1:]
            swapped_seq[:, :, swap_b, -1:] = tmp

        swapped_recon_list = []
        swapped_alpha_list = []
        swapped_rgb_list = []
        for t in range(burnin):
            sr, sa, srg = model.decoder(swapped_seq[:, t], return_rgb=True)
            swapped_recon_list.append(sr[0])
            swapped_alpha_list.append(sa[0])
            swapped_rgb_list.append(srg[0])

        orig_alpha_list = []
        orig_rgb_list = []
        for t in range(burnin):
            _, oa, org = model.decoder(slots_seq[:, t], return_rgb=True)
            orig_alpha_list.append(oa[0])
            orig_rgb_list.append(org[0])

        n_display = burnin
        n_cols = min(burnin, 6)
        display_gt = frames[0, :burnin]
        display_recon = recon_seq[0]
        display_swapped_recon = swapped_recon_list

    else:
        # === FINETUNE MODE ===
        with torch.no_grad():
            feat = model._encode_features(frames)
        B = 1
        buf_sz = getattr(cfg, 'buffer_len', total_frames)
        slot_dim_z = model.static_dim + model.dynamic_dim

        slots = None
        gru2_hidden = None
        prev_appearance = None
        burnin_Z_list = []
        for t in range(burnin):
            if t > 0 and slots is not None:
                new_appearance, gru2_hidden = model._gru2_step(prev_appearance, gru2_hidden)
                slots = torch.cat([
                    new_appearance,
                    slots[:, :, -3:-1].contiguous(),
                    slots[:, :, -1:].contiguous(),
                ], dim=-1)
            with torch.no_grad():
                slots, attn = model._sa(feat[:, t], slots, t)
            prev_appearance = slots[:, :, :-3].detach()
            if t == 0:
                BN = prev_appearance.shape[0] * prev_appearance.shape[1]
                gru2_hidden = torch.zeros(BN, model.gru2_hidden_dim, device=device)
                gru2_hidden = model.gru2(
                    prev_appearance.reshape(-1, model.appearance_dim),
                    gru2_hidden,
                )
            Z_core = model.f_z(slots[:, :, :model.appearance_dim])
            Z_full = torch.cat([Z_core, slots[:, :, -3:]], dim=-1)
            burnin_Z_list.append(Z_full)

        freeze_C = getattr(cfg, 'freeze_C', False)
        global_C = model.predictor.compute_C(torch.stack(burnin_Z_list, dim=1)) if freeze_C else None

        # Original rollout
        Z_buffer_orig = list(burnin_Z_list)
        pred_Z_list = []
        cur_Z = burnin_Z_list[-1]
        for t in range(rollout):
            C_use = global_C if freeze_C else cur_Z[:, :, :model.static_dim]
            Z_buf_t = torch.stack(Z_buffer_orig[:burnin + t], dim=1)
            next_Z = model.predictor(cur_Z, Z_buf_t, C=C_use)
            pred_Z_list.append(next_Z)
            Z_buffer_orig.append(next_Z)
            cur_Z = next_Z
        pred_Z = torch.stack(pred_Z_list, dim=1)

        # Decode original rollout
        pred_S_list = []
        for t in range(rollout):
            Z_app = pred_Z[:, t, :, :model.appearance_dim]
            pos_depth = pred_Z[:, t, :, model.appearance_dim:]
            S_raw = model.f_z.inverse(Z_app)
            S = torch.cat([S_raw, pos_depth], dim=-1)
            pred_S_list.append(S)
        pred_S = torch.stack(pred_S_list, dim=1)

        # Detect foreground from last burnin frame
        _, alpha, rgb = model.decoder(slots, return_rgb=True)
        fg_idxs, bg_idxs, scores = detect_foreground_slots(alpha, rgb)

        if len(fg_idxs) < 2:
            print(f'Only {len(fg_idxs)} foreground slots, skipping swap')
            continue

        if slot_a_arg != 'auto':
            swap_a = int(slot_a_arg)
            swap_b = int(slot_b_arg)
        else:
            swap_a, swap_b = fg_idxs[0], fg_idxs[1]
            print(f'Auto-selected: {swap_a} (score={scores[swap_a]:.4f}), {swap_b} (score={scores[swap_b]:.4f})')

        # === Swap: apply to burnin Z, recompute C, then full rollout ===
        swapped_burnin_Z_list = []
        for z in burnin_Z_list:
            sz = z.clone()
            if mode == 'cswap':
                app_a = sz[:, swap_a, :model.appearance_dim].clone()
                app_b = sz[:, swap_b, :model.appearance_dim].clone()
                sz[:, swap_a, :model.appearance_dim] = app_b
                sz[:, swap_b, :model.appearance_dim] = app_a
            else:
                depth_a = sz[:, swap_a, -1:].clone()
                depth_b = sz[:, swap_b, -1:].clone()
                sz[:, swap_a, -1:] = depth_b
                sz[:, swap_b, -1:] = depth_a
            swapped_burnin_Z_list.append(sz)

        swapped_global_C = model.predictor.compute_C(
            torch.stack(swapped_burnin_Z_list, dim=1)) if freeze_C else None

        # Rollout from swapped burnin
        Z_buffer_swap = list(swapped_burnin_Z_list)
        swapped_pred_Z_list = []
        cur_Z_swap = swapped_burnin_Z_list[-1]
        for t in range(rollout):
            C_use = swapped_global_C if freeze_C else cur_Z_swap[:, :, :model.static_dim]
            Z_buf_t = torch.stack(Z_buffer_swap[:burnin + t], dim=1)
            next_Z = model.predictor(cur_Z_swap, Z_buf_t, C=C_use)
            swapped_pred_Z_list.append(next_Z)
            Z_buffer_swap.append(next_Z)
            cur_Z_swap = next_Z

        # Decode swapped rollout
        swapped_pred_S_list = []
        for t in range(rollout):
            Z_app = swapped_pred_Z_list[t][:, :, :model.appearance_dim]
            pos_depth = swapped_pred_Z_list[t][:, :, model.appearance_dim:]
            S_raw = model.f_z.inverse(Z_app)
            S = torch.cat([S_raw, pos_depth], dim=-1)
            swapped_pred_S_list.append(S)

        # Decode original and swapped for per-slot display
        swapped_recon_list = []
        swapped_alpha_list = []
        swapped_rgb_list = []
        orig_alpha_list = []
        orig_rgb_list = []

        for t in range(rollout):
            # Original
            _, oa, org = model.decoder(pred_S[:, t], return_rgb=True)
            orig_alpha_list.append(oa[0])
            orig_rgb_list.append(org[0])
            # Swapped
            sr, sa, srg = model.decoder(swapped_pred_S_list[t], return_rgb=True)
            swapped_recon_list.append(sr[0])
            swapped_alpha_list.append(sa[0])
            swapped_rgb_list.append(srg[0])

        n_display = rollout
        n_cols = min(rollout, 10)
        display_gt = frames[0, burnin:burnin + rollout]
        display_recon = [model.decoder(pred_S[:, t], return_rgb=True)[0][0] for t in range(rollout)]
        display_swapped_recon = swapped_recon_list

    # === Common visualization ===
    n_slots = model.slot_attention.num_slots
    slot_order = [swap_a, swap_b] + [j for j in range(n_slots) if j not in (swap_a, swap_b)]

    row_offset = 3
    n_rows = row_offset + 2 * n_slots

    S_px = 64
    PAD = 2
    LABEL_W = 84
    canvas = Image.new('RGB', (LABEL_W + n_cols * (S_px + PAD), n_rows * (S_px + PAD)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    except:
        font = ImageFont.load_default()

    def put_rgb(t, r, c):
        arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        x = LABEL_W + c * (S_px + PAD)
        y = r * (S_px + PAD)
        canvas.paste(Image.fromarray((arr * 255).astype('uint8')), (x, y))

    def put_contrib(rgb_t, alpha_t, r, c):
        arr_rgb = rgb_t.detach().cpu().permute(1, 2, 0).numpy()
        arr_a = alpha_t.detach().cpu().numpy()
        disp = arr_rgb * arr_a[..., None] + (1.0 - arr_a[..., None])
        x = LABEL_W + c * (S_px + PAD)
        y = r * (S_px + PAD)
        canvas.paste(Image.fromarray((disp * 255).astype('uint8')), (x, y))

    def heatmap_rgb(val):
        h = 0.67 * (1.0 - val)
        return tuple(int(c * 255) for c in hsv_to_rgb(h, 1.0, 1.0))

    # Row 0: GT
    gt_label = 'Original' if is_pretrain else 'GT Rollout'
    draw.text((2, 2), gt_label, fill=(0, 0, 0), font=font)
    for t in range(n_cols):
        put_rgb(display_gt[t], 0, t)

    # Row 1: Recon
    recon_label = 'Recon' if is_pretrain else 'Pred Rollout'
    draw.text((2, (S_px + PAD) + 2), recon_label, fill=(0, 0, 0), font=font)
    for t in range(n_cols):
        put_rgb(display_recon[t], 1, t)

    # Row 2: Recon Swapped
    swap_label = 'Recon Swapped' if is_pretrain else 'Pred Swapped'
    draw.text((2, 2 * (S_px + PAD) + 2), swap_label, fill=(0, 0, 0), font=font)
    for t in range(n_cols):
        put_rgb(display_swapped_recon[t], 2, t)

    # Row 3+: slot contributions BEFORE/AFTER
    for idx_in_row, si in enumerate(slot_order):
        r_before = row_offset + 2 * idx_in_row
        r_after = row_offset + 2 * idx_in_row + 1
        tag = f'Slot {si}'
        if si == swap_a:
            tag += ' (←B)' if mode == 'cswap' else ' (depth ←B)'
        elif si == swap_b:
            tag += ' (←A)' if mode == 'cswap' else ' (depth ←A)'
        if si in bg_idxs:
            tag += ' (bg)'
        draw.text((2, r_before * (S_px + PAD) + 2), f'{tag} BEFORE', fill=(0, 0, 0), font=font)
        draw.text((2, r_after * (S_px + PAD) + 2), f'{tag} AFTER', fill=(0, 0, 0), font=font)
        for t in range(n_cols):
            put_contrib(orig_rgb_list[t][si], orig_alpha_list[t][si, 0], r_before, t)
            put_contrib(swapped_rgb_list[t][si], swapped_alpha_list[t][si, 0], r_after, t)

    # Slot position/depth info
    if is_pretrain:
        info_slots = out['slots']['corrected'][0, 0]
    else:
        info_slots = slots[0]
    y = n_rows * (S_px + PAD) + 2
    for si in range(n_slots):
        pos = info_slots[si, -3:-1]
        dep = info_slots[si, -1:]
        tag = "BG" if si in bg_idxs else (
            f"swap[{swap_a}]" if si == swap_a else
            f"swap[{swap_b}]" if si == swap_b else str(si)
        )
        draw.text((2, y), f"{tag}: pos({pos[0]:+.3f},{pos[1]:+.3f}) d({dep[0]:.3f})",
                  fill=(128, 0, 128), font=font)
        y += 14

    out_dir = os.path.join(cfg.workdir, 'vis_slots', f'step_{step}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{mode}_{sample_idx}_s{swap_a}s{swap_b}.png')
    canvas.save(out_path)
    print(f'Saved: {out_path}')
    break
