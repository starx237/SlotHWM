#!/usr/bin/env python3
import sys, os, yaml
import numpy as np
from colorsys import hsv_to_rgb
import torch
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from isa_pretrain.models.isa_dynamics import SlotDynamicsModelISA
from data.obj3d_dataset import OBJ3DDataset
from PIL import Image, ImageDraw, ImageFont

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sample_idx = int(sys.argv[1])
step = int(sys.argv[2])
mode = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] in ('cswap', 'sswap') else 'cswap'
# Remaining args: slot_a slot_b (or 'auto')

with open('isa_pretrain/config/isa_obj3d.yaml') as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))
cfg.workdir = 'experiments/isa_obj3d'

model = SlotDynamicsModelISA(cfg).to(device)
ckpt_path = f'experiments/isa_obj3d/checkpoints/step_{step}.pt'
sd = torch.load(ckpt_path, map_location=device)
state = sd.get('model', sd)
matched = {}
for mk in model.state_dict():
    mk_clean = mk.replace('_orig_mod.', '')
    if mk_clean in state: matched[mk] = state[mk_clean]
    elif mk in state: matched[mk] = state[mk]
model.load_state_dict(matched, strict=False)
model.eval()

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=6, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=1, shuffle=True, num_workers=0)

def detect_foreground_slots(alpha, rgb):
    B, N, C, H, W = rgb.shape
    dominant = alpha.argmax(dim=1).squeeze(1)  # (B, H, W)
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
    feat = model._encode_features(frames)
    slots, attn = model.slot_attention(feat[:, 0], num_iterations=cfg.burnin_iters)
    recon, alpha, rgb = model.decoder(slots, return_rgb=True)

    fg_idxs, bg_idxs, scores = detect_foreground_slots(alpha, rgb)

    auto_slot_arg = sys.argv[4] if len(sys.argv) >= 5 else 'auto'

    if auto_slot_arg != 'auto':
        swap_a = int(sys.argv[4])
        swap_b = int(sys.argv[5])
        if swap_a in bg_idxs or swap_b in bg_idxs:
            print(f'Warning: slot {swap_a if swap_a in bg_idxs else swap_b} appears to be background (colorfulness={scores[swap_a if swap_a in bg_idxs else swap_b]:.4f})')
    else:
        swap_a, swap_b = fg_idxs[0], fg_idxs[1]
        print(f'Auto-selected: {swap_a} (score={scores[swap_a]:.4f}), {swap_b} (score={scores[swap_b]:.4f})')

    slots_swapped = slots.clone()
    if mode == 'cswap':
        app_a = slots_swapped[0, swap_a, :-3].clone()
        app_b = slots_swapped[0, swap_b, :-3].clone()
        slots_swapped[0, swap_a, :-3] = app_b
        slots_swapped[0, swap_b, :-3] = app_a
    else:
        tmp = slots_swapped[0, swap_a, -1:].clone()
        slots_swapped[0, swap_a, -1:] = slots_swapped[0, swap_b, -1:]
        slots_swapped[0, swap_b, -1:] = tmp

    recon_swapped, alpha_swapped, rgb_swapped = model.decoder(slots_swapped, return_rgb=True)

    _, attn_swapped = model.slot_attention(feat[:, 0], slots_swapped, num_iterations=cfg.rollout_iters)

    n_slots = model.slot_attention.num_slots
    attn_orig = attn[0].reshape(n_slots, 16, 16)
    attn_swap = attn_swapped[0].reshape(n_slots, 16, 16)
    attn_global_max = max(attn_orig.max().item(), attn_swap.max().item())

    S = 64
    PAD = 2
    LABEL_W = 84
    n_cols = 4
    rows = 3 + n_slots + 1
    canvas = Image.new('RGB', (LABEL_W + n_cols * (S + PAD), rows * (S + PAD)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except:
        font = ImageFont.load_default()

    def put_rgb(t, r, c):
        arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        x = LABEL_W + c * (S + PAD)
        y = r * (S + PAD)
        canvas.paste(Image.fromarray((arr * 255).astype('uint8')), (x, y))

    def put_contrib(rgb_t, alpha_t, r, c):
        arr_rgb = rgb_t.detach().cpu().permute(1, 2, 0).numpy()
        arr_a = alpha_t.detach().cpu().numpy()
        disp = arr_rgb * arr_a[..., None] + (1.0 - arr_a[..., None])
        x = LABEL_W + c * (S + PAD)
        y = r * (S + PAD)
        canvas.paste(Image.fromarray((disp * 255).astype('uint8')), (x, y))

    def heatmap_rgb(val):
        h = 0.67 * (1.0 - val)
        return tuple(int(c * 255) for c in hsv_to_rgb(h, 1.0, 1.0))

    def put_attn(attn_map, r, c, global_max):
        arr = attn_map.detach().cpu().numpy()
        arr = Image.fromarray(arr, mode='F')
        arr = arr.resize((S, S), Image.BILINEAR)
        arr = np.array(arr)
        norm_max = max(global_max, 0.004)
        arr = np.clip(arr / norm_max, 0, 1)
        rgb = np.zeros((S, S, 3), dtype='uint8')
        for py in range(S):
            for px in range(S):
                rgb[py, px] = heatmap_rgb(arr[py, px])
        x = LABEL_W + c * (S + PAD)
        y = r * (S + PAD)
        canvas.paste(Image.fromarray(rgb), (x, y))

    r = 0
    draw.text((2, 2), 'Image', fill=(0, 0, 0), font=font)
    put_rgb(frames[0, 0], r, 0)

    r = 1
    draw.text((2, (S + PAD) + 2), 'Recon', fill=(0, 0, 0), font=font)
    put_rgb(recon[0], r, 0)

    r = 2
    label = f'{mode} s{swap_a}↔s{swap_b}'
    draw.text((2, 2 * (S + PAD) + 2), label, fill=(0, 0, 0), font=font)
    put_rgb(recon_swapped[0], r, 0)

    for j in range(n_slots):
        r = 3 + j
        label = f'Slot {j}'
        if j in bg_idxs: label += ' (bg)'
        if j == swap_a: label += ' ←B'
        if j == swap_b: label += ' ←A'
        draw.text((2, r * (S + PAD) + 2), label, fill=(0, 0, 0), font=font)
        put_contrib(rgb[0, j], alpha[0, j, 0], r, 0)
        put_contrib(rgb_swapped[0, j], alpha_swapped[0, j, 0], r, 1)
        put_attn(attn_orig[j], r, 2, attn_global_max)
        put_attn(attn_swap[j], r, 3, attn_global_max)

    y = rows * (S + PAD) + 2
    for i in range(slots.shape[1]):
        pos = slots[0, i, -3:-1]
        dep = slots[0, i, -1:]
        tag = "BG" if i in bg_idxs else (
            f"swap[{swap_a}]" if i == swap_a else
            f"swap[{swap_b}]" if i == swap_b else str(i)
        )
        draw.text((2, y), f"{tag}: pos({pos[0]:+.3f},{pos[1]:+.3f}) d({dep[0]:.3f})",
                  fill=(128, 0, 128), font=font)
        y += 14

    out_dir = f'experiments/isa_obj3d/vis_slots/step_{step}'
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/{mode}_{sample_idx}_s{swap_a}s{swap_b}.png'
    canvas.save(out_path)
    print(f'Saved: {out_path}')
    break
