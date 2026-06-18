#!/usr/bin/env python3
import sys, os, yaml
import torch
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from isa_pretrain.models.isa_dynamics import SlotDynamicsModelISA
from data.obj3d_dataset import OBJ3DDataset
from PIL import Image, ImageDraw, ImageFont
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('isa_pretrain/config/isa_obj3d.yaml') as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))
cfg.workdir = 'experiments/isa_obj3d'

model = SlotDynamicsModelISA(cfg).to(device)
sd = torch.load('experiments/isa_obj3d/checkpoints/best.pt', map_location=device)
ckpt = sd.get('model', sd)
matched = {}
for mk in model.state_dict():
    mk_clean = mk.replace('_orig_mod.', '')
    if mk_clean in ckpt: matched[mk] = ckpt[mk_clean]
    elif mk in ckpt: matched[mk] = ckpt[mk]
model.load_state_dict(matched, strict=False)
model.eval()
sa = model.slot_attention

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=6, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=1, shuffle=True, num_workers=0)

font = ImageFont.load_default()
S = 64
PAD = 2

for sample_idx, batch in enumerate(loader):
    if sample_idx >= 5:
        break
    frames = batch['video'].to(device)
    feat = model._encode_features(frames)

    slots = None
    for t in range(feat.shape[1]):
        slots, attn_out = model.slot_attention(feat[:, t], slots)

    img = frames[0, feat.shape[1] - 1]  # use last burnin frame (matches slots)
    N = slots.shape[1]

    blended, alphas = model.decoder(slots, return_alphas=True)

    # Per-slot: attention map (8x8), alpha mask, decoded RGB, overlay
    attn_2d = attn_out[0].reshape(N, 8, 8)
    attn_64 = torch.nn.functional.interpolate(
        attn_out[0].unsqueeze(1).reshape(N, 1, 8, 8),
        size=(64, 64), mode='bilinear'
    )[:, 0]

    rows_per_slot = 1 + 1  # attention row + decoded row per slot
    n_rows = 1 + N * rows_per_slot + 1  # header + per-slot rows + summary
    canvas_w = S * 4 + PAD * 3
    canvas_h = n_rows * (S + 16)
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    def put_img(t, row, col):
        arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        canvas.paste(Image.fromarray((arr * 255).astype('uint8')), (col * (S + PAD), row * (S + 16)))

    def put_gray(t, row, col):
        arr = t.detach().cpu().clamp(0, 1).numpy()
        canvas.paste(Image.fromarray((arr * 255).astype('uint8'), mode='L'), (col * (S + PAD), row * (S + 16)))

    # Header row: GT, Blended, Diff, Certainty
    put_img(img, 0, 0)
    draw.text((2, 2), 'GT', fill=(0, 0, 0), font=font)
    put_img(blended[0], 0, 1)
    draw.text((S + PAD + 2, 2), 'Blended', fill=(0, 0, 0), font=font)
    diff = (img - blended[0]).abs().mean(dim=0, keepdim=True).expand(3, -1, -1)
    put_img(diff * 20, 0, 2)
    draw.text((2 * (S + PAD) + 2, 2), 'Diff×20', fill=(0, 0, 0), font=font)

    entropy = -(attn_out[0] * (attn_out[0] + 1e-8).log()).sum(dim=0)
    norm_entropy = 1 - entropy / math.log(N)
    norm_entropy_img = norm_entropy.unsqueeze(0).expand(3, -1, -1)
    put_img(norm_entropy_img, 0, 3)
    draw.text((3 * (S + PAD) + 2, 2), 'Certainty', fill=(0, 0, 0), font=font)

    def pos_str(pos):
        return f'({pos[0]:+.3f},{pos[1]:+.3f})'

    # Per-slot rows
    for j in range(N):
        base_row = 1 + j * rows_per_slot
        pos = slots[0, j, -4:-2]
        sc = slots[0, j, -2:]

        # Attention map
        a = attn_64[j]
        a_img = a.unsqueeze(0).expand(3, -1, -1)
        put_img(a_img, base_row, 0)
        draw.text((2, base_row * (S + 16) + 2), f'Slot {j} attn', fill=(0, 0, 0), font=font)

        # Alpha mask
        alpha = alphas[0, j]
        alpha_img = alpha.expand(3, -1, -1)
        put_img(alpha_img, base_row, 1)
        draw.text((S + PAD + 2, base_row * (S + 16) + 2), 'alpha', fill=(0, 0, 0), font=font)

        # Overlay
        overlay = img * 0.6 + a.unsqueeze(0).expand(3, -1, -1) * 0.4
        put_img(overlay, base_row, 2)
        draw.text((2 * (S + PAD) + 2, base_row * (S + 16) + 2), 'overlay', fill=(0, 0, 0), font=font)

        # Contribution to blend = decoded RGB × alpha
        slot_only = torch.cat([slots[:, j:j+1, :-4], slots[:, j:j+1, -4:-2], slots[:, j:j+1, -2:]], dim=-1)
        rgb_j = model.decoder(slot_only, return_alphas=False)[0]
        contrib = rgb_j * alpha
        put_img(contrib, base_row, 3)
        draw.text((3 * (S + PAD) + 2, base_row * (S + 16) + 2), 'contrib', fill=(0, 0, 0), font=font)

        # Text info below the images
        ty = base_row * (S + 16) + S + 2
        a_sum = attn_out[0, j].sum().item()
        com_y = (attn_2d[j].sum(dim=1) * torch.linspace(-1, 1, 8, device=device)).sum() / a_sum
        com_x = (attn_2d[j].sum(dim=0) * torch.linspace(-1, 1, 8, device=device)).sum() / a_sum
        peak_val, peak_idx = attn_2d[j].reshape(-1).max(0)
        py, px = peak_idx.item() // 8, peak_idx.item() % 8
        pyn = py / 7.5 * 2 - 1
        pxn = px / 7.5 * 2 - 1

        label = f'Slot {j}: p={pos_str(pos)} s=({sc[0]:.3f},{sc[1]:.3f}) com=({com_x:.3f},{com_y:.3f}) pk=({pxn:.3f},{pyn:.3f}) sum={a_sum:.3f}'
        draw.text((2, ty), label, fill=(0, 0, 0), font=font)

        # Classification
        ty_text = ty + 14
        if a_sum < 0.05:
            desc = 'BG (negligible)'
        elif sc[0] > 0.5 or sc[1] > 0.6:
            desc = f'BG (large scale {sc[0]:.2f},{sc[1]:.2f})'
        elif sc[0] < 0.2:
            desc = f'Object/Detail (small scale {sc[0]:.2f},{sc[1]:.2f})'
        else:
            desc = f'Unknown (scale={sc[0]:.2f},{sc[1]:.2f})'
        draw.text((2, ty_text), desc, fill=(128, 0, 128), font=font)

    # Summary row
    last_row = 1 + N * rows_per_slot
    y_sum = last_row * (S + 16) + 2
    draw.text((2, y_sum), f'Avg entropy: {entropy.mean():.4f} / {math.log(N):.4f} (max)', fill=(0, 0, 0), font=font)
    max_per_pixel, _ = attn_out[0].max(dim=0)
    conf_pixels = (max_per_pixel > 0.3).float().mean().item()
    draw.text((2, y_sum + 14), f'Pixels with >0.3 max-attn: {conf_pixels*100:.1f}%', fill=(0, 0, 0), font=font)

    # Per-slot attention coverage analysis
    for j in range(N):
        slot_frac = (attn_64[j] > 0.1).float().mean().item()
        desc = 'BG' if j in [0, 1, 5, 6] else 'OBJ'
        draw.text((2, y_sum + 28 + j * 14),
                  f'  Slot {j} [{desc}]: attn>0.1 covers {slot_frac*100:.1f}% of image',
                  fill=(0, 0, 0), font=font)

    out_path = f'experiments/isa_obj3d/eval_images/debug_sample_{sample_idx}.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    print(f'Saved: {out_path}')

    # Also print text analysis
    print(f'\n{"="*60}')
    print(f'Sample {sample_idx} Per-Slot Analysis')
    print(f'{"="*60}')
    for j in range(N):
        pos = slots[0, j, -4:-2]
        sc = slots[0, j, -2:]
        a_sum = attn_out[0, j].sum().item()
        com_y = (attn_2d[j].sum(dim=1) * torch.linspace(-1, 1, 8, device=device)).sum() / a_sum
        com_x = (attn_2d[j].sum(dim=0) * torch.linspace(-1, 1, 8, device=device)).sum() / a_sum
        frac10 = (attn_64[j] > 0.1).float().mean().item()
        frac20 = (attn_64[j] > 0.2).float().mean().item()
        max_a = attn_64[j].max().item()
        print(f'  Slot {j}: pos({pos[0]:+.3f},{pos[1]:+.3f}) sc({sc[0]:.3f},{sc[1]:.3f})')
        print(f'          com=({com_x:.3f},{com_y:.3f}) sum={a_sum:.3f} max_attn={max_a:.4f}')
        print(f'          coverage: >0.1={frac10*100:.1f}% >0.2={frac20*100:.1f}%')

print('Done.')
