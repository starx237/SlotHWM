#!/usr/bin/env python3
import sys, os, yaml
import numpy as np
import torch
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from isa_pretrain.models.isa_dynamics import SlotDynamicsModelISA
from data.obj3d_dataset import OBJ3DDataset
from PIL import Image, ImageDraw, ImageFont

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sample_arg = sys.argv[1]
step = int(sys.argv[2])

if '-' in sample_arg:
    start, end = map(int, sample_arg.split('-', 1))
    sample_ids = list(range(start, end + 1))
else:
    sample_ids = [int(sample_arg)]

with open('isa_pretrain/config/isa_obj3d.yaml') as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))
cfg.workdir = 'experiments/isa_obj3d'

model = SlotDynamicsModelISA(cfg).to(device)
ckpt_path = f'experiments/isa_obj3d/checkpoints/step_{step}.pt'
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

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=6, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=1, shuffle=True, num_workers=0)

burnin_iters = getattr(cfg, 'burnin_iters', 3)
rollout_iters = getattr(cfg, 'rollout_iters', 1)

processed = 0
for i, batch in enumerate(loader):
    if i not in sample_ids:
        continue
    frames = batch['video'].to(device)
    feat = model._encode_features(frames)

    all_recons = []
    all_alphas = []
    all_rgbs = []
    slots = None
    for t in range(feat.shape[1]):
        iters = burnin_iters if t == 0 else rollout_iters
        slots, _ = model.slot_attention(feat[:, t], slots, num_iterations=iters)
        recon, alpha, rgb = model.decoder(slots, return_rgb=True)
        all_recons.append(recon[0])
        all_alphas.append(alpha[0, :, 0])
        all_rgbs.append(rgb[0])

    recons = torch.stack(all_recons, dim=0)
    alphas = torch.stack(all_alphas, dim=0)
    rgbs = torch.stack(all_rgbs, dim=0)

    S = 64
    PAD = 2
    LABEL_W = 60
    n_frames = 6
    n_slots = model.slot_attention.num_slots

    canvas = Image.new('RGB', (LABEL_W + n_frames * (S + PAD), (2 + n_slots) * (S + PAD)), (255, 255, 255))
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

    def put_contribution(rgb_slot, alpha_slot, r, c):
        arr_rgb = rgb_slot.detach().cpu().permute(1, 2, 0).numpy()
        arr_alpha = alpha_slot.detach().cpu().numpy()
        arr_alpha = np.expand_dims(arr_alpha, axis=-1)
        display = arr_rgb * arr_alpha + (1.0 - arr_alpha)  # 0=white
        x = LABEL_W + c * (S + PAD)
        y = r * (S + PAD)
        canvas.paste(Image.fromarray((display * 255).astype('uint8')), (x, y))

    draw.text((2, 2), 'Video', fill=(0, 0, 0), font=font)
    for t in range(n_frames):
        put_rgb(frames[0, t], 0, t)

    draw.text((2, (S + PAD) + 2), 'Recon', fill=(0, 0, 0), font=font)
    for t in range(n_frames):
        put_rgb(recons[t], 1, t)

    for j in range(n_slots):
        y = (2 + j) * (S + PAD) + 2
        draw.text((2, y), f'Slot {j}', fill=(0, 0, 0), font=font)
        for t in range(n_frames):
            put_contribution(rgbs[t, j], alphas[t, j], 2 + j, t)

    out_dir = f'experiments/isa_obj3d/vis_slots/step_{step}'
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/sample_{i}.png'
    canvas.save(out_path)
    print(f'Saved: {out_path}')
    processed += 1
    if processed >= len(sample_ids):
        break
