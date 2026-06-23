#!/usr/bin/env python3
import sys, os, yaml
import numpy as np
import torch
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset
from PIL import Image, ImageDraw, ImageFont

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sample_arg = sys.argv[1]
step = int(sys.argv[2])
config_path = sys.argv[3] if len(sys.argv) > 3 else 'config/pretrain_obj3d.yaml'
workdir = sys.argv[4] if len(sys.argv) > 4 else None

if '-' in sample_arg:
    start, end = map(int, sample_arg.split('-', 1))
    sample_ids = list(range(start, end + 1))
else:
    sample_ids = [int(sample_arg)]

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

n_iters = model.slot_attention.num_iterations

processed = 0
for i, batch in enumerate(loader):
    if i not in sample_ids:
        continue
    frames = batch['video'].to(device)

    with torch.no_grad():
        out = model(frames)

    if is_pretrain:
        recons = out['outputs']['video_burnin'][0]  # (T, 3, H, W)
        slots_seq = out['slots']['corrected'][0]     # (T, N, 67)
        n_display = burnin
        display_frames = frames[0, :burnin]
        display_recons = recons
        display_slots = slots_seq
        n_cols = min(n_display, 10)
    else:
        pred_recons = out['outputs']['video_pred'][0]   # (R, 3, H, W)
        pred_slots_S = out['slots']['predicted'][0]      # (R, N, 67)
        burnin_slots_S = out['slots']['corrected'][0]    # (B, N, 67)
        burnin_recons = out['outputs']['video_burnin'][0] # (B, 3, H, W)
        n_display = burnin + rollout
        display_frames = torch.cat([frames[0, :burnin], frames[0, burnin:burnin + rollout]], dim=0)
        display_recons = torch.cat([burnin_recons, pred_recons], dim=0)
        display_slots = torch.cat([burnin_slots_S, pred_slots_S], dim=0)
        n_cols = min(n_display, 16)

    all_alphas = []
    all_rgbs = []
    for t in range(n_display):
        _, alpha_t, rgb_t = model.decoder(display_slots[t:t+1], return_rgb=True)
        all_alphas.append(alpha_t[0, :, 0])  # (N, H, W)
        all_rgbs.append(rgb_t[0])            # (N, 3, H, W)

    alphas = torch.stack(all_alphas, dim=0)  # (T, N, H, W)
    rgbs = torch.stack(all_rgbs, dim=0)      # (T, N, 3, H, W)

    S = 64
    PAD = 2
    LABEL_W = 60
    n_slots = model.slot_attention.num_slots

    canvas = Image.new('RGB', (LABEL_W + n_cols * (S + PAD), (2 + n_slots) * (S + PAD)), (255, 255, 255))
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
        display = arr_rgb * arr_alpha + (1.0 - arr_alpha)
        x = LABEL_W + c * (S + PAD)
        y = r * (S + PAD)
        canvas.paste(Image.fromarray((display * 255).astype('uint8')), (x, y))

    row0_label = 'Video (burnin)' if is_pretrain else 'Video (burnin+rollout)'
    row1_label = 'Recon (burnin)' if is_pretrain else 'Recon+Pred (burnin+rollout)'

    draw.text((2, 2), row0_label, fill=(0, 0, 0), font=font)
    for t in range(n_cols):
        put_rgb(display_frames[t], 0, t)

    draw.text((2, (S + PAD) + 2), row1_label, fill=(0, 0, 0), font=font)
    for t in range(n_cols):
        put_rgb(display_recons[t], 1, t)

    if not is_pretrain and burnin < n_cols:
        for row in range(2 + n_slots):
            x = LABEL_W + burnin * (S + PAD) - 1
            y0 = row * (S + PAD)
            draw.line([(x, y0), (x, y0 + S)], fill=(255, 0, 0), width=2)

    for j in range(n_slots):
        y = (2 + j) * (S + PAD) + 2
        draw.text((2, y), f'Slot {j}', fill=(0, 0, 0), font=font)
        for t in range(n_cols):
            put_contribution(rgbs[t, j], alphas[t, j], 2 + j, t)

    out_dir = os.path.join(cfg.workdir, 'vis_slots', f'step_{step}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'sample_{i}.png')
    canvas.save(out_path)
    print(f'Saved: {out_path}')
    processed += 1
    if processed >= len(sample_ids):
        break
