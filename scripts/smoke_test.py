#!/usr/bin/env python3
# Smoke test: overfit on a single frame (burnin=1, rollout=1)
# Tests whether encoder → slot attention → decoder can reconstruct an image.

import os, sys, math, time, yaml, glob, shutil
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from data.clevrer_dataset import CLEVRERDataset
from models.dynamics import SlotDynamicsModel
from train.trainer import Trainer, WandBLogger
from train import create_optimizer

# ── Config ──────────────────────────────────────────────────────────────────
WORKDIR = './experiments/smoke_test'
os.makedirs(os.path.join(WORKDIR, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(WORKDIR, 'tb_logs'), exist_ok=True)

LOG = 'log.txt'
def log(msg):
    ts = time.strftime('%H:%M:%S')
    with open(LOG, 'a') as f:
        f.write(f'[{ts}] {msg}\n')
    print(msg, flush=True)

# Load base CLEVRER config and override for smoke test
with open('config/clevrer.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg_dict['burnin_frames'] = 1
cfg_dict['rollout_frames'] = 1
cfg_dict['num_steps'] = 3000
cfg_dict['batch_size'] = 1
cfg_dict['num_workers'] = 0
cfg_dict['log_every'] = 10
cfg_dict['save_every'] = 500
cfg_dict['keep_last_checkpoints'] = 2
cfg_dict['warmup_steps'] = 100
cfg_dict['workdir'] = WORKDIR
cfg = SimpleNamespace(**cfg_dict)

# ── WandB ───────────────────────────────────────────────────────────────────
load_dotenv()
use_wandb = os.getenv('WANDB_ENABLED', 'false').lower() == 'true'
wandb_logger = WandBLogger(enabled=use_wandb)
if use_wandb:
    os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY', '')
    import wandb
    wandb_logger.init(
        project=os.getenv('WANDB_PROJECT', 'slotpi'),
        entity=os.getenv('WANDB_ENTITY', None),
        name='smoke_test_single_frame',
        config=cfg_dict,
        dir=WORKDIR,
        settings=wandb.Settings(sync_tensorboard=False),
    )
    wandb_proj = os.getenv('WANDB_PROJECT', 'slotpi')
    log(f'WandB enabled: project={wandb_proj}, name=smoke_test_single_frame')

# ── Data: one frame repeated twice ──────────────────────────────────────────
ds = CLEVRERDataset(data_path='data/clevrer', num_frames=16)
loader = ds.get_dataloader(batch_size=1, shuffle=True, num_workers=0)
batch = next(iter(loader))
one_frame = batch['video'][0, 0:1]  # (1, 3, 128, 128)
video = one_frame.repeat(1, 2, 1, 1, 1).cuda()  # (1, 2, 3, 128, 128) — burnin=1, rollout=1
data_loader = [(video,)]  # single batch, reused

log(f'Input shape: {video.shape}, range=[{video.min():.4f}, {video.max():.4f}]')

# ── Model ───────────────────────────────────────────────────────────────────
torch.manual_seed(42)
    model = SlotDynamicsModel(cfg)
model.cuda()
log(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

optimizer, scheduler = create_optimizer(model.parameters(), cfg)
trainer = Trainer(model, optimizer, scheduler, cfg, wandb_logger=wandb_logger)

# ── Training loop ────────────────────────────────────────────────────────────
log(f'Starting training: {cfg.num_steps} steps')
t0 = time.time()
best_loss = float('inf')

for step in range(cfg.num_steps):
    gamma = trainer._set_gamma(step)  # set GRL gamma before forward
    model.train()
    optimizer.zero_grad()
    out = model(video)
    # Downsample target to dec_size for loss
    dec_size = out['outputs']['video_burnin'].shape[-1]
    target_b = F.interpolate(video[:, :1].reshape(-1, 3, 128, 128), size=dec_size, mode='bilinear'
    ).reshape(1, 1, 3, dec_size, dec_size)
    target_r = F.interpolate(video[:, 1:2].reshape(-1, 3, 128, 128), size=dec_size, mode='bilinear'
    ).reshape(1, 1, 3, dec_size, dec_size)

    recon_burnin = cfg.lambda_recon_burnin * F.mse_loss(out['outputs']['video_burnin'], target_b)
    recon_rollout = cfg.lambda_recon_rollout * F.mse_loss(out['outputs']['video_pred'], target_r)
    recon_loss = recon_burnin + recon_rollout

    all_slots = torch.cat([out['slots']['corrected'], out['slots']['predicted']], dim=1)
    slot_pred_loss, aux = trainer.loss_fn(
        out['slots']['predicted'], out['slots']['target'],
        slots_full_seq=all_slots,
        rev_pred=out.get('rev_pred'),
        S_c=out.get('S_c'),
        energy=out.get('energy_pairs'))
    total_loss = recon_loss + slot_pred_loss
    aux['recon_burnin'] = recon_burnin.item()
    aux['recon_rollout'] = recon_rollout.item()

    total_loss.backward()
    if hasattr(model, 'mlp_rev') and trainer.rev_grad_max_norm > 0:
        nn.utils.clip_grad_norm_(model.mlp_rev.parameters(), trainer.rev_grad_max_norm)
    if cfg.max_grad_norm > 0:
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    else:
        grad_norm = 0.0
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    # ── Logging ─────────────────────────────────────────────────────────────
    if step % cfg.log_every == 0:
        lr = optimizer.param_groups[0]['lr']
        msg = (f'step {step:5d} | loss={total_loss.item():.4f} '
               f'recon_b={aux["recon_burnin"]:.4f} recon_r={aux["recon_rollout"]:.4f} '
               f'slot={aux["slot_loss"]:.4f} static={aux["static_loss"]:.6f} '
               f'rev={aux["rev_loss"]:.6f} gn={grad_norm:.1f} lr={lr:.2e} gamma={gamma:.3f}')
        log(msg)
        trainer.writer.add_scalar('loss/total', total_loss.item(), step)
        trainer.writer.add_scalar('loss/recon_burnin', aux['recon_burnin'], step)
        trainer.writer.add_scalar('loss/recon_rollout', aux['recon_rollout'], step)
        trainer.writer.add_scalar('loss/slot', aux['slot_loss'], step)
        trainer.writer.add_scalar('loss/static', aux['static_loss'], step)
        trainer.writer.add_scalar('loss/rev', aux['rev_loss'], step)
        trainer.writer.add_scalar('lr', lr, step)
        trainer.writer.add_scalar('rev_weight', gamma, step)
        trainer.writer.add_scalar('grad_norm', grad_norm, step)
        wandb_logger.log({
            'loss/total': total_loss.item(),
            'loss/recon_burnin': aux['recon_burnin'],
            'loss/recon_rollout': aux['recon_rollout'],
            'loss/slot': aux['slot_loss'],
            'loss/static': aux['static_loss'],
            'loss/rev': aux['rev_loss'],
            'train/lr': lr,
            'train/rev_weight': gamma,
            'train/grad_norm': grad_norm,
        }, step=step)

    # ── Save & Eval ─────────────────────────────────────────────────────────
    if (step + 1) % cfg.save_every == 0 or step == cfg.num_steps - 1:
        trainer.save_checkpoint(
            os.path.join(trainer.ckpt_dir, f'step_{step+1}.pt'), step + 1, total_loss.item())
        trainer._cleanup_old_checkpoints()
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            trainer.save_checkpoint(
                os.path.join(trainer.ckpt_dir, 'best.pt'), step + 1, total_loss.item())

        # ── Visualize ───────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            out_viz = model(video)
        from PIL import Image, ImageDraw
        viz_dir = os.path.join(WORKDIR, 'eval_images', f'step_{step+1}')
        os.makedirs(viz_dir, exist_ok=True)
        S = 128
        canvas = Image.new('RGB', (4 * S, 2 * S), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        def put(t, row, col):
            arr = t.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            im = Image.fromarray((arr * 255).astype('uint8'))
            im = im.resize((S, S), Image.BILINEAR)
            canvas.paste(im, (col * S, row * S))
        # Row 0: GT   Col 0=burnin, Col 1=rollout
        put(video[0, 0], 0, 0)   # GT burnin
        put(video[0, 1], 0, 1)   # GT rollout
        # Row 1: Recon
        dec_b = out_viz['outputs']['video_burnin']
        dec_p = out_viz['outputs']['video_pred']
        ds_b = dec_b.shape[-1]
        dec_b_up = F.interpolate(dec_b.reshape(-1, 3, ds_b, ds_b), size=128, mode='bilinear').reshape(1, 1, 3, 128, 128)
        dec_p_up = F.interpolate(dec_p.reshape(-1, 3, ds_b, ds_b), size=128, mode='bilinear').reshape(1, 1, 3, 128, 128)
        put(dec_b_up[0, 0], 1, 0)  # Recon burnin
        put(dec_p_up[0, 0], 1, 1)  # Pred rollout
        draw.text((2, 2), 'GT', fill=(0, 0, 0))
        draw.text((2, S + 2), 'Recon', fill=(0, 0, 0))
        path = os.path.join(viz_dir, 'recon.png')
        canvas.save(path)
        log(f'  Eval step {step+1}: saved viz to {path}')
        if use_wandb:
            wandb_logger.log({'eval/recon': wandb.Image(str(path))}, step=step + 1)

        # Delete old eval steps
        for d in glob.glob(os.path.join(WORKDIR, 'eval_images', 'step_*')):
            if d != viz_dir:
                shutil.rmtree(d, ignore_errors=True)

elapsed = time.time() - t0
log(f'Finished {cfg.num_steps} steps in {elapsed:.0f}s ({elapsed/cfg.num_steps*1000:.1f}ms/step)')
wandb_logger.finish()
