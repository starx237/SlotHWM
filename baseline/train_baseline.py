#!/usr/bin/env python3
# Baseline 训练入口（原论文对比实验）
# Usage: python train_baseline.py --config config/baseline_clevrer.yaml --workdir experiments/baseline_clevrer

import os, sys, argparse, yaml, math, glob
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from PIL import Image, ImageDraw
    _PIL_AVAIL = True
except ImportError:
    _PIL_AVAIL = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.baseline_model import BaselineModel
from train.optimizer import create_optimizer
from train.trainer import Trainer, WandBLogger
from data import get_dataset


def setup_cuda(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    return torch.Generator().manual_seed(seed)


def load_wandb_config():
    load_dotenv()
    enabled = os.getenv('WANDB_ENABLED', 'false').lower() == 'true'
    return {
        'enabled': enabled,
        'api_key': os.getenv('WANDB_API_KEY', ''),
        'project': os.getenv('WANDB_PROJECT', 'slotpi'),
        'name': os.getenv('WANDB_NAME', ''),
        'entity': os.getenv('WANDB_ENTITY', ''),
        'notes': os.getenv('WANDB_NOTES', ''),
    }


def compute_loss(model_out, frames, cfg):
    burnin = cfg.burnin_frames
    rollout = cfg.rollout_frames
    device = frames.device

    video_burnin = model_out["outputs"]["video_burnin"]
    video_pred = model_out["outputs"]["video_pred"]

    target_burnin = frames[:, :burnin]
    if rollout > 0 and video_pred is not None:
        target_rollout = frames[:, burnin:burnin + rollout]

    recon_burnin = getattr(cfg, 'lambda_recon_burnin', 1.0) * F.mse_loss(video_burnin, target_burnin)

    if rollout == 0 or video_pred is None:
        return recon_burnin, {
            "recon_burnin": recon_burnin.item(),
            "recon_rollout": 0.0,
            "slot_loss": 0.0,
        }

    recon_rollout = getattr(cfg, 'lambda_recon_rollout', 1.0) * F.mse_loss(video_pred, target_rollout)
    recon_loss = recon_burnin + recon_rollout

    pred_slots = model_out["slots"]["predicted"]
    target_slots = model_out["slots"]["target"]
    slot_loss = F.mse_loss(pred_slots, target_slots)
    lambda_slots = getattr(cfg, 'lambda_slots', 1.0)

    total_loss = recon_loss + lambda_slots * slot_loss

    aux = {
        "recon_burnin": recon_burnin.item(),
        "recon_rollout": recon_rollout.item(),
        "slot_loss": (lambda_slots * slot_loss).item(),
    }
    return total_loss, aux


def save_checkpoint(model, optimizer, scheduler, step, loss, path):
    raw = model.state_dict()
    clean = {k.replace('_orig_mod.', ''): v for k, v in raw.items()}
    torch.save({
        "step": step,
        "model": clean,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "loss": loss,
    }, path)


def cleanup_old_checkpoints(ckpt_dir, keep_last):
    if keep_last <= 0:
        return
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    while len(ckpts) > keep_last:
        os.remove(ckpts[0])
        ckpts = ckpts[1:]


@torch.no_grad()
def save_viz(samples, step, workdir, wandb_logger, cfg):
    burnin = cfg.burnin_frames
    rollout = cfg.rollout_frames
    if rollout == 0:
        return
    viz_dir = os.path.join(workdir, 'eval_images')
    for old in glob.glob(os.path.join(viz_dir, 'step_*')):
        import shutil
        shutil.rmtree(old, ignore_errors=True)
    step_dir = os.path.join(viz_dir, f'step_{step}')
    os.makedirs(step_dir, exist_ok=True)

    target_size = None

    def upscale(t):
        ds = t.shape[-1]
        flat = t.reshape(-1, 3, ds, ds)
        up = F.interpolate(flat, size=target_size, mode='bilinear')
        return up.reshape(t.shape[0], t.shape[1], 3, target_size, target_size)

    for idx, (out, frames) in enumerate(samples):
        B, T, C, H, W = frames.shape
        if target_size is None:
            target_size = W

        dec_b = upscale(out["outputs"]["video_burnin"][:1])
        gt_b = frames[:, :burnin]

        n_cols = max(burnin, rollout) if rollout > 0 else burnin
        n_rows = 4 if rollout > 0 and out["outputs"]["video_pred"] is not None else 2
        S = min(80, 2000 // max(n_cols, 1))
        canvas = Image.new('RGB', (n_cols * S, n_rows * S), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        def put(t, row, col):
            arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            im = Image.fromarray((arr * 255).astype('uint8'))
            im = im.resize((S, S), Image.BILINEAR)
            canvas.paste(im, (col * S, row * S))

        for t in range(burnin):
            put(gt_b[0, t], 0, t)
            put(dec_b[0, t], 1, t)

        if rollout > 0 and out["outputs"]["video_pred"] is not None:
            dec_p = upscale(out["outputs"]["video_pred"][:1])
            gt_r = frames[:, burnin:burnin + rollout]
            for t in range(rollout):
                put(gt_r[0, t], 2, t)
                put(dec_p[0, t], 3, t)
            labels = ['GT Burnin', 'Recon Burnin', 'GT Rollout', 'Pred Rollout']
        else:
            labels = ['GT Burnin', 'Recon Burnin']

        for i, label in enumerate(labels):
            draw.text((2, i * S + 2), label, fill=(0, 0, 0))

        path = os.path.join(step_dir, f'sample_{idx}.png')
        canvas.save(path)
        if wandb_logger.enabled:
            wandb_logger.log({f"eval/recon_{idx}": wandb_logger.wandb.Image(path)}, step=step)


@torch.no_grad()
def evaluate(model, loader, cfg, device, step=None, workdir=None, wandb_logger=None):
    model.eval()
    total_loss = 0
    viz_samples = []
    for batch in loader:
        frames = batch["video"].to(device)
        out = model(frames)
        loss, _ = compute_loss(out, frames, cfg)
        total_loss += loss.item()

        if _PIL_AVAIL and step is not None and len(viz_samples) < 5 and workdir:
            viz_samples.append((out, frames[0:1]))

    if _PIL_AVAIL and step is not None and viz_samples and workdir:
        save_viz(viz_samples, step, workdir, wandb_logger, cfg)

    return total_loss / max(1, len(loader))


def main():
    log_fp = open('log_baseline.txt', 'a', buffering=1)
    os.dup2(log_fp.fileno(), sys.stdout.fileno())
    os.dup2(log_fp.fileno(), sys.stderr.fileno())

    seed_gen = setup_cuda()
    parser = argparse.ArgumentParser(description='Baseline Training')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--workdir', type=str, default='./experiments/baseline_default')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict['workdir'] = args.workdir
    cfg = SimpleNamespace(**cfg_dict)

    os.makedirs(os.path.join(args.workdir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.workdir, 'tb_logs'), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.workdir, 'tb_logs'))

    wb_cfg = load_wandb_config()
    use_wandb = wb_cfg['enabled'] and not args.no_wandb
    wandb_logger = WandBLogger(enabled=use_wandb)
    if use_wandb:
        if wb_cfg['api_key']:
            os.environ['WANDB_API_KEY'] = wb_cfg['api_key']
        run_name = (wb_cfg['name'] or os.path.basename(args.workdir)) + " (baseline)"
        import wandb
        wandb_logger.init(
            project=wb_cfg['project'],
            entity=wb_cfg['entity'] or None,
            name=run_name,
            config=cfg_dict,
            notes=wb_cfg['notes'] or None,
            dir=args.workdir,
            settings=wandb.Settings(sync_tensorboard=False),
        )
        print(f"WandB enabled: project={wb_cfg['project']}, name={run_name}")
    else:
        print("WandB disabled")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineModel(cfg).to(device)

    # 冻结 STATM-SAVi（后训练：仅训练 predictor）
    if getattr(cfg, 'freeze_slot', False):
        for name in ['encoder', 'slot_attention', 'decoder']:
            mod = getattr(model, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(False)
        print("Frozen: encoder + slot_attention + decoder (STATM-SAVi)")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,} / {sum(p.numel() for p in model.parameters()):,}")

    optimizer, scheduler = create_optimizer(
        (p for p in model.parameters() if p.requires_grad), cfg)

    # Load pretrained weights for encoder/decoder/slot_attention
    pretrained_path = getattr(cfg, 'pretrained_path', None)
    if pretrained_path and os.path.isfile(pretrained_path) and not getattr(cfg, 'pretrain', False):
        ckpt = torch.load(pretrained_path, map_location=device)
        ckpt_state = ckpt["model"] if "model" in ckpt else ckpt
        model_state = model.state_dict()
        loaded = Trainer._match_and_load(model_state, ckpt_state)
        model.load_state_dict(loaded, strict=False)
        missing = set(model_state.keys()) - set(loaded.keys())
        print(f"Loaded pretrained: {len(loaded)} keys matched, {len(missing)} not in checkpoint")
        del ckpt, ckpt_state, model_state, loaded

    start_step = 0
    resume_path = args.resume
    if not resume_path and getattr(cfg, 'resume', False) and getattr(cfg, 'resume_step', 0) > 0:
        resume_path = os.path.join(args.workdir, 'checkpoints', f'step_{cfg.resume_step}.pt')
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        ckpt_state = ckpt["model"]
        model_state = model.state_dict()
        filtered = Trainer._match_and_load(model_state, ckpt_state)
        skipped = len(ckpt_state) - len(filtered)
        model.load_state_dict(filtered, strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, RuntimeError) as e:
            print(f"Optimizer state incompatible, reinitializing: {e}")
        if scheduler and ckpt.get("scheduler"):
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except (ValueError, RuntimeError) as e:
                print(f"Scheduler state incompatible, reinitializing: {e}")
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}: {len(filtered)}/{len(ckpt_state)} keys loaded, {skipped} skipped")

    num_frames = getattr(cfg, 'num_frames', None) or (cfg.burnin_frames + cfg.rollout_frames)
    slide_stride = getattr(cfg, 'slide_stride', 1)
    ds = get_dataset(cfg.dataset, data_path=cfg.data_root,
                     num_frames=num_frames, stride=slide_stride)
    loader = ds.get_dataloader(batch_size=cfg.batch_size, shuffle=True,
                               num_workers=getattr(cfg, 'num_workers', 4),
                               generator=seed_gen)

    # 验证集使用相同的 dataset（主项目同样做法）
    val_loader = ds.get_dataloader(batch_size=cfg.batch_size, shuffle=False,
                                   num_workers=getattr(cfg, 'num_workers', 2),
                                   generator=seed_gen)

    best_loss = float('inf')
    log_every = getattr(cfg, 'log_every', 10)
    save_every = getattr(cfg, 'save_every', 1000)
    keep_last = getattr(cfg, 'keep_last_checkpoints', 0)
    max_grad_norm = getattr(cfg, 'max_grad_norm', 1.0)
    eval_every_epochs = getattr(cfg, 'eval_every_epochs', 0)
    ckpt_dir = os.path.join(args.workdir, 'checkpoints')

    global_step = start_step
    epoch = 0
    pbar = tqdm(total=cfg.num_steps, initial=start_step, desc="Baseline")

    while global_step < cfg.num_steps:
        for batch in loader:
            if global_step >= cfg.num_steps:
                break

            frames = batch["video"].to(device)
            model.train()
            optimizer.zero_grad()

            out = model(frames)
            loss, aux = compute_loss(out, frames, cfg)
            loss.backward()

            if max_grad_norm > 0:
                gn = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            else:
                gn = 0.0

            # 子模块梯度监控
            grad_norms = {}
            for name in ['encoder', 'slot_attention', 'decoder', 'predictor']:
                mod = getattr(model, name, None)
                if mod is None:
                    continue
                g = sum(p.grad.norm().item()**2 for p in mod.parameters() if p.grad is not None)
                grad_norms[f'grad/{name}'] = g**0.5 if g > 0 else 0.0

            optimizer.step()
            if scheduler:
                scheduler.step()

            global_step += 1
            pbar.update(1)

            if global_step % log_every == 0:
                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "recon_b": f"{aux['recon_burnin']:.4f}",
                    "recon_r": f"{aux['recon_rollout']:.4f}",
                    "slot": f"{aux['slot_loss']:.4f}",
                    "lr": f"{lr:.2e}",
                })
                writer.add_scalar("loss/total", loss.item()*10, global_step)
                writer.add_scalar("loss/recon_burnin", aux['recon_burnin']*10, global_step)
                writer.add_scalar("loss/recon_rollout", aux['recon_rollout']*10, global_step)
                writer.add_scalar("loss/slot", aux['slot_loss']*10, global_step)
                writer.add_scalar("lr", lr, global_step)
                writer.add_scalar("grad_norm", gn, global_step)
                for k, v in grad_norms.items():
                    writer.add_scalar(k, v, global_step)
                wandb_logger.log({
                    "loss/total": loss.item()*10,
                    "loss/recon_burnin": aux['recon_burnin']*10,
                    "loss/recon_rollout": aux['recon_rollout']*10,
                    "loss/slot": aux['slot_loss']*10,
                    "train/lr": lr,
                    "train/grad_norm": gn,
                    **grad_norms,
                }, step=global_step)

            if global_step % save_every == 0:
                save_checkpoint(model, optimizer, scheduler, global_step,
                                loss.item(), os.path.join(ckpt_dir, f"step_{global_step}.pt"))
                cleanup_old_checkpoints(ckpt_dir, keep_last)

            if loss.item() < best_loss:
                best_loss = loss.item()
                save_checkpoint(model, optimizer, scheduler, global_step,
                                loss.item(), os.path.join(ckpt_dir, "best.pt"))

        epoch += 1
        if eval_every_epochs > 0 and epoch % eval_every_epochs == 0:
            val_loss = evaluate(model, val_loader, cfg, device,
                                step=global_step, workdir=args.workdir, wandb_logger=wandb_logger)
            writer.add_scalar("loss/eval", val_loss*10, global_step)
            wandb_logger.log({"loss/eval": val_loss*10}, step=global_step)
            print(f"Eval step {global_step}: val_loss={val_loss:.6f}")

    pbar.close()
    writer.close()
    wandb_logger.finish()


if __name__ == '__main__':
    main()
