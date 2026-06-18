#!/usr/bin/env python3
import os, sys, argparse, yaml, subprocess
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import torch
from types import SimpleNamespace
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isa_pretrain.models.isa_dynamics import SlotDynamicsModelISA
from train import Trainer, create_optimizer
from train.trainer import WandBLogger
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


def main():
    log_fp = open('log_isa.txt', 'a', buffering=1)
    os.dup2(log_fp.fileno(), sys.stdout.fileno())
    os.dup2(log_fp.fileno(), sys.stderr.fileno())

    seed_gen = setup_cuda()
    parser = argparse.ArgumentParser(description='ISA Pretrain')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--workdir', type=str, default='./experiments/isa_obj3d')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--vis-every', type=int, default=2000,
                        help='Run slot visualization every N steps (0=disable)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict['workdir'] = args.workdir
    cfg = SimpleNamespace(**cfg_dict)

    os.makedirs(os.path.join(args.workdir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.workdir, 'tb_logs'), exist_ok=True)

    wb_cfg = load_wandb_config()
    use_wandb = wb_cfg['enabled'] and not args.no_wandb
    wandb_logger = WandBLogger(enabled=use_wandb)
    if use_wandb:
        if wb_cfg['api_key']:
            os.environ['WANDB_API_KEY'] = wb_cfg['api_key']
        run_name = wb_cfg['name'] or os.path.basename(args.workdir)
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

    model = SlotDynamicsModelISA(cfg)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,} / {sum(p.numel() for p in model.parameters()):,}")

    print(f"  encoder: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  slot_attention: {sum(p.numel() for p in model.slot_attention.parameters()):,}")
    print(f"  decoder: {sum(p.numel() for p in model.decoder.parameters()):,}")

    optimizer, scheduler = create_optimizer(
        (p for p in model.parameters() if p.requires_grad), cfg)
    trainer = Trainer(model, optimizer, scheduler, cfg, wandb_logger=wandb_logger)

    if args.vis_every > 0:
        vis_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'scripts', 'visualize_slots.py')
        swap_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'scripts', 'swap_test.py')
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        def vis_callback(step):
            if step > 0 and step % args.vis_every == 0:
                subprocess.run([sys.executable, vis_script, '0-4', str(step)],
                               cwd=base_dir)
                for i in range(5):
                    subprocess.run(
                        [sys.executable, swap_script, str(i), str(step), 'cswap', 'auto'],
                        cwd=base_dir)

        trainer.post_save_callback = vis_callback

    num_frames = cfg.burnin_frames
    slide_stride = getattr(cfg, 'slide_stride', 1)
    subsample = getattr(cfg, 'subsample', 1)
    ds = get_dataset(cfg.dataset, data_path=cfg.data_root,
                     num_frames=num_frames, stride=slide_stride,
                     subsample=subsample)
    loader = ds.get_dataloader(batch_size=cfg.batch_size, shuffle=True,
                                num_workers=getattr(cfg, 'num_workers', 4),
                                generator=seed_gen)

    start_step = 0
    resume_path = args.resume
    if not resume_path:
        if getattr(cfg, 'resume', False) and getattr(cfg, 'resume_step', 0) > 0:
            resume_path = os.path.join(args.workdir, 'checkpoints', f'step_{cfg.resume_step}.pt')
    if resume_path and os.path.isfile(resume_path):
        start_step, _ = trainer.load_checkpoint(resume_path)
        print(f"Resumed from step {start_step}")

    trainer.train(loader, loader, cfg.num_steps, start_step=start_step)


if __name__ == '__main__':
    main()
