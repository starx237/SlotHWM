#!/usr/bin/env python3
# SlotPi 端到端训练入口
# Usage: python scripts/train.py --config config/obj3d.yaml --workdir experiments/obj3d

import os, sys, argparse, yaml
import torch
from types import SimpleNamespace
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.slotpi import SlotPi
from train import Trainer, create_optimizer
from train.trainer import WandBLogger
from data import get_dataset


def setup_cuda():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')


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
    setup_cuda()
    parser = argparse.ArgumentParser(description='SlotPi Training')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--workdir', type=str, default='./experiments/default')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-wandb', action='store_true', help='Force disable wandb')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict['workdir'] = args.workdir
    cfg = SimpleNamespace(**cfg_dict)

    os.makedirs(os.path.join(args.workdir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.workdir, 'tb_logs'), exist_ok=True)

    # WandB
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

    model = SlotPi(cfg)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer, scheduler = create_optimizer(model.parameters(), cfg)
    trainer = Trainer(model, optimizer, scheduler, cfg, wandb_logger=wandb_logger)

    num_frames = getattr(cfg, 'num_frames', None) or (getattr(cfg, 'burnin_frames', 6) + getattr(cfg, 'rollout_frames', 10))
    slide_stride = getattr(cfg, 'slide_stride', 1)
    ds = get_dataset(cfg.dataset, data_path=cfg.data_root,
                     num_frames=num_frames, stride=slide_stride)
    loader = ds.get_dataloader(batch_size=cfg.batch_size, shuffle=True,
                               num_workers=getattr(cfg, 'num_workers', 4))

    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        start_step, _ = trainer.load_checkpoint(args.resume)
        print(f"Resumed from step {start_step}")

    trainer.train(loader, loader, cfg.num_steps, start_step=start_step)


if __name__ == '__main__':
    main()
