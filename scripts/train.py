#!/usr/bin/env python3
# SlotPi 端到端训练入口
# Usage: python scripts/train.py --config config/obj3d.yaml --workdir experiments/obj3d

import os, sys, argparse, yaml, subprocess
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import torch
from types import SimpleNamespace
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dynamics import SlotDynamicsModel
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
    log_fp = open('log.txt', 'a', buffering=1)
    os.dup2(log_fp.fileno(), sys.stdout.fileno())
    os.dup2(log_fp.fileno(), sys.stderr.fileno())

    seed_gen = setup_cuda()
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

    model = SlotDynamicsModel(cfg)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,} / {sum(p.numel() for p in model.parameters()):,}")

    optimizer, scheduler = create_optimizer(
        (p for p in model.parameters() if p.requires_grad), cfg)
    trainer = Trainer(model, optimizer, scheduler, cfg, wandb_logger=wandb_logger)

    # Visualization callback (pretrain only: slot decomposition + swap test)
    viz_every = getattr(cfg, 'viz_every_steps', 0)
    if viz_every > 0 and getattr(cfg, 'pretrain', False):
        vis_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vis_slots.py')
        swap_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vis_swap.py')
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.abspath(args.config)
        workdir = os.path.abspath(args.workdir)

        def vis_callback(step):
            if step > 0 and step % viz_every == 0:
                subprocess.run([sys.executable, vis_script, '0-4', str(step),
                                config_path, workdir], cwd=base_dir)
                for i in range(5):
                    subprocess.run(
                        [sys.executable, swap_script, str(i), str(step), 'cswap',
                         config_path, workdir, 'auto', 'auto'], cwd=base_dir)

        trainer.viz_callback = vis_callback

    num_frames = getattr(cfg, 'num_frames', None) or (getattr(cfg, 'burnin_frames', 6) + getattr(cfg, 'rollout_frames', 10))
    slide_stride = getattr(cfg, 'slide_stride', 1)
    subsample = getattr(cfg, 'subsample', 1)
    ds = get_dataset(cfg.dataset, data_path=cfg.data_root,
                     num_frames=num_frames, stride=slide_stride,
                     subsample=subsample)
    loader = ds.get_dataloader(batch_size=cfg.batch_size, shuffle=True,
                               num_workers=getattr(cfg, 'num_workers', 4),
                               generator=seed_gen)

    start_step = 0

    # 加载预训练权重（仅 STATM-SAVi 部分：encoder, slot_attention, decoder）
    pretrained_path = getattr(cfg, 'pretrained_path', None)
    if pretrained_path and not getattr(cfg, 'pretrain', False):
        if os.path.isfile(pretrained_path):
            loaded, skipped = trainer.load_pretrained(pretrained_path)
            print(f"Loaded pretrained: {len(loaded)} keys matched, {len(skipped)} skipped")

    # Resume checkpoint（完整状态恢复）
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
