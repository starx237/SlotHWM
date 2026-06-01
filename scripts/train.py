import torch
import numpy as np
import random
import argparse
import json
import os
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.config import SlotPiConfig, CLEVRERConfig, OBJ3DConfig, PhysionConfig, FluidConfig, RealWorldConfig
from models.slotpi_model import SlotPiModel
from models.encoder import FrameEncoder, CNNEncoder, ResNetEncoder
from models.decoder import SpatialBroadcastDecoder
from models.statm_savi import STATMSAVi
from training.trainer import SlotPiTrainer, SatelliteLoss, SlotPiLoss
from evaluation.evaluator import Evaluator
from data.datasets import SlotPiDataset


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_config(name):
    config_map = {
        "clevrer": CLEVRERConfig,
        "obj3d": OBJ3DConfig,
        "physion": PhysionConfig,
        "fluid": FluidConfig,
        "realworld": RealWorldConfig,
    }
    return config_map[name]()


def build_model(config):
    encoder = None
    if config.dataset in ["clevrer", "obj3d", "physion", "realworld"]:
        if config.dataset == "realworld":
            encoder = ResNetEncoder("resnet18", config.slot_dim)
        else:
            encoder = CNNEncoder(config.slot_dim)
    elif config.dataset == "fluid":
        encoder = ResNetEncoder("resnet18", config.slot_dim)

    decoder = SpatialBroadcastDecoder(
        slot_dim=config.slot_dim,
        out_channels=3,
        broadcast_size=8 if config.dataset != "realworld" else (12, 7),
    )

    model = SlotPiModel(
        num_slots=config.num_slots,
        embed_dim=config.slot_dim,
        num_heads=config.num_heads,
        qkv_size=config.qkv_size,
        mlp_size=config.mlp_size,
        num_layers=config.num_layers,
        h_layers=config.h_layers,
        out_mlp=config.out_mlp,
        out_hidden_layers=config.out_hidden_layers,
        pre_norm=config.pre_norm,
        dropout_rate=config.dropout_rate,
        delta_t=config.delta_t,
        integrator_method=config.integrator_method,
        lambda_phys=config.lambda_phys,
        encoder=encoder,
        decoder=decoder,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="clevrer", choices=["clevrer", "obj3d", "physion", "fluid", "realworld"])
    parser.add_argument("--workdir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "extract"])
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    config = build_config(args.config)
    config.workdir = args.workdir

    device = torch.device(config.device)

    model = build_model(config).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    if args.mode == "train":
        train_dataset = SlotPiDataset(config, split=config.train_split)
        val_dataset = SlotPiDataset(config, split=config.val_split)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, pin_memory=config.pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                num_workers=config.num_workers, pin_memory=config.pin_memory)

        if config.optimizer == "adamw":
            optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma) \
            if hasattr(config, 'scheduler_step_size') else None

        trainer = SlotPiTrainer(model, optimizer, scheduler, config, device)
        trainer.train(train_loader, val_loader, config.num_epochs)

    elif args.mode == "eval":
        val_dataset = SlotPiDataset(config, split=config.val_split)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                num_workers=config.num_workers, pin_memory=config.pin_memory)
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        evaluator = Evaluator(config)
        results = evaluator.evaluate(model, val_loader)
        print(f"Evaluation results: {results}")

    elif args.mode == "extract":
        # Extract slots using STATM-SAVi encoder
        from models.statm_savi import STATMSAVi
        statm_model = STATMSAVi(slot_dim=config.slot_dim, num_slots=config.num_slots).to(device)
        statm_model.load_state_dict(torch.load(args.resume, map_location=device)["model_state_dict"])
        dataset = SlotPiDataset(config, split=config.train_split)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)
        all_slots = []
        for batch in tqdm(loader, desc="Extracting slots"):
            x = batch["video"].to(device)
            with torch.no_grad():
                slots = statm_model.encode_video(x)
            all_slots.append(slots.cpu())
        all_slots = torch.cat(all_slots, dim=0)
        torch.save(all_slots, os.path.join(config.workdir, "extracted_slots.pt"))
        print(f"Saved extracted slots with shape {all_slots.shape}")


if __name__ == "__main__":
    main()