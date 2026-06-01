import os
from dataclasses import dataclass, fields
from typing import Optional, List
import torch


@dataclass
class SlotPiConfig:
    # Model architecture
    num_slots: int = 7
    slot_dim: int = 128
    hidden_dim: int = 256
    num_heads: int = 4
    qkv_size: int = 128
    mlp_size: int = 256
    num_layers: int = 2
    h_layers: int = 1
    out_mlp: bool = False
    out_hidden_layers: int = 1
    pre_norm: bool = False
    dropout_rate: float = 0.1

    # Encoder
    encoder_type: str = "cnn"
    in_channels: int = 3
    img_size: int = 64
    encoder_hidden: int = 32
    slot_hidden: int = 128
    slot_iters: int = 3

    # Decoder
    decoder_hidden: int = 64
    broadcast_size: int = 8

    # Physics module
    delta_t: float = 0.125
    integrator_method: str = "Leapfrog"
    lambda_phys: float = 1.0

    # Spatiotemporal
    num_spatiotemporal_blocks: int = 2

    # Training
    burnin_frames: int = 6
    rollout_frames: int = 10
    buffer_len: int = 10
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_steps: int = 160000
    warmup_steps: int = 2500

    # Loss weights
    lambda_slots: float = 1.0
    lambda_images: float = 0.0
    lambda_energy: float = 0.0
    lambda_static: float = 0.001

    # Data
    dataset: str = "obj3d"
    data_root: str = "./data"
    num_workers: int = 4

    # Logging & Saving
    workdir: str = "./experiments/default"
    log_every: int = 10
    save_every: int = 1000

    def __post_init__(self):
        os.makedirs(os.path.join(self.workdir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.workdir, "tb_logs"), exist_ok=True)
