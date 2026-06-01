import os
from dataclasses import dataclass
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
    num_layers: int = 1
    h_layers: int = 1
    out_mlp: bool = False
    out_hidden_layers: int = 1
    pre_norm: bool = False
    dropout_rate: float = 0.1

    # Physics module
    delta_t: float = 0.125
    integrator_method: str = "Leapfrog"
    lambda_phys: float = 1.0

    # Training
    burnin_frames: int = 6
    rollout_frames: int = 10
    buffer_len: int = 10
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_epochs: int = 50
    warmup_steps: int = 1000

    # Data
    dataset: str = "clevrer"
    data_root: str = "./data"
    train_split: str = "train"
    val_split: str = "val"
    num_workers: int = 4
    pin_memory: bool = True

    # Logging & Saving
    workdir: str = "./output"
    log_every: int = 10
    save_every: int = 10
    eval_every: int = 1

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 静态特征方差正则化系数（idea.md §3）
    lambda_static: float = 0.001

    def __post_init__(self):
        os.makedirs(os.path.join(self.workdir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.workdir, "tb_logs"), exist_ok=True)


@dataclass
class Stage1Config(SlotPiConfig):
    """Configuration for stage 1 training (STATM-SAVi encoder + decoder)."""
    # Encoder
    encoder_backbone: str = "cnn"
    encoder_features: List[int] = None

    # Corrector
    corrector_iterations: int = 1

    # Decoder
    broadcast_size: tuple = (8, 8)
    decoder_features: List[int] = None

    # Video targets
    targets: dict = None

    def __post_init__(self):
        super().__post_init__()
        if self.encoder_features is None:
            self.encoder_features = [32, 32, 32, 32]
        if self.decoder_features is None:
            self.decoder_features = [64, 64, 64, 64]
        if self.targets is None:
            self.targets = {"video": 3}


@dataclass
class Stage2Config(SlotPiConfig):
    """Configuration for stage 2 training (SlotPi physics + spatiotemporal)."""
    load_pretrained_encoder: str = ""
    freeze_encoder: bool = True
    use_pretrained_slots: bool = False
    pretrained_slot_path: str = ""


@dataclass
class CLEVRERStage1Config(Stage1Config):
    dataset: str = "clevrer"
    num_slots: int = 7
    slot_dim: int = 128
    burnin_frames: int = 6
    batch_size: int = 64
    learning_rate: float = 2e-4
    num_epochs: int = 21


@dataclass
class CLEVRERStage2Config(Stage2Config):
    dataset: str = "clevrer"
    num_slots: int = 7
    slot_dim: int = 128
    burnin_frames: int = 6
    rollout_frames: int = 10
    batch_size: int = 64
    learning_rate: float = 2e-4
    num_epochs: int = 50


@dataclass
class OBJ3DStage1Config(Stage1Config):
    dataset: str = "obj3d"
    num_slots: int = 7
    slot_dim: int = 128
    burnin_frames: int = 6
    batch_size: int = 64
    learning_rate: float = 2e-4
    num_epochs: int = 21


@dataclass
class OBJ3DStage2Config(Stage2Config):
    dataset: str = "obj3d"
    num_slots: int = 7
    slot_dim: int = 128
    burnin_frames: int = 6
    rollout_frames: int = 10
    batch_size: int = 64
    learning_rate: float = 2e-4
    num_epochs: int = 50


@dataclass
class PhysionStage2Config(Stage2Config):
    dataset: str = "physion"
    num_slots: int = 8
    slot_dim: int = 128
    burnin_frames: int = 6
    rollout_frames: int = 10
    batch_size: int = 64
    learning_rate: float = 2e-4
    num_epochs: int = 50


@dataclass
class FluidStage2Config(Stage2Config):
    dataset: str = "fluid"
    num_slots: int = 1024
    slot_dim: int = 512
    num_heads: int = 8
    qkv_size: int = 512
    mlp_size: int = 512
    num_layers: int = 2
    burnin_frames: int = 6
    rollout_frames: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50


@dataclass
class RealWorldStage2Config(Stage2Config):
    dataset: str = "realworld"
    num_slots: int = 6
    slot_dim: int = 192
    num_heads: int = 4
    burnin_frames: int = 10
    rollout_frames: int = 15
    batch_size: int = 64
    learning_rate: float = 2e-4
    num_epochs: int = 40