import torch
import torch.nn as nn
from einops import rearrange
from .hamiltonian import Slot_HamiltonianNet


class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.slot_dim = config.slot_dim
        self.num_slots = config.num_slots

        self.hamiltonian = Slot_HamiltonianNet(
            num_slots=self.num_slots,
            embed_dim=self.slot_dim,
            num_heads=getattr(config, 'num_heads', 4),
            qkv_size=getattr(config, 'qkv_size', 128),
            mlp_size=getattr(config, 'mlp_size', 256),
            pre_norm=getattr(config, 'pre_norm', False),
            dropout_rate=getattr(config, 'dropout_rate', 0.1),
            num_layers=getattr(config, 'num_layers', 2),
            h_layers=getattr(config, 'h_layers', 1),
            out_mlp=False,
            out_hidden_layers=1,
        )

        from models.attention import TimeSpaceTransformerBlock2
        nst = getattr(config, 'num_spatiotemporal_blocks', 2)
        self.st_blocks = nn.ModuleList([
            TimeSpaceTransformerBlock2(
                embed_dim=self.slot_dim,
                num_heads=getattr(config, 'num_heads', 4),
                qkv_size=getattr(config, 'qkv_size', 128),
                mlp_size=getattr(config, 'mlp_size', 256),
                pre_norm=getattr(config, 'pre_norm', False),
                dropout_rate=getattr(config, 'dropout_rate', 0.1),
            ) for _ in range(nst)
        ])

    def forward(self, slots, buffer, return_energy=False):
        q_next, p_next = self.hamiltonian(slots, buffer)
        st_out = torch.zeros_like(slots)
        for block in self.st_blocks:
            st_out = block(st_out, buffer)
        next_slots = q_next + st_out
        if return_energy:
            return next_slots, None
        return next_slots
