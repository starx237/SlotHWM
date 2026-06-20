import torch
import torch.nn as nn
from typing import Optional
from models.hamiltonian import Slot_HamiltonianNet, TimeAttentionBlock
from models.attention import GeneralizedDotProductAttention, TimeSpaceTransformerBlock2
from models.misc import MLP


class ResidualMLP(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
    def forward(self, x):
        return x + self.mlp(x)


class SlotPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ISA slot dim = appearance_dim(64) + pos(2) + depth(1) = 67
        # Z = [Z^c(static_dim) | Z_d_temp(64-static_dim) | pos(2) | depth(1)]
        # Z 总维数 = static_dim + dynamic_dim = 67
        # Z^d = [Z_d_temp | pos | depth] = dynamic_dim
        self.static_dim = getattr(config, 'static_dim', 34)
        self.dynamic_dim = getattr(config, 'dynamic_dim', 33)
        self.slot_dim = self.static_dim + self.dynamic_dim
        self.dyn_total_dim = self.dynamic_dim
        self.appearance_dim = getattr(config, 'appearance_dim', 64)
        self.hidden_dim = getattr(config, 'hidden_dim', 256)
        self.num_heads = getattr(config, 'num_heads', 4)
        self.qkv_size = getattr(config, 'qkv_size', 128)
        self.dropout_rate = getattr(config, 'dropout_rate', 0.1)
        self.num_slots = getattr(config, 'num_slots', 7)
        self.buffer_len = getattr(config, 'buffer_len', 10)

        self.freeze_C = getattr(config, 'freeze_C', False)
        if self.freeze_C:
            self.C_time_attn = TimeAttentionBlock(
                embed_dim=self.static_dim,
                num_heads=self.num_heads,
                qkv_size=self.qkv_size,
                mlp_size=self.hidden_dim // 2,
                pre_norm=True,
                dropout_rate=self.dropout_rate,
                zero_init=True,
            )

        # 物理模块：在 Z^d 空间运动 (dynamic_dim)
        self.physics_module = Slot_HamiltonianNet(
            num_slots=self.num_slots,
            embed_dim=self.dyn_total_dim,
            static_dim=self.static_dim,
            integrator_method=getattr(config, 'integrator_method', 'Euler'),
            num_heads=self.num_heads,
            qkv_size=self.qkv_size,
            mlp_size=getattr(config, 'mlp_size', 256),
            pre_norm=getattr(config, 'pre_norm', False),
            dropout_rate=self.dropout_rate,
            num_layers=getattr(config, 'num_layers', 1),
            h_layers=getattr(config, 'h_layers', 1),
            out_mlp=getattr(config, 'out_mlp', False),
            out_hidden_layers=getattr(config, 'out_hidden_layers', 1),
        )

        # 时空推理模块：freeze_C=true 时只作用在 Z^d，否则作用在完整 Z
        num_spatiotemporal_blocks = getattr(config, 'num_spatiotemporal_blocks', 2)
        spatiotemporal_mlp_size = getattr(config, 'spatiotemporal_mlp_size',
                                           getattr(config, 'mlp_size', 256))
        spatiotemporal_pre_norm = getattr(config, 'spatiotemporal_pre_norm',
                                           getattr(config, 'pre_norm', False))
        st_dim = self.dyn_total_dim if self.freeze_C else self.slot_dim
        self.spatiotemporal_module = nn.ModuleList([
            TimeSpaceTransformerBlock2(
                embed_dim=st_dim,
                num_heads=self.num_heads,
                qkv_size=self.qkv_size,
                mlp_size=spatiotemporal_mlp_size,
                pre_norm=spatiotemporal_pre_norm,
            ) for _ in range(num_spatiotemporal_blocks)
        ])

        self.fusion_mlp = ResidualMLP(MLP(
            input_size=self.dyn_total_dim,
            hidden_size=self.hidden_dim // 2,
            output_size=self.dyn_total_dim,
            num_hidden_layers=2,
            activate_output=False,
        ))
        nn.init.zeros_(self.fusion_mlp.mlp.net[-1].weight)
        nn.init.zeros_(self.fusion_mlp.mlp.net[-1].bias)

    def compute_C(self, burnin_slots):
        if not self.freeze_C:
            return burnin_slots[:, -1, :, :self.static_dim]

        B, T, N, _ = burnin_slots.shape
        D_sta = self.static_dim
        sta = burnin_slots[:, :, :, :D_sta]
        query = sta.mean(dim=1)

        return self.C_time_attn(query, sta, query_pos=-1)

    def forward(self, z, z_buffer, C: Optional[torch.Tensor] = None, return_energy=False,
                return_qp=False):
        B, N, D = z.shape
        D_sta = self.static_dim
        D_dyn = self.dyn_total_dim

        if C is None:
            C = z[:, :, :D_sta]

        # Z^d = Z 的后 dynamic_dim 维
        z_dyn = z[:, :, D_sta:]
        z_buffer_dyn = z_buffer[:, :, :, D_sta:]
        phys_out = self.physics_module(z_dyn, z_buffer_dyn, C=C,
                                        return_energy=return_energy,
                                        return_qp=return_qp)

        if return_energy and return_qp:
            q_next, p_next, energy_pair, (fresh_q, fresh_p) = phys_out
        elif return_energy:
            q_next, p_next, energy_pair = phys_out
        elif return_qp:
            q_next, p_next, (fresh_q, fresh_p) = phys_out
        else:
            q_next, p_next = phys_out
            energy_pair, fresh_q, fresh_p = None, None, None

        next_dyn = self.fusion_mlp(q_next)

        if self.freeze_C:
            z_phys = torch.cat([C, next_dyn], dim=-1)
        else:
            z_phys = torch.cat([z[:, :, :D_sta], next_dyn], dim=-1)

        # 时空推理模块：freeze_C 时只处理 Z^d 部分
        if self.freeze_C:
            # st_in = next_dyn
            # for block in self.spatiotemporal_module:
            #     st_dyn = block(st_in, z_buffer_dyn)
            st_out = torch.zeros_like(z_phys)
            # st_out[:, :, D_sta:] = st_dyn
        else:
            st_out = torch.zeros_like(z_phys)
            for block in self.spatiotemporal_module:
                st_out = block(st_out, z_buffer)

        if self.freeze_C:
            pred_z_next = z_phys.clone()
            pred_z_next[:, :, D_sta:] = pred_z_next[:, :, D_sta:] + st_out[:, :, D_sta:]
        else:
            pred_z_next = z_phys + st_out

        ret = [pred_z_next]
        if return_energy:
            ret.append(energy_pair)
        if return_qp:
            ret.append((fresh_q, fresh_p))
            ret.append((q_next, p_next))

        if len(ret) == 1:
            return pred_z_next
        return tuple(ret)

    def predict_rollout(self, initial_z, initial_z_buffer, C, rollout_steps):
        B, N, D = initial_z.shape
        z_buffer = initial_z_buffer
        z = initial_z
        predictions = []

        for _ in range(rollout_steps):
            next_z = self.forward(z, z_buffer, C=C)
            predictions.append(next_z)
            z_buffer = torch.cat([z_buffer[:, 1:], next_z.unsqueeze(1)], dim=1)
            z = next_z

        return torch.stack(predictions, dim=1)
