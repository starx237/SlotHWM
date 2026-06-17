import torch
import torch.nn as nn
from typing import Optional
from models.hamiltonian import Slot_HamiltonianNet, TimeAttentionBlock
from models.attention import GeneralizedDotProductAttention, TimeSpaceTransformerBlock2
from models.misc import MLP


class ResidualMLP(nn.Module):
    '''残差 MLP：forward(x) = x + MLP(x)。零初始化输出层时 ≈ identity。'''
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
    def forward(self, x):
        return x + self.mlp(x)


class SlotPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 维度说明：
        #   slot_hidden (slot_attention 输出) = static_dim + dynamic_dim (f_z 总维数)
        #   f_z: S_raw (slot_hidden) ↔ [Z^c(static_dim) | Z₀(dynamic_dim)]
        #   Z^d = [Z₀ | p], p = pos_enc_dim 维位置编码
        #   total Z = [Z^c | Z₀ | p], slot_dim = static_dim + dynamic_dim + pos_enc_dim
        #
        # Z^d 中的位置编码是 DETERMINISTIC 三角编码，无需可学习参数：
        #   编码: p = encode_pos_to_zd(centroid, pos_enc_dim)
        #         centroid = (pos_y, pos_x) 从 slot attention 注意力图计算
        #   解码: (pos_y, pos_x) = atan2(p[0], p[1]), atan2(p[2], p[3])
        #         PE_32 = reconstruct_pe(pos_y, pos_x) 无损重建
        self.static_dim = getattr(config, 'static_dim', 128)
        self.dyn_core_dim = getattr(config, 'dynamic_dim', 128)
        self.pos_enc_dim = getattr(config, 'pos_enc_dim', 8)
        self.dyn_total_dim = self.dyn_core_dim + self.pos_enc_dim
        self.slot_dim = self.static_dim + self.dyn_total_dim
        self.hidden_dim = getattr(config, 'hidden_dim', 256)
        self.num_heads = getattr(config, 'num_heads', 4)
        self.qkv_size = getattr(config, 'qkv_size', 128)
        self.dropout_rate = getattr(config, 'dropout_rate', 0.1)
        self.num_slots = getattr(config, 'num_slots', 7)
        self.buffer_len = getattr(config, 'buffer_len', 10)

        # 第四次修改：freeze_C 配置 — C 使用 TimeAttentionBlock 聚合 burnin Z^c
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

        # === 物理模块：基于哈密顿力学 ===
        # embed_dim = dyn_total_dim = dyn_core_dim + 2 (pos_y, pos_x)
        # 积分器在融合了位置信息的 Z^d 空间运动
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

        # === 时空推理模块 ===
        num_spatiotemporal_blocks = getattr(config, 'num_spatiotemporal_blocks', 2)
        spatiotemporal_mlp_size = getattr(config, 'spatiotemporal_mlp_size',
                                           getattr(config, 'mlp_size', 256))
        spatiotemporal_pre_norm = getattr(config, 'spatiotemporal_pre_norm',
                                           getattr(config, 'pre_norm', False))
        self.spatiotemporal_module = nn.ModuleList([
            TimeSpaceTransformerBlock2(
                embed_dim=self.slot_dim,
                num_heads=self.num_heads,
                qkv_size=self.qkv_size,
                mlp_size=spatiotemporal_mlp_size,
                pre_norm=spatiotemporal_pre_norm,
            ) for _ in range(num_spatiotemporal_blocks)
        ])

        # fusion_mlp 将积分器输出的 q_next 映射回 Z^d
        # ResidualMLP: forward(x) = x + MLP(x)，零初始化时 ≈ identity
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
        '''
        从 burnin Z^c 序列通过 TimeAttentionBlock 聚合全局 C。
        - Query: 所有 burnin 帧 Z^c 的均值（每个 Z^c_t 有 1/T 的残差路径）
        - Key/Value: 所有 burnin 帧 Z^c（K 加时间 PE, V 不加）
        '''
        if not self.freeze_C:
            return burnin_slots[:, -1, :, :self.static_dim]

        B, T, N, _ = burnin_slots.shape
        D_sta = self.static_dim
        sta = burnin_slots[:, :, :, :D_sta]  # (B, T, N, D_sta)
        query = sta.mean(dim=1)  # (B, N, D_sta)

        return self.C_time_attn(query, sta, query_pos=-1)
        # return query

    def forward(self, z, z_buffer, C: Optional[torch.Tensor] = None, return_energy=False,
                return_qp=False):
        '''
        在 Z 空间做单步预测。
        Args:
            ...
            return_energy: 是否返回能量对
            return_qp: 是否返回新鲜 (q, p) 和预测 (q_next, p_next)
        Returns:
            pred_z_next, 可选额外信息。
            return_energy=True → (…, energy_pair)
            return_qp=True    → (…, (fresh_q, fresh_p), (q_next, p_next))
            顺序: pred_z_next, [energy_pair], [fresh_qp], [q_next_p_next]
        '''
        B, N, D = z.shape
        D_sta = self.static_dim
        D_dyn = self.dyn_total_dim

        if C is None:
            C = z[:, :, :D_sta]

        # ============ 物理模块 ============
        z_dyn = z[:, :, D_sta:]  # (B, N, dyn_total_dim = [Z₀ | p])
        z_buffer_dyn = z_buffer[:, :, :, D_sta:]  # (B, T, N, dyn_total_dim)
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

        # 第四次修改：MLP_Z 只接收 Q_next（不含 C），因为 Z^d 应不包含 C 的信息
        next_dyn = self.fusion_mlp(q_next)             # (B, N, dynamic_dim)

        # 第四次修改：根据 freeze_C 决定 Z^c 的来源
        if self.freeze_C:
            z_phys = torch.cat([C, next_dyn], dim=-1)  # Z^c 冻结为全局 C
        else:
            z_phys = torch.cat([z[:, :, :D_sta], next_dyn], dim=-1)

        # ============ 时空推理模块（纯增量修正） ============
        st_out = torch.zeros_like(z)
        # for block in self.spatiotemporal_module:
        #     st_out = block(st_out, z_buffer)

        # ============ 融合 ============
        if self.freeze_C:
            pred_z_next = z_phys.clone()
            pred_z_next[:, :, D_sta:] = pred_z_next[:, :, D_sta:] + st_out[:, :, D_sta:]
        else:
            pred_z_next = z_phys + st_out  # (B, N, D)

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