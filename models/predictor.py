import torch
import torch.nn as nn
from typing import Optional
from models.hamiltonian import Slot_HamiltonianNet
from models.attention import GeneralizedDotProductAttention, TimeSpaceTransformerBlock2
from models.misc import MLP



class SlotPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 从配置中获取两个子空间的维度
        self.static_dim = getattr(config, 'static_dim', 128)
        self.dynamic_dim = getattr(config, 'dynamic_dim', 128)
        self.slot_dim = self.static_dim + self.dynamic_dim
        self.hidden_dim = getattr(config, 'hidden_dim', 256)
        self.num_heads = getattr(config, 'num_heads', 4)
        self.qkv_size = getattr(config, 'qkv_size', 128)
        self.dropout_rate = getattr(config, 'dropout_rate', 0.1)
        self.num_slots = getattr(config, 'num_slots', 7)
        self.buffer_len = getattr(config, 'buffer_len', 10)

        # 第二次修改：C_t = S^c_t，不再从 burnin 序列计算全局 C
        # 原先的 C 上下文计算模块：
        # self.C_attn = GeneralizedDotProductAttention()
        # self.dense_cq = nn.Linear(self.static_dim, self.qkv_size)
        # self.dense_ck = nn.Linear(self.static_dim, self.qkv_size)
        # self.dense_cv = nn.Linear(self.static_dim, self.qkv_size)
        # self.dense_co = nn.Linear(self.qkv_size, self.static_dim)

        # === 物理模块：基于哈密顿力学 ===
        self.physics_module = Slot_HamiltonianNet(
            num_slots=self.num_slots,
            embed_dim=self.dynamic_dim,
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

        # === MLP_S / MLP_Z：融合 Q_next 和 C 生成新的动态特征（2 层隐藏层，hidden_dim//2，SiLU）===
        self.fusion_mlp = MLP(
            input_size=self.static_dim + self.dynamic_dim,
            hidden_size=self.hidden_dim // 2,
            output_size=self.dynamic_dim,
            num_hidden_layers=2,
            activate_output=True,
            activation_fn=nn.SiLU,
        )

    def compute_C(self, burnin_slots):
        '''
        第二次修改：C_t = S^c_t，此方法不再使用。
        原先从 burnin 序列计算全局 C 的代码已被注释保留。
        '''
        # 原始实现：
        # B, T, N, D = burnin_slots.shape
        # D_sta = self.static_dim
        # sta = burnin_slots[:, :, :, :D_sta]  # (B, T, N, D')
        #
        # # 注入时间位置编码
        # pos_freq = 1.0 / (10000.0 ** (torch.arange(0, D_sta, 2, device=burnin_slots.device).float() / D_sta))
        # pos_t = torch.arange(T, device=burnin_slots.device).float().unsqueeze(1)
        # pos_enc = torch.zeros(T, D_sta, device=burnin_slots.device)
        # pos_enc[:, 0::2] = torch.sin(pos_t * pos_freq)
        # pos_enc[:, 1::2] = torch.cos(pos_t * pos_freq)
        # pos_enc = pos_enc[None, :, None, :]  # (1, T, 1, D')
        # sta = sta + pos_enc
        #
        # # 在时间维度上做自注意力，得到聚合的上下文 C
        # sta_flat = sta.reshape(B * N, T, D_sta)  # (B*N, T, D')
        # q = self.dense_cq(sta_flat).view(B * N, T, self.num_heads, -1)
        # k = self.dense_ck(sta_flat).view(B * N, T, self.num_heads, -1)
        # v = self.dense_cv(sta_flat).view(B * N, T, self.num_heads, -1)
        # attn_out, _ = self.C_attn(q[:, -1:], k, v)  # (B*N, 1, H, Dh)
        # attn_out = attn_out.view(B, N, self.qkv_size)
        # C = self.dense_co(attn_out)  # (B, N, D')
        # return C
        return burnin_slots[:, -1, :, :self.static_dim]

    def forward(self, z, z_buffer, C: Optional[torch.Tensor] = None, return_energy=False):
        '''
        在 Z 空间做单步预测。
        Args:
            z: 当前时间步的 Z (B, N, D)，E2E 训练时就是 S
            z_buffer: 历史 Z 缓存 (B, T, N, D)
            C: 静态部分 Z^c (B, N, D')，默认从 z 切片
            return_energy: 是否返回哈密顿量能量
        Returns:
            pred_z_next: 预测的下一时间步 Z (B, N, D)
        '''
        B, N, D = z.shape
        D_sta = self.static_dim
        D_dyn = self.dynamic_dim

        if C is None:
            C = z[:, :, :D_sta]

        # ============ 物理模块 ============
        z_dyn = z[:, :, D_sta:]  # (B, N, dynamic_dim)
        z_buffer_dyn = z_buffer[:, :, :, D_sta:]  # (B, T, N, dynamic_dim)
        phys_out = self.physics_module(z_dyn, z_buffer_dyn, C=C,
                                        return_energy=return_energy)
        if return_energy:
            q_next, p_next, energy_pair = phys_out
        else:
            q_next, p_next = phys_out
            energy_pair = None

        # 融合：MLP(Q_next, C) → Z^d_next
        fusion_input = torch.cat([q_next, C], dim=-1)  # (B, N, static_dim + dynamic_dim)
        next_dyn = self.fusion_mlp(fusion_input)       # (B, N, dynamic_dim)
        z_phys = torch.cat([z[:, :, :D_sta], next_dyn], dim=-1)  # (B, N, D) = [Z^c | Z^d_hat]

        # ============ 时空推理模块（纯增量修正） ============
        st_out = torch.zeros_like(z)
        for block in self.spatiotemporal_module:
            st_out = block(st_out, z_buffer)

        # ============ 融合 ============
        pred_z_next = z_phys + st_out  # (B, N, D)

        if return_energy:
            return pred_z_next, energy_pair
        return pred_z_next

    def predict_rollout(self, initial_z, initial_z_buffer, C, rollout_steps):
        '''自回归的 rollout 预测（Z 空间，多步）。'''
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