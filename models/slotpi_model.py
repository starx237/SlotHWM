import torch
import torch.nn as nn
from typing import Optional
from models.hamiltonian import Slot_HamiltonianNet
from models.attention import GeneralizedDotProductAttention, TimeSpaceTransformerBlock2
from models.misc import MLP



class SlotPiModel(nn.Module):
    '''
    SlotPI（Slot Physics Integration）模型 - 新架构版本（idea.md 修正）。
    将原始 slot 拆分为两部分：
      S = [S^c || S^d]，其中 S^c 为静态特征 (first half)，S^d 为动态特征 (last half)。
    - 动态特征 S^d → qp_Attentions → (Q, P) → HamiltonianNet(Q, P, C) → Integrator → (Q_next, P_next)
    - 静态特征 S^c → SelfAtt_C → Linear_C → C（在 burn-in 阶段计算，rollout 阶段固定）
    - 融合构建物理预测：slot_phys = [S^c || MLP_S(Q_next, C)]
    - 时空推理 ST_{t+1} 修正完整 slot（包含静态和动态两部分）
    - 最终预测：S_{t+1} = slot_phys + ST_{t+1}
    - L_LC 损失约束 S^c 在时间维上的方差（静态特征应变化尽量小）
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 从配置中获取两个子空间的维度（使用 getattr 确保兼容性）
        self.slot_dim = getattr(config, 'slot_dim', 128)
        self.hidden_dim = getattr(config, 'hidden_dim', 256)
        self.num_heads = getattr(config, 'num_heads', 4)
        self.qkv_size = getattr(config, 'qkv_size', 128)
        self.dropout_rate = getattr(config, 'dropout_rate', 0.1)
        self.num_slots = getattr(config, 'num_slots', 7)
        self.buffer_len = getattr(config, 'buffer_len', 10)

        # === C 上下文计算模块：从静态特征序列中提取上下文 C ===
        # SelfAtt_C: 自注意力作用于历史帧的静态特征
        self.C_attn = GeneralizedDotProductAttention()
        self.dense_cq = nn.Linear(self.slot_dim // 2, self.qkv_size)
        self.dense_ck = nn.Linear(self.slot_dim // 2, self.qkv_size)
        self.dense_cv = nn.Linear(self.slot_dim // 2, self.qkv_size)
        self.dense_co = nn.Linear(self.qkv_size, self.slot_dim // 2)

        # === 物理模块：基于哈密顿力学 ===
        self.physics_module = Slot_HamiltonianNet(
            num_slots=self.num_slots,
            embed_dim=self.slot_dim // 2,  # q, p, C 的维度（动态子空间）
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

        # === MLP_S：融合 Q_next 和 C 生成新的动态特征 ===
        self.fusion_mlp = MLP(
            input_size=self.slot_dim,  # [Q_next, C] 的拼接：slot_dim/2 + slot_dim/2 = slot_dim
            hidden_size=self.hidden_dim,
            output_size=self.slot_dim // 2,
            num_hidden_layers=1,
            activate_output=True,
        )

    def compute_C(self, burnin_slots):
        '''
        从 burn-in 阶段的静态特征序列计算 C。
        在 SelfAtt_C 之前，S^c 会先注入时间位置编码（idea.md §0）。
        Args:
            burnin_slots: (B, T_burn, N, D) burn-in 阶段的所有 slots
        Returns:
            C: (B, N, D') 静态上下文，D' = slot_dim//2
        '''
        B, T, N, D = burnin_slots.shape
        D_sta = D // 2
        D_dyn = D - D_sta
        # 提取静态特征 S^c，只对前半部分（静态特征）操作
        sta = burnin_slots[:, :, :, :D_sta]  # (B, T, N, D')

        # 注入时间位置编码（S^c 只需注入时间位置编码，无需注入空间位置编码）
        pos_freq = 1.0 / (10000.0 ** (torch.arange(0, D_sta, 2, device=burnin_slots.device).float() / D_sta))
        pos_t = torch.arange(T, device=burnin_slots.device).float().unsqueeze(1)
        pos_enc = torch.zeros(T, D_sta, device=burnin_slots.device)
        pos_enc[:, 0::2] = torch.sin(pos_t * pos_freq)
        pos_enc[:, 1::2] = torch.cos(pos_t * pos_freq)
        pos_enc = pos_enc[None, :, None, :]  # (1, T, 1, D')
        sta = sta + pos_enc

        # 在时间维度上做自注意力，得到聚合的上下文 C
        sta_flat = sta.reshape(B * N, T, D_sta)  # (B*N, T, D')
        q = self.dense_cq(sta_flat).view(B * N, T, self.num_heads, -1)
        k = self.dense_ck(sta_flat).view(B * N, T, self.num_heads, -1)
        v = self.dense_cv(sta_flat).view(B * N, T, self.num_heads, -1)
        # 用最后一个时间步作为 query（更稳定，因为 burnin 结束时信息最丰富）
        attn_out, _ = self.C_attn(q[:, -1:], k, v)  # (B*N, 1, H, Dh)
        attn_out = attn_out.view(B, N, self.qkv_size)
        C = self.dense_co(attn_out)  # (B, N, D')
        return C

    def forward(self, slots, buffer, C: Optional[torch.Tensor] = None, return_energy=False):
        '''
        Args:
            slots: 当前时间步的 Slot (B, N, D)
            buffer: 历史帧缓存 (B, T, N, D)
            C: 静态上下文 (B, N, D')，如果为 None 则在 buffer 上即时计算
            return_energy: 是否返回哈密顿量能量
        Returns:
            pred_slots_next: 预测的下一时间步 Slot (B, N, D)
            如果 return_energy=True，额外返回 energy (B,)
        '''
        B, N, D = slots.shape
        D_sta = D // 2

        # 计算或获取 C（静态上下文）
        if C is None:
            C = self.compute_C(buffer)  # (B, N, D')

        # ============ 物理模块 ============
        # 传入的是完整的 slot，但物理模块内部只使用后半段（动态特征）
        q_next, p_next = self.physics_module(slots, buffer, C=C)
        # q_next, p_next: (B, N, D/2)

        # 融合：MLP_S(Q_next, C) → S^d_next，与静态部分拼接得到完整物理预测
        fusion_input = torch.cat([q_next, C], dim=-1)  # (B, N, D)
        next_dyn = self.fusion_mlp(fusion_input)       # (B, N, D/2)
        # 物理预测的完整 slot
        slot_phys = torch.cat([slots[:, :, :D_sta], next_dyn], dim=-1)  # (B, N, D)

        # ============ 时空推理模块 ============
        st_out = slots
        for block in self.spatiotemporal_module:
            st_out = block(st_out, buffer)  # (B, N, D) — 完整 slot 的修正量

        # ============ 融合 ============
        # ST_{t+1} 修正完整 slot（包含静态和动态两个部分）
        pred_slots_next = slot_phys + st_out  # (B, N, D)

        if return_energy:
            q, p = self.physics_module.qp_net(slots, buffer)
            energy = self.physics_module.H_net(q, p, C=C)
            return pred_slots_next, energy

        return pred_slots_next

    def predict_rollout(self, initial_slots, initial_buffer, C, rollout_steps):
        '''自回归的 rollout 预测（多步预测）。'''
        B, N, D = initial_slots.shape
        buffer = initial_buffer
        slots = initial_slots
        predictions = []

        for _ in range(rollout_steps):
            next_slots = self.forward(slots, buffer, C=C)
            predictions.append(next_slots)
            buffer = torch.cat([buffer[:, 1:], next_slots.unsqueeze(1)], dim=1)
            slots = next_slots

        return torch.stack(predictions, dim=1)