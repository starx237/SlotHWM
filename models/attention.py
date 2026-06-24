# attention.py — 注意力机制模块
# 包含通用点积注意力、TransformerBlock、SlotAttention、TimeSpaceTransformerBlock2等

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange
from models.misc import MLP


class GeneralizedDotProductAttention(nn.Module):
    '''
    广义点积注意力机制。
    支持任意批次维度、多头注意力和可选的注意力偏置。
    '''
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, bias=None,
                symmetrize_attn=False, self_mask=False):
        """
        Args:
            query: (B, *Q, H, D) 查询张量
            key: (B, *K, H, D) 键张量
            value: (B, *K, H, D) 值张量
            bias: 可选的注意力偏置
            symmetrize_attn: 是否对称化注意力权重 W = (W + W^T) / 2
            self_mask: 是否将对角线（Q=K）位置置零（阻止自交互）
        Returns:
            (B, *Q, H, D) 注意力加权后的值张量 和 注意力权重
        """
        assert query.shape[-1] == key.shape[-1]
        attn = torch.einsum("...qh d,...kh d->...qkh", query, key)
        if bias is not None:
            attn = attn + bias
        attn = F.softmax(attn, dim=-1)
        if self_mask:
            N = attn.shape[-2]
            eye = torch.eye(N, dtype=torch.bool, device=attn.device)
            attn = attn.masked_fill(eye[None, :, :, None], 0)
        if symmetrize_attn:
            attn = (attn + attn.transpose(-3, -2)) / 2
        return torch.einsum("...qkh,...kh d->...qh d", attn, value), attn


class TransformerBlock(nn.Module):
    '''
    标准 Transformer 块，包含多头自注意力和 MLP 前馈网络。
    支持预归一化和后归一化两种模式，可选 dropout。
    '''
    def __init__(self, embed_dim, num_heads, qkv_size, mlp_size, pre_norm=False, weight_init=None, dropout_rate=0.1, activation_fn=nn.ReLU):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.pre_norm = pre_norm
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn

        assert num_heads >= 1
        assert qkv_size % num_heads == 0, "embed dim must be divisible by num_heads"
        self.head_dim = qkv_size // num_heads

        # 多头自注意力
        self.attn = GeneralizedDotProductAttention()
        self.dense_q = nn.Linear(embed_dim, qkv_size)
        self.dense_k = nn.Linear(embed_dim, qkv_size)
        self.dense_v = nn.Linear(embed_dim, qkv_size)
        self.dense_o = nn.Linear(qkv_size, embed_dim)

        # MLP 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_size),
            activation_fn(),
            nn.Linear(mlp_size, embed_dim),
        )

        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        if self.dropout_rate > 0:
            self.att_drop = nn.Dropout(self.dropout_rate)
            self.ff_drop = nn.Dropout(self.dropout_rate)

    def forward(self, x, padding_mask=None, bias=None,
                symmetrize_attn=False, self_mask=False):
        B, N, D = x.shape

        if self.pre_norm:
            x_norm = self.layernorm1(x)
            q = self.dense_q(x_norm).view(B, N, self.num_heads, self.head_dim)
            k = self.dense_k(x_norm).view(B, N, self.num_heads, self.head_dim)
            v = self.dense_v(x_norm).view(B, N, self.num_heads, self.head_dim)
            attn_out, _ = self.attn(query=q, key=k, value=v, bias=bias,
                                    symmetrize_attn=symmetrize_attn, self_mask=self_mask)
            attn_out = self.dense_o(attn_out.reshape(B, N, self.qkv_size))
            attn_out = x + self.att_drop(attn_out) if self.dropout_rate > 0 else x + attn_out
            y = attn_out
            y_norm = self.layernorm2(y)
            z = self.mlp(y_norm)
            z = y + self.ff_drop(z) if self.dropout_rate > 0 else y + z
        else:
            q = self.dense_q(x).view(B, N, self.num_heads, self.head_dim)
            k = self.dense_k(x).view(B, N, self.num_heads, self.head_dim)
            v = self.dense_v(x).view(B, N, self.num_heads, self.head_dim)
            attn_out, _ = self.attn(query=q, key=k, value=v, bias=bias,
                                    symmetrize_attn=symmetrize_attn, self_mask=self_mask)
            attn_out = self.dense_o(attn_out.reshape(B, N, self.qkv_size))
            attn_out = x + self.att_drop(attn_out) if self.dropout_rate > 0 else x + attn_out
            attn_out = self.layernorm1(attn_out)
            y = attn_out
            z = self.mlp(y)
            z = y + self.ff_drop(z) if self.dropout_rate > 0 else y + z
            z = self.layernorm2(z)
        return z


class SlotAttention(nn.Module):
    '''
    Slot Attention 模块，用于从输入特征中迭代提取 Slot 表示。
    支持多头注意力和 GRU 循环更新，可选 MLP 前馈网络。
    '''
    def __init__(self, num_slots, slot_dim, hidden_dim, iters=3, num_heads=1, qkv_size=None, mlp_size=None, epsilon=1e-8):
        super().__init__()
        self.num_slots = num_slots          # Slot 数量
        self.slot_dim = slot_dim             # Slot 特征维度
        self.hidden_dim = hidden_dim         # GRU 隐藏层维度
        self.iters = iters                   # 迭代次数
        self.num_heads = num_heads           # 注意力头数
        self.qkv_size = qkv_size or slot_dim # QKV 投影维度
        self.mlp_size = mlp_size             # MLP 隐藏层大小（可选）
        self.epsilon = epsilon               # 防止除零的小常数

        head_dim = self.qkv_size // self.num_heads

        # Slot 的初始参数（可学习）
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slot_sigma = nn.Parameter(torch.randn(1, 1, slot_dim))

        # QKV 投影层
        self.dense_q = nn.Linear(slot_dim, num_heads * head_dim, bias=False)
        self.dense_k = nn.Linear(slot_dim, num_heads * head_dim, bias=False)
        self.dense_v = nn.Linear(slot_dim, num_heads * head_dim, bias=False)

        # GRU 循环更新
        self.gru = nn.GRUCell(input_size=slot_dim, hidden_size=slot_dim)

        # 可选 MLP
        if self.mlp_size is not None:
            self.mlp = nn.Sequential(
                nn.Linear(slot_dim, self.mlp_size),
                nn.ReLU(),
                nn.Linear(self.mlp_size, slot_dim),
            )

    def forward(self, inputs, slots=None):
        '''
        Args:
            inputs: 输入特征 (B, N, D)
            slots: 上一时间步的 Slot（可选），若为 None 则使用随机初始化的 Slot
        Returns:
            slots: 更新后的 Slot (B, num_slots, slot_dim)
            attn: 注意力权重 (B, num_slots, N)
        '''
        B, N, D = inputs.shape
        device = inputs.device

        # 初始化 Slot（若首次调用，使用可学习参数初始化；否则使用上一帧的 Slot）
        if slots is None:
            slots = self.slot_mu + torch.exp(self.slot_sigma) * torch.randn(B, self.num_slots, self.slot_dim, device=device)
        else:
            slots = slots.view(B, self.num_slots, self.slot_dim)

        # 对输入做 LayerNorm
        inputs_norm = F.layer_norm(inputs, (D,))
        # 投影 key 和 value
        k = self.dense_k(inputs_norm).view(B, N, self.num_heads, self.qkv_size // self.num_heads)
        v = self.dense_v(inputs_norm).view(B, N, self.num_heads, self.qkv_size // self.num_heads)

        # 多次迭代更新 Slot
        for _ in range(self.iters):
            slots_prev = slots
            # Slot 上的 LayerNorm
            slots_norm = F.layer_norm(slots, (self.slot_dim,))
            # 投影 query
            q = self.dense_q(slots_norm).view(B, self.num_slots, self.num_heads, self.qkv_size // self.num_heads)

            # 计算注意力（反转注意力：Slot 作为 query，特征作为 key-value）
            attn = torch.einsum("...qhd,...khd->...hqk", q, k)
            attn = attn / (self.qkv_size // self.num_heads) ** 0.5
            attn = F.softmax(attn, dim=-2)  # 在 Slot 维度上做 softmax（归一化 query 轴）
            attn = attn + self.epsilon
            # 归一化 key 轴（加权平均而非加权和）
            attn = attn / attn.sum(dim=-1, keepdim=True)

            # 计算更新量
            updates = torch.einsum("...hqk,...khd->...qhd", attn, v)
            updates = updates.reshape(B, self.num_slots, -1)

            # GRU 更新（直接替换，不加残差 — 残差在高维下会导致数值爆炸）
            gru_in = self.gru(updates.reshape(-1, self.slot_dim), slots_prev.reshape(-1, self.slot_dim))
            slots = gru_in.reshape(B, self.num_slots, self.slot_dim)

        # 可选 MLP
        if self.mlp_size is not None:
            slots = self.mlp(slots) + slots

        return slots, attn



class TimeSpaceTransformerBlock2(nn.Module):
    '''
    时空 Transformer 块（SOTA 方案）。
    同时进行空间注意力和时间注意力，然后相加融合，最后通过 MLP。
    '''
    def __init__(self, embed_dim, num_heads, qkv_size, mlp_size, pre_norm=False, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.pre_norm = pre_norm
        self.dropout_rate = dropout_rate

        assert num_heads >= 1
        assert qkv_size % num_heads == 0, "qkv_size 必须能被 num_heads 整除"
        self.head_dim = qkv_size // num_heads

        # 通用注意力机制
        self.attn = GeneralizedDotProductAttention()

        # QKV 投影（空间注意力用）
        self.dense_qs = nn.Linear(embed_dim, qkv_size)
        self.dense_ks = nn.Linear(embed_dim, qkv_size)
        self.dense_vs = nn.Linear(embed_dim, qkv_size)

        # QKV 投影（时间注意力用）
        self.dense_qt = nn.Linear(embed_dim, qkv_size)
        self.dense_kt = nn.Linear(embed_dim, qkv_size)
        self.dense_vt = nn.Linear(embed_dim, qkv_size)

        # 输出投影（空间和时间共享）
        self.dense_o = nn.Linear(qkv_size, embed_dim)
        # zero_init: 时空推理模块建模残差，初始输出应≈0
        nn.init.zeros_(self.dense_o.weight)
        nn.init.zeros_(self.dense_o.bias)

        # MLP
        self.mlp = MLP(
            input_size=embed_dim, hidden_size=mlp_size,
            output_size=embed_dim, weight_init=None)
        # zero_init mlp 最后一层
        nn.init.zeros_(self.mlp.net[-1].weight)
        nn.init.zeros_(self.mlp.net[-1].bias)

        # LayerNorm
        self.layernorm_q = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_kv = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_mlp = nn.LayerNorm(embed_dim, eps=1e-6)

        # Dropout
        if self.dropout_rate > 0:
            self.att_drop = nn.Dropout(self.dropout_rate)
            self.ff_drop = nn.Dropout(self.dropout_rate)

    def forward(self, queries, inputs, padding_mask=None, train=False):
        """
        Args:
            queries: (B, O, D) 当前查询（通常是当前帧的 Slot）
            inputs: (B, T, O, D) 历史帧的 Slot 缓存（时间维度 T）
            padding_mask: 可选填充掩码
            train: 是否为训练模式
        Returns:
            (B, O, D) 融合时空信息后的输出
        """
        del padding_mask, train  # 当前未使用
        assert queries.ndim == 3
        assert inputs.ndim == 4
        B, O, D = queries.shape

        if self.pre_norm:
            # === 预归一化路径 ===

            # 空间注意力：在当前帧的 Slot 之间做自注意力
            xs = self.layernorm_q(queries)
            qs = self.dense_qs(xs).view(B, O, self.num_heads, self.head_dim)
            ks = self.dense_ks(xs).view(B, O, self.num_heads, self.head_dim)
            vs = self.dense_vs(xs).view(B, O, self.num_heads, self.head_dim)
            xs, _ = self.attn(query=qs, key=ks, value=vs)
            xs = self.dense_o(xs.reshape(B, O, self.qkv_size))
            if self.dropout_rate > 0:
                xs = self.att_drop(xs)

            # 时间注意力：当前帧的 Slot 从历史缓存中聚合信息
            xt = self.layernorm_q(queries)
            xt = torch.unsqueeze(xt, dim=1)  # B, 1, O, D
            xt = rearrange(xt, 'b t o d -> (b o) t d')   # B*O, 1, D
            xt_buffer = self.layernorm_kv(inputs)
            xt_buffer = rearrange(xt_buffer, 'b t o d -> (b o) t d')  # B*O, T, D
            qt = self.dense_qt(xt).view(xt.shape[0], xt.shape[1], self.num_heads, self.head_dim)
            kt = self.dense_kt(xt_buffer).view(xt_buffer.shape[0], xt_buffer.shape[1], self.num_heads, self.head_dim)
            vt = self.dense_vt(xt_buffer).view(xt_buffer.shape[0], xt_buffer.shape[1], self.num_heads, self.head_dim)
            xt, _ = self.attn(query=qt, key=kt, value=vt)  # B*O, 1, h, d
            xt = xt.reshape(B, O, self.qkv_size)
            xt = self.dense_o(xt).view(B, O, self.embed_dim)
            if self.dropout_rate > 0:
                xt = self.att_drop(xt)

            # 融合空间和时间信息，一次残差连接 + MLP
            y = xt + xs + queries
            z = self.layernorm_mlp(y)
            z = self.mlp(z)
            if self.dropout_rate > 0:
                z = self.ff_drop(z)
            z = z + y
            # 建模残差：输出 = 变化量，不含 queries 本身
            return z - queries
        else:
            # === 后归一化路径 ===
            # 空间注意力
            xs = queries
            qs = self.dense_qs(xs).view(B, O, self.num_heads, self.head_dim)
            ks = self.dense_ks(xs).view(B, O, self.num_heads, self.head_dim)
            vs = self.dense_vs(xs).view(B, O, self.num_heads, self.head_dim)
            xs, _ = self.attn(query=qs, key=ks, value=vs)
            xs = self.dense_o(xs.reshape(B, O, self.qkv_size))
            if self.dropout_rate > 0:
                xs = self.att_drop(xs)

            # 时间注意力
            xt = torch.unsqueeze(queries, dim=1)  # B, 1, O, D
            xt = rearrange(xt, 'b t o d -> (b o) t d')   # B*O, 1, D
            xt_buffer = inputs
            xt_buffer = rearrange(xt_buffer, 'b t o d -> (b o) t d')  # B*O, T, D
            qt = self.dense_qt(xt).view(xt.shape[0], xt.shape[1], self.num_heads, self.head_dim)
            kt = self.dense_kt(xt_buffer).view(xt_buffer.shape[0], xt_buffer.shape[1], self.num_heads, self.head_dim)
            vt = self.dense_vt(xt_buffer).view(xt_buffer.shape[0], xt_buffer.shape[1], self.num_heads, self.head_dim)
            xt, _ = self.attn(query=qt, key=kt, value=vt)  # B*O, 1, h, d
            xt = xt.reshape(B, O, self.qkv_size)
            xt = self.dense_o(xt).view(B, O, self.embed_dim)
            if self.dropout_rate > 0:
                xt = self.att_drop(xt)

            # 融合 + MLP + 一次残差
            y = xt + xs + queries
            z = self.mlp(y)
            if self.dropout_rate > 0:
                z = self.ff_drop(z)
            z = z + y
            z = self.layernorm_mlp(z)
            # 建模残差：输出 = 变化量，不含 queries 本身
            return z - queries


class InvertedDotProductAttentionKeyPerQuery(nn.Module):
    def __init__(self, epsilon=1e-8, renormalize_keys=True,
                 softmax_temperature=1.0, value_per_query=False):
        super().__init__()
        self.epsilon = epsilon
        self.renormalize_keys = renormalize_keys
        self.softmax_temperature = softmax_temperature
        self.value_per_query = value_per_query

    def forward(self, query, key, value):
        qk_features = query.shape[-1]
        query = query / math.sqrt(qk_features)

        attn = torch.einsum("bqd,bqnd->bqn", query, key)

        attn = F.softmax(attn / self.softmax_temperature, dim=-2)

        if self.renormalize_keys:
            normalizer = attn.sum(dim=-1, keepdim=True) + self.epsilon
            attn = attn / normalizer

        if self.value_per_query:
            output = torch.einsum("bqn,bqnd->bqd", attn, value)
        else:
            output = torch.einsum("bqn,bnd->bqd", attn, value)

        return output, attn


class SlotAttentionTranslScaleEquiv(nn.Module):
    def __init__(self, num_slots, appearance_dim, feat_dim=64, qkv_size=None,
                 grid_enc_hidden=256, mlp_size=256, num_iterations=3, epsilon=1e-8,
                 min_scale=0.001, max_scale=2.0, scales_factor=5.0,
                 zero_position_init=False, init_with_fixed_scale=None,
                 add_rel_pos_to_values=True, softmax_temperature=1.0,
                 append_statistics=False):
        super().__init__()
        self.num_slots = num_slots
        self.appearance_dim = appearance_dim
        self.slot_dim = appearance_dim + 3
        self.qkv_size = qkv_size or self.slot_dim
        self.mlp_size = mlp_size
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scales_factor = scales_factor
        self.zero_position_init = zero_position_init
        self.init_with_fixed_scale = init_with_fixed_scale
        self.add_rel_pos_to_values = add_rel_pos_to_values
        self.softmax_temperature = softmax_temperature
        self.append_statistics = append_statistics

        if append_statistics:
            self.embed_statistics = nn.Linear(appearance_dim + 3, appearance_dim)

        self.input_norm = nn.LayerNorm(feat_dim)
        self.slot_norm = nn.LayerNorm(appearance_dim)

        self.grid_proj = nn.Linear(2, self.qkv_size)

        self.grid_enc = nn.Sequential(
            nn.LayerNorm(self.qkv_size),
            nn.Linear(self.qkv_size, grid_enc_hidden),
            nn.ReLU(),
            nn.Linear(grid_enc_hidden, self.qkv_size),
        )

        self.dense_q = nn.Linear(appearance_dim, self.qkv_size, bias=False)
        self.dense_k = nn.Linear(feat_dim, self.qkv_size, bias=False)
        self.dense_v = nn.Linear(feat_dim, self.qkv_size, bias=False)

        self.inverted_attention = InvertedDotProductAttentionKeyPerQuery(
            epsilon=epsilon,
            renormalize_keys=True,
            softmax_temperature=softmax_temperature,
            value_per_query=add_rel_pos_to_values,
        )

        self.gru = nn.GRUCell(self.qkv_size, appearance_dim)
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, appearance_dim))
        self.slot_depth = nn.Parameter(torch.ones(1, num_slots, 1))

        if mlp_size is not None:
            self.mlp = nn.Sequential(
                nn.LayerNorm(appearance_dim),
                nn.Linear(appearance_dim, mlp_size),
                nn.ReLU(),
                nn.Linear(mlp_size, appearance_dim),
            )

    def forward(self, inputs, slots=None, num_iterations=None):
        B, N, D = inputs.shape
        grid = inputs[..., -2:]
        features = inputs[..., :-2]

        if slots is None:
            appearance = self.slot_mu.expand(B, -1, -1)
            positions = torch.zeros(B, self.num_slots, 2, device=inputs.device)
            if not self.zero_position_init:
                positions = torch.empty(B, self.num_slots, 2, device=inputs.device).uniform_(-1, 1)
            depth = self.slot_depth.expand(B, -1, -1).clone()
            if self.init_with_fixed_scale is not None:
                depth = depth * 0. + self.init_with_fixed_scale
        else:
            appearance = slots[..., :-3]
            positions = slots[..., -3:-1]
            depth = slots[..., -1:]

        if self.zero_position_init:
            positions = positions * 0.

        depth = depth.clamp(self.min_scale, self.max_scale)

        inputs = self.input_norm(features)

        k = self.dense_k(inputs)
        v = self.dense_v(inputs)

        k_expand = k.unsqueeze(1).expand(-1, self.num_slots, -1, -1)
        v_expand = v.unsqueeze(1).expand(-1, self.num_slots, -1, -1)

        grid_expand = grid.unsqueeze(1).expand(-1, self.num_slots, -1, -1)

        n_iters = num_iterations if num_iterations is not None else self.num_iterations
        for attn_round in range(n_iters + 1):
            relative_grid = grid_expand - positions.unsqueeze(2)
            relative_grid = relative_grid / self.scales_factor
            relative_grid = relative_grid / (depth.unsqueeze(2) + self.epsilon)

            grid_emb = self.grid_proj(relative_grid)

            k_rel = self.grid_enc(k_expand + grid_emb)

            slots_n = self.slot_norm(appearance)
            q = self.dense_q(slots_n)

            if self.add_rel_pos_to_values:
                v_rel = self.grid_enc(v_expand + grid_emb)
                value = v_rel
            else:
                value = v

            updates, attn = self.inverted_attention(query=q, key=k_rel, value=value)

            positions = torch.einsum("bqn,bnd->bqd", attn, grid)

            spread = (grid.unsqueeze(1) - positions.unsqueeze(2)).pow(2).sum(dim=-1)
            depth = torch.sqrt(
                torch.einsum("bqn,bqn->bq", attn + self.epsilon, spread))
            depth = depth.unsqueeze(-1).clamp(self.min_scale, self.max_scale)

            if attn_round < n_iters:
                if self.append_statistics:
                    stats = torch.cat([appearance, positions, depth], dim=-1)
                    appearance = self.embed_statistics(stats)

                slots = self.gru(
                    updates.reshape(-1, self.qkv_size),
                    appearance.reshape(-1, self.appearance_dim),
                )
                appearance = slots.reshape(B, self.num_slots, self.appearance_dim)

                if self.mlp_size is not None:
                    appearance = self.mlp(appearance) + appearance

        output = torch.cat([appearance, positions, depth], dim=-1)
        return output, attn