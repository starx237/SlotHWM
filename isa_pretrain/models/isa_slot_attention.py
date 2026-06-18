import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
