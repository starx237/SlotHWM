import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange


class GeneralizedDotProductAttention(nn.Module):
    def forward(self, query, key, value, bias=None):
        attn = torch.einsum("...qh d,...kh d->...qkh", query, key)
        if bias is not None:
            attn = attn + bias
        attn = F.softmax(attn, dim=-1)
        return torch.einsum("...qkh,...kh d->...qh d", attn, value), attn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_size, mlp_size, pre_norm=False,
                 weight_init=None, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.pre_norm = pre_norm
        self.dropout_rate = dropout_rate

        assert num_heads >= 1
        assert qkv_size % num_heads == 0
        self.head_dim = qkv_size // num_heads

        self.attn = GeneralizedDotProductAttention()
        self.dense_q = nn.Linear(embed_dim, qkv_size)
        self.dense_k = nn.Linear(embed_dim, qkv_size)
        self.dense_v = nn.Linear(embed_dim, qkv_size)
        self.dense_o = nn.Linear(qkv_size, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, embed_dim),
        )

        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        if self.dropout_rate > 0:
            self.att_drop = nn.Dropout(self.dropout_rate)
            self.ff_drop = nn.Dropout(self.dropout_rate)

    def forward(self, x, padding_mask=None):
        B, N, D = x.shape

        if self.pre_norm:
            x_norm = self.layernorm1(x)
            q = self.dense_q(x_norm).view(B, N, self.num_heads, self.head_dim)
            k = self.dense_k(x_norm).view(B, N, self.num_heads, self.head_dim)
            v = self.dense_v(x_norm).view(B, N, self.num_heads, self.head_dim)
            attn_out, _ = self.attn(query=q, key=k, value=v)
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
            attn_out, _ = self.attn(query=q, key=k, value=v)
            attn_out = self.dense_o(attn_out.reshape(B, N, self.qkv_size))
            attn_out = x + self.att_drop(attn_out) if self.dropout_rate > 0 else x + attn_out
            attn_out = self.layernorm1(attn_out)
            y = attn_out
            z = self.mlp(y)
            z = y + self.ff_drop(z) if self.dropout_rate > 0 else y + z
            z = self.layernorm2(z)
        return z
