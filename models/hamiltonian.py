# hamiltonian.py — 哈密顿力学网络模块
# 实现基于哈密顿力学的物理建模，包含积分器、空间/时间注意力块和哈密顿量网络

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.misc import MLP
from models.attention import GeneralizedDotProductAttention, TransformerBlock
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, TypeAlias, Union
from einops import rearrange

DType = Any
Array: TypeAlias = torch.Tensor  # np.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[
    str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]

class Integrator:
    '''
    哈密顿积分求解器，支持 Euler、RK4、Leapfrog 和 Yoshida 四种积分方法
    '''
    METHODS = ["Euler", "RK4", "Leapfrog", "Yoshida"]

    def __init__(self, delta_t=0.125, method="Euler"):
        """
        Args:
            delta_t: 积分时间步长
            method: 积分方法，可选 "Euler", "RK4", "Leapfrog", "Yoshida"
        """
        if method not in self.METHODS:
            msg = "%s is not a supported method. " % (method)
            msg += "Available methods are: " + "".join("%s " % m
                                                       for m in self.METHODS)
            raise KeyError(msg)

        self.delta_t = delta_t
        self.method = method

    def _get_grads(self, q, p, hnn, C=None, remember_energy=False):
        energy = hnn(q=q, p=p, C=C) if C is not None else hnn(q=q, p=p)
        dq_dt = torch.autograd.grad(energy, p, create_graph=True,
                                    grad_outputs=torch.ones_like(energy))[0]
        dp_dt = -torch.autograd.grad(energy, q, create_graph=True,
                                     grad_outputs=torch.ones_like(energy))[0]
        # 哈密顿梯度防爆：clip 范数而非归一化，保留幅值信息
        dq_norm = dq_dt.norm()
        dp_norm = dp_dt.norm()
        if dq_norm > 10.0:
            dq_dt = dq_dt / dq_norm * 10.0
        if dp_norm > 10.0:
            dp_dt = dp_dt / dp_norm * 10.0

        if remember_energy:
            self.energy = energy.detach().cpu().numpy()

        return dq_dt, dp_dt

    def _euler_step(self, q, p, hnn):
        '''欧拉法一步积分'''
        dq_dt, dp_dt = self._get_grads(q, p, hnn, C=self.C, remember_energy=True)
        q_next = q + self.delta_t * dq_dt
        p_next = p + self.delta_t * dp_dt
        return self._clamp_qp(q_next, p_next)

    def _rk_step(self, q, p, hnn):
        '''龙格-库塔四阶（RK4）方法一步积分'''
        k1_q, k1_p = self._get_grads(q, p, hnn, C=self.C, remember_energy=True)
        q_2 = q + self.delta_t * k1_q / 2
        p_2 = p + self.delta_t * k1_p / 2
        k2_q, k2_p = self._get_grads(q_2, p_2, hnn, C=self.C)
        q_3 = q + self.delta_t * k2_q / 2
        p_3 = p + self.delta_t * k2_p / 2
        k3_q, k3_p = self._get_grads(q_3, p_3, hnn, C=self.C)
        q_3 = q + self.delta_t * k3_q
        p_3 = p + self.delta_t * k3_p
        k4_q, k4_p = self._get_grads(q_3, p_3, hnn, C=self.C)
        q_next = q + self.delta_t * ((k1_q / 6) + (k2_q / 3) + (k3_q / 3) + (k4_q / 6))
        p_next = p + self.delta_t * ((k1_p / 6) + (k2_p / 3) + (k3_p / 3) + (k4_p / 6))
        return self._clamp_qp(q_next, p_next)

    def _lf_step(self, q, p, hnn):
        '''蛙跳（Leapfrog）法一步积分（二阶辛积分器）'''
        _, dp_dt = self._get_grads(q, p, hnn, C=self.C, remember_energy=True)
        p_next_half = p + dp_dt * (self.delta_t) / 2
        q_next = q + p_next_half * self.delta_t
        _, dp_next_dt = self._get_grads(q_next, p_next_half, hnn, C=self.C)
        p_next = p_next_half + dp_next_dt * (self.delta_t) / 2
        return self._clamp_qp(q_next, p_next)

    def _clamp_qp(self, q, p, max_norm=10.0):
        '''钳位 q/p 的逐元素范数，防止哈密顿积分发散。'''
        q = torch.clamp(q, -max_norm, max_norm)
        p = torch.clamp(p, -max_norm, max_norm)
        return q, p

    def _ys_step(self, q, p, hnn):
        '''四阶 Yoshida 辛积分器（由三个蛙跳复合而成）'''
        w_1 = 1./(2 - 2**(1./3))
        w_0 = -(2**(1./3))*w_1
        c_1 = c_4 = w_1/2.
        c_2 = c_3 = (w_0 + w_1)/2.
        d_1 = d_3 = w_1
        d_2 = w_0

        q_1 = q + c_1*p*self.delta_t
        _, a_1 = self._get_grads(q_1, p, hnn, C=self.C, remember_energy=True)
        p_1 = p + d_1*a_1*self.delta_t
        q_2 = q_1 + c_2*p_1*self.delta_t
        _, a_2 = self._get_grads(q_2, p_1, hnn, C=self.C)
        p_2 = p_1 + d_2*a_2*self.delta_t
        q_3 = q_2 + c_3*p_2*self.delta_t
        _, a_3 = self._get_grads(q_3, p_2, hnn, C=self.C)
        p_3 = p_2 + d_3*a_3*self.delta_t
        q_4 = q_3 + c_4*p_3*self.delta_t

        return self._clamp_qp(q_4, p_3)

    def step(self, q, p, hnn, C=None):
        self.C = C
        if self.method == "Euler":
            return self._euler_step(q, p, hnn)
        if self.method == "RK4":
            return self._rk_step(q, p, hnn)
        if self.method == "Leapfrog":
            return self._lf_step(q, p, hnn)
        if self.method == "Yoshida":
            return self._ys_step(q, p, hnn)
        raise NotImplementedError


class SpaceAttentionBlock(nn.Module):
    '''
    空间注意力块（q_AttentionBlock），用于同一时间步内对 Slot 之间的空间关系建模。
    输出作为位置（q）的更新。
    '''
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 qkv_size: int,
                 mlp_size: int,
                 pre_norm: bool = False,
                 weight_init=None,
                 dropout_rate: float = 0.1,
                 zero_init: bool = False,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        self.dropout_rate = dropout_rate

        assert num_heads >= 1
        assert qkv_size % num_heads == 0, "qkv_size 必须能被 num_heads 整除"
        self.head_dim = qkv_size // num_heads

        self.attn = GeneralizedDotProductAttention()
        # QKV 投影层
        self.dense_q = nn.Linear(embed_dim, qkv_size)
        self.dense_k = nn.Linear(embed_dim, qkv_size)
        self.dense_v = nn.Linear(embed_dim, qkv_size)
        # 输出 MLP
        self.mlp = MLP(
            input_size=embed_dim, hidden_size=mlp_size,
            output_size=embed_dim, weight_init=weight_init)

        self.layernorm_query = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_mlp = nn.LayerNorm(embed_dim, eps=1e-6)

        if self.num_heads > 1:
            self.dense_o = nn.Linear(qkv_size, embed_dim)

        if self.dropout_rate > 0:
            self.att_drop = nn.Dropout(self.dropout_rate)
            self.ff_dropout = nn.Dropout(self.dropout_rate)

        if zero_init:
            if self.num_heads > 1:
                nn.init.zeros_(self.dense_o.weight)
                nn.init.zeros_(self.dense_o.bias)
            nn.init.zeros_(self.mlp.net[-1].weight)
            nn.init.zeros_(self.mlp.net[-1].bias)

    def forward(self, inputs: Array, buffer: Array,
                padding_mask: Optional[Array] = None,
                train: bool = False) -> Array:
        assert inputs.ndim == 3
        del buffer, padding_mask, train  # 当前未使用 buffer 和 padding_mask
        B, N, D = inputs.shape
        head_dim = self.qkv_size // self.num_heads

        if self.pre_norm:
            # 预归一化路径
            x = self.layernorm_query(inputs)
            # 空间自注意力：所有 Slot 之间互相做注意力
            q = self.dense_q(x).view(B, N, self.num_heads, head_dim)
            k = self.dense_k(x).view(B, N, self.num_heads, head_dim)
            v = self.dense_v(x).view(B, N, self.num_heads, head_dim)
            xs, _ = self.attn(query=q, key=k, value=v)
            if self.num_heads > 1:
                xs = self.dense_o(xs.reshape(B, N, self.qkv_size)).view(B, N, self.embed_dim)
            else:
                xs = xs.squeeze(-2)
            if self.dropout_rate > 0:
                xs = self.att_drop(xs)
            xs = xs + inputs
            # MLP
            y = xs
            z = self.layernorm_mlp(y)
            z = self.mlp(z)
            if self.dropout_rate > 0:
                z = self.ff_dropout(z)
            z = z + y
        else:
            # 后归一化路径
            x = inputs
            q = self.dense_q(x).view(B, N, self.num_heads, head_dim)
            k = self.dense_k(x).view(B, N, self.num_heads, head_dim)
            v = self.dense_v(x).view(B, N, self.num_heads, head_dim)
            xs, _ = self.attn(query=q, key=k, value=v)
            if self.num_heads > 1:
                xs = self.dense_o(xs.reshape(B, N, self.qkv_size)).view(B, N, self.embed_dim)
            else:
                xs = xs.squeeze(-2)
            if self.dropout_rate > 0:
                xs = self.att_drop(xs)
            xs = xs + inputs
            xs = self.layernorm_query(xs)
            y = xs
            z = self.mlp(y)
            if self.dropout_rate > 0:
                z = self.ff_dropout(z)
            z = z + y
            z = self.layernorm_mlp(z)
        return z


class TimeAttentionBlock(nn.Module):
    '''
    时间注意力块，用于跨时间步对 Slot 的时序建模。
    输入为当前帧和缓存的历史帧，输出作为动量（p）的更新，也可以用于计算 C（内容变量）。
    '''
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 qkv_size: int,
                 mlp_size: int,
                 pre_norm: bool = False,
                 weight_init=None,
                 dropout_rate: float = 0.1,
                 zero_init: bool = False   # True 时初始化输出为零（用于 C）
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        self.dropout_rate = dropout_rate

        assert num_heads >= 1
        assert qkv_size % num_heads == 0, "qkv_size 必须能被 num_heads 整除"
        self.head_dim = qkv_size // num_heads

        self.attn = GeneralizedDotProductAttention()
        # QKV 投影层
        self.dense_tq = nn.Linear(embed_dim, qkv_size)
        self.dense_tk = nn.Linear(embed_dim, qkv_size)
        self.dense_tv = nn.Linear(embed_dim, qkv_size)
        # 输出 MLP
        self.mlp = MLP(
            input_size=embed_dim, hidden_size=mlp_size,
            output_size=embed_dim, weight_init=weight_init)

        self.layernorm_tquery = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_mlp = nn.LayerNorm(embed_dim, eps=1e-6)

        if self.num_heads > 1:
            self.dense_to = nn.Linear(qkv_size, embed_dim)

        if self.dropout_rate > 0:
            self.att_drop = nn.Dropout(self.dropout_rate)
            self.ff_dropout = nn.Dropout(self.dropout_rate)

        if zero_init:
            if self.num_heads > 1:
                nn.init.zeros_(self.dense_to.weight)
                nn.init.zeros_(self.dense_to.bias)
            nn.init.zeros_(self.mlp.net[-1].weight)
            nn.init.zeros_(self.mlp.net[-1].bias)

    def forward(self, inputs: Array, buffer: Array,
                padding_mask: Optional[Array] = None,
                train: bool = False,
                query_pos: Optional[int] = None) -> Array:
        '''
        Args:
            inputs: 当前帧 (B, N, D) — 用作 Query
            buffer: 历史帧缓存 (B, T, N, D) — 用作 Key/Value
            query_pos: Query 的时间位置，None=默认 T（向后兼容），-1=无 PE
        '''
        assert inputs.ndim == 3
        del padding_mask, train
        B, T, N, D = buffer.shape

        # 时间位置编码：加在 Q 和 K 上，不加在 V 上
        pe_buffer = None
        pe_query = None
        if T > 0:
            half = D // 2
            n_sin = (D + 1) // 2
            if n_sin > 0:
                freq = 1.0 / (10000.0 ** (torch.arange(0, n_sin, device=buffer.device).float() / max(1, half)))
                t_idx = torch.arange(T, device=buffer.device).float().unsqueeze(1)
                pe_buffer = torch.zeros(T, D, device=buffer.device)
                pe_buffer[:, 0::2] = torch.sin(t_idx * freq)
                if half > 0:
                    pe_buffer[:, 1::2] = torch.cos(t_idx * freq[:half])
                # Query PE: None=位置T（默认），-1=无PE，>=0=指定位置
                pe_query = torch.zeros(D, device=buffer.device)
                if query_pos is not None and query_pos < 0:
                    pass  # pe_query 保持全零
                else:
                    t_q = torch.tensor([query_pos if query_pos is not None else T],
                                       device=buffer.device).float()
                    pe_query[0::2] = torch.sin(t_q * freq)
                    if half > 0:
                        pe_query[1::2] = torch.cos(t_q * freq[:half])
            else:
                pe_buffer = torch.zeros(T, D, device=buffer.device)
                pe_query = torch.zeros(D, device=buffer.device)

        if self.pre_norm:
            # 预归一化路径：Q/K 加 PE，V 不加
            x = self.layernorm_tquery(inputs) + pe_query   # B, N, D
            x_buffer_v = self.layernorm_tquery(buffer)      # B, T, N, D (V: no PE)
            x_buffer_k = x_buffer_v                         # B, T, N, D
            if pe_buffer is not None:
                x_buffer_k = x_buffer_k + pe_buffer[None, :, None, :]

            xt = torch.unsqueeze(x, dim=1)                  # B, 1, N, D
            xt = rearrange(xt, 'b t o d -> (b o) t d')     # B*N, 1, D
            xt_buffer_v = rearrange(x_buffer_v, 'b t o d -> (b o) t d')
            xt_buffer_k = rearrange(x_buffer_k, 'b t o d -> (b o) t d')
            qt = self.dense_tq(xt).view(xt.shape[0], xt.shape[1], self.num_heads, self.head_dim)
            kt = self.dense_tk(xt_buffer_k).view(xt_buffer_k.shape[0], xt_buffer_k.shape[1], self.num_heads, self.head_dim)
            vt = self.dense_tv(xt_buffer_v).view(xt_buffer_v.shape[0], xt_buffer_v.shape[1], self.num_heads, self.head_dim)
            xt, _ = self.attn(query=qt, key=kt, value=vt)  # B*N, 1, h, d
            if self.num_heads > 1:
                xt = self.dense_to(xt.reshape(B, N, self.qkv_size)).view(B, N, self.embed_dim)
            else:
                xt = xt.squeeze(-2)
            if self.dropout_rate > 0:
                xt = self.att_drop(xt)
            xt = xt + inputs
            # MLP
            y = xt
            z = self.layernorm_mlp(y)
            z = self.mlp(z)
            if self.dropout_rate > 0:
                z = self.ff_dropout(z)
            z = z + y
        else:
            # 后归一化路径：Q/K 加 PE，V 不加
            x = inputs + pe_query
            x_buffer_v = buffer                           # B, T, N, D (V: no PE)
            x_buffer_k = x_buffer_v
            if pe_buffer is not None:
                x_buffer_k = x_buffer_k + pe_buffer[None, :, None, :]
            xt = torch.unsqueeze(x, dim=1)                  # B, 1, N, D
            xt = rearrange(xt, 'b t o d -> (b o) t d')     # B*N, 1, D
            xt_buffer_v = rearrange(x_buffer_v, 'b t o d -> (b o) t d')
            xt_buffer_k = rearrange(x_buffer_k, 'b t o d -> (b o) t d')
            qt = self.dense_tq(xt).view(xt.shape[0], xt.shape[1], self.num_heads, self.head_dim)
            kt = self.dense_tk(xt_buffer_k).view(xt_buffer_k.shape[0], xt_buffer_k.shape[1], self.num_heads, self.head_dim)
            vt = self.dense_tv(xt_buffer_v).view(xt_buffer_v.shape[0], xt_buffer_v.shape[1], self.num_heads, self.head_dim)
            xt, _ = self.attn(query=qt, key=kt, value=vt)  # B*N, 1, h, d
            if self.num_heads > 1:
                xt = self.dense_to(xt.reshape(B, N, self.qkv_size)).view(B, N, self.embed_dim)
            else:
                xt = xt.squeeze(-2)
            if self.dropout_rate > 0:
                xt = self.att_drop(xt)
            xt = xt + inputs
            xt = self.layernorm_tquery(xt)
            y = xt
            z = self.mlp(y)
            if self.dropout_rate > 0:
                z = self.ff_dropout(z)
            z = z + y
            z = self.layernorm_mlp(z)
        return z


class qp_Attentions(nn.Module):
    '''
    q-p 注意力模块，组合空间注意力和时间注意力来分别求解位置 q 和动量 p。
    多层堆叠后可选地使用 MLP 输出最终 q 和 p。
    '''
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 qkv_size: int,
                 mlp_size: int,
                 pre_norm: bool = False,
                 weight_init=None,
                 num_layers: int = 1,
                 dropout_rate: float = 0.1,
                 zero_init: bool = False,
                 out_mlp=False,
                 out_hidden_layers=1
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.out_mlp = out_mlp
        self.out_hidden_layers = out_hidden_layers

        assert num_heads >= 1
        assert qkv_size % num_heads == 0, "qkv_size 必须能被 num_heads 整除"
        self.head_dim = qkv_size // num_heads

        self.space_model = nn.ModuleList()
        self.time_model = nn.ModuleList()
        for i in range(self.num_layers):
            self.space_model.append(
                SpaceAttentionBlock(embed_dim=embed_dim,
                                   num_heads=num_heads,
                                   qkv_size=qkv_size,
                                   mlp_size=mlp_size,
                                   pre_norm=pre_norm,
                                   weight_init=weight_init,
                                   dropout_rate=dropout_rate,
                                   zero_init=zero_init))
            self.time_model.append(
                TimeAttentionBlock(embed_dim=embed_dim,
                                  num_heads=num_heads,
                                  qkv_size=qkv_size,
                                  mlp_size=mlp_size,
                                  pre_norm=pre_norm,
                                  weight_init=weight_init,
                                  dropout_rate=dropout_rate))
        # 可选择在最终输出的 q 和 p 上再过 MLP
        if self.out_mlp:
            self.q_mlp = MLP(
                input_size=embed_dim,
                hidden_size=2*embed_dim,
                output_size=embed_dim,
                num_hidden_layers=self.out_hidden_layers,
                activate_output=False,
                weight_init=weight_init,
            )
            self.p_mlp = MLP(
                input_size=embed_dim,
                hidden_size=2*embed_dim,
                output_size=embed_dim,
                num_hidden_layers=self.out_hidden_layers,
                activate_output=False,
                weight_init=weight_init,
            )

    def forward(self, inputs: Array, buffer: Array,
                padding_mask: Optional[Array] = None,
                train: bool = False) -> Array:
        xs = inputs   # (B, N, D)
        xt = inputs   # (B, N, D)
        x_buffer = buffer  # (B, T, N, D)
        for i in range(self.num_layers):
            # 空间注意力求解坐标 q
            xs = self.space_model[i](xs, xs, padding_mask, train)
            # 时间注意力求解动量 p
            xt = self.time_model[i](xt, x_buffer, padding_mask, train)
        q = xs
        p = xt
        if self.out_mlp:
            q = self.q_mlp(q)
            p = self.p_mlp(p)
        return q, p


class HamiltonianNet(nn.Module):
    '''
    哈密顿量网络，输入 q, p, C，输出标量能量 H。
    将能量分解为三项：
      H = T(P, C) + sum_i MLP_V(Q_i, C_i) + sum(Transformer_I(Concat(Q_t|C_t)))
    动能采用质量矩阵形式：
      M^{-1}(C) = L(C)L(C)^T + eps*I,  L(C) = MLP_L(C) 输出下三角矩阵
      T = 0.5 * sum_i P_i^T M^{-1}(C_i) P_i
    当 P=0 时 T=0，保证 zero init。
    '''
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 qkv_size: int,
                 mlp_size: int,
                 static_dim: int = None,
                 num_layers: int = 1,
                 pre_norm: bool = False,
                 weight_init=None,
                 dropout_rate=0.,
                 ):
        super().__init__()
        if static_dim is None:
            static_dim = embed_dim
        self.static_dim = static_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.num_layers = num_layers
        self.pre_norm = pre_norm
        self.weight_init = weight_init
        self.dropout_rate = dropout_rate

        tril_size = embed_dim * (embed_dim + 1) // 2
        self.tril_mlp = MLP(
            input_size=static_dim,
            hidden_size=mlp_size // 2,
            output_size=tril_size,
            num_hidden_layers=2,
            activate_output=False,
            activation_fn=nn.SiLU,
        )
        nn.init.zeros_(self.tril_mlp.net[-1].weight)
        nn.init.zeros_(self.tril_mlp.net[-1].bias)
        self.eps = 1e-4

        self.potential_net = MLP(
            input_size=embed_dim + static_dim,
            hidden_size=mlp_size // 2,
            output_size=1,
            num_hidden_layers=2,
            activate_output=False,
            weight_init=weight_init,
            activation_fn=nn.Softplus,
        )

        self.interaction_transformer = TransformerBlock(
            embed_dim=embed_dim + static_dim,
            num_heads=num_heads,
            qkv_size=qkv_size,
            mlp_size=mlp_size,
            pre_norm=pre_norm,
            dropout_rate=0.,
            activation_fn=nn.Softplus,
        )
        self.interaction_mlp = MLP(
            input_size=embed_dim + static_dim,
            hidden_size=mlp_size // 2,
            output_size=1,
            num_hidden_layers=2,
            activate_output=False,
            activation_fn=nn.Softplus,
        )

    def forward(self, q: Array, p: Array, C: Array) -> Array:
        B, N, D = q.shape

        tril_elems = self.tril_mlp(C)
        L = torch.zeros(B, N, D, D, device=C.device, dtype=C.dtype)
        indices = torch.tril_indices(D, D, device=C.device)
        L[:, :, indices[0], indices[1]] = tril_elems
        M_inv = L @ L.transpose(-1, -2) + self.eps * torch.eye(D, device=C.device, dtype=C.dtype)
        kinetic = 0.5 * (p.unsqueeze(-2) @ M_inv @ p.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        kinetic = kinetic.sum(dim=1)

        potential_input = torch.cat([q, C], dim=-1)
        potential = self.potential_net(potential_input).squeeze(-1)
        potential = potential.sum(dim=1)

        combined = torch.cat([q, C], dim=-1)
        x = self.interaction_transformer(combined, symmetrize_attn=True, self_mask=True)
        interaction = self.interaction_mlp(x).squeeze(-1)
        interaction = interaction.sum(dim=1)

        H = kinetic + potential + interaction
        return H


class Slot_HamiltonianNet(nn.Module):
    '''
    Slot 哈密顿网络，用于单时间步内的求解。
    包含：
    - qp_Attentions：从当前 Slot 求解坐标 q 和动量 p
    - HamiltonianNet：求解哈密顿量 H
    - Integrator：积分器，用于求解下一个时间步的 q 和 p
    '''
    def __init__(self,
                 num_slots: int,
                 embed_dim: int,
                 num_heads: int,
                 qkv_size: int,
                 mlp_size: int,
                 static_dim: int = None,
                 integrator_method: str = "Leapfrog",
                 pre_norm: bool = False,
                 weight_init=None,
                 dropout_rate: float = 0.1,
                 num_layers: int = 1,
                 h_layers=1,
                 out_mlp=False,
                 out_hidden_layers=1,
                 zero_init: bool = False,
                 ):
        super().__init__()
        if static_dim is None:
            static_dim = embed_dim
        self.static_dim = static_dim
        self.dropout_rate = dropout_rate
        self.weight_init = weight_init
        self.pre_norm = pre_norm
        self.num_heads = num_heads
        self.mlp_size = mlp_size
        self.embed_dim = embed_dim
        self.qkv_size = qkv_size
        self.num_layers = num_layers
        self.H_layers = h_layers
        self.out_mlp = out_mlp
        self.out_hidden_layers = out_hidden_layers
        self.integrator_method = integrator_method

        self.qp_net = qp_Attentions(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_size=self.qkv_size,
            mlp_size=self.mlp_size,
            pre_norm=self.pre_norm,
            weight_init=self.weight_init,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            zero_init=zero_init,
            out_mlp=self.out_mlp,
            out_hidden_layers=self.out_hidden_layers
        )

        # 哈密顿量网络
        self.H_net = HamiltonianNet(
            embed_dim=self.embed_dim,
            static_dim=self.static_dim,
            num_heads=self.num_heads,
            qkv_size=self.qkv_size,
            mlp_size=self.mlp_size,
            num_layers=self.H_layers,
            pre_norm=self.pre_norm,
            weight_init=self.weight_init,
            dropout_rate=self.dropout_rate,
        )
        # 积分器（方法从配置传入）
        self.integrator = Integrator(method=self.integrator_method)
        # 第四次修改：P_t = MLP_P(CrossAtt_P(Z_t^d, history), C_t)
        self.p_mlp_C = MLP(
            input_size=embed_dim + static_dim,
            hidden_size=mlp_size // 2,
            output_size=embed_dim,
            num_hidden_layers=2,
            activate_output=False,
            activation_fn=nn.SiLU,
        )
        # 最后一层零初始化，使初始 p_mlp_C ≈ identity
        nn.init.zeros_(self.p_mlp_C.net[-1].weight)
        nn.init.zeros_(self.p_mlp_C.net[-1].bias)

        # 输入输出归一化层
        self.mlp_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

    def forward(self, slot: Array, buffer: Array, C: Optional[Array] = None,
                return_energy: bool = False,
                return_qp: bool = False):
        '''
        Args:
            ...
            return_energy: 是否返回积分前后的能量对用于 L_energy
            return_qp: 是否返回 qp_net 输出的新鲜 (q, p)
        Returns:
            q_next, p_next 或带额外信息的元组：
              return_energy=True → (…, (E_before, E_after))
              return_qp=True    → (…, (fresh_q, fresh_p))
              顺序：q_next, p_next, [energy_pair], [fresh_qp]
        '''
        q, p = self.qp_net(slot, buffer)
        # 第四次修改：P_t = MLP_P(CrossAtt_P(Z_t^d, history), C_t)
        if C is not None:
            p = self.p_mlp_C(torch.cat([p, C], dim=-1))
        q.requires_grad_(True)
        p.requires_grad_(True)

        E_before = self.H_net(q, p, C=C) if (return_energy and C is not None) else None

        with torch.enable_grad():
            q_next, p_next = self.integrator.step(q, p, self.H_net, C=C)

        if not self.pre_norm:
            q_next = self.mlp_norm(q_next)

        ret = [q_next, p_next]
        if return_energy and C is not None:
            E_after = self.H_net(q_next, p_next, C=C)
            ret.append((E_before, E_after))
        if return_qp:
            ret.append((q, p))

        if len(ret) == 2:
            return q_next, p_next
        return tuple(ret)