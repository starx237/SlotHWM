# hamiltonian.py — 哈密顿力学网络模块
# 实现基于哈密顿力学的物理建模，包含积分器、空间/时间注意力块和荷量网络

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

    def __init__(self, delta_t=0.125, method="Leapfrog"):
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
        '''应用哈密顿方程计算 dq_dt 和 dp_dt（通过自动求导）'''
        # 计算系统的哈密顿量（传入 C 用于三部分能量计算）
        energy = hnn(q=q, p=p, C=C) if C is not None else hnn(q=q, p=p)

        # dq_dt = dH/dp（位置的时间导数）
        dq_dt = torch.autograd.grad(energy,
                                    p,
                                    create_graph=True,
                                    retain_graph=True,
                                    grad_outputs=torch.ones_like(energy))[0]

        # dp_dt = -dH/dq（动量的时间导数）
        dp_dt = -torch.autograd.grad(energy,
                                     q,
                                     create_graph=True,
                                     retain_graph=True,
                                     grad_outputs=torch.ones_like(energy))[0]

        if remember_energy:
            self.energy = energy.detach().cpu().numpy()

        return dq_dt, dp_dt

    def _euler_step(self, q, p, hnn):
        '''欧拉法一步积分'''
        dq_dt, dp_dt = self._get_grads(q, p, hnn, C=self.C, remember_energy=True)
        q_next = q + self.delta_t * dq_dt
        p_next = p + self.delta_t * dp_dt
        return q_next, p_next

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
        return q_next, p_next

    def _lf_step(self, q, p, hnn):
        '''蛙跳（Leapfrog）法一步积分（二阶辛积分器）'''
        _, dp_dt = self._get_grads(q, p, hnn, C=self.C, remember_energy=True)
        p_next_half = p + dp_dt * (self.delta_t) / 2
        q_next = q + p_next_half * self.delta_t
        _, dp_next_dt = self._get_grads(q_next, p_next_half, hnn, C=self.C)
        p_next = p_next_half + dp_next_dt * (self.delta_t) / 2
        return q_next, p_next

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

        return q_4, p_3

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
                 dropout_rate: float = 0.1
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
    时间注意力块（p_AttentionBlock），用于跨时间步对 Slot 的时序建模。
    输入为当前帧和缓存的历史帧，输出作为动量（p）的更新。
    '''
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 qkv_size: int,
                 mlp_size: int,
                 pre_norm: bool = False,
                 weight_init=None,
                 dropout_rate: float = 0.1
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
        # QKV 投影层（时间注意力用）
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

    def forward(self, inputs: Array, buffer: Array,
                padding_mask: Optional[Array] = None,
                train: bool = False) -> Array:
        assert inputs.ndim == 3
        del padding_mask, train
        B, T, N, _ = buffer.shape

        if self.pre_norm:
            # 预归一化路径
            x = self.layernorm_tquery(inputs)              # B, N, D
            x_buffer = self.layernorm_tquery(buffer)        # B, T, N, D

            # 时间注意力：当前帧从历史缓存中聚合信息
            xt = torch.unsqueeze(x, dim=1)                  # B, 1, N, D
            xt = rearrange(xt, 'b t o d -> (b o) t d')     # B*N, 1, D
            xt_buffer = rearrange(x_buffer, 'b t o d -> (b o) t d')  # B*N, T, D
            qt = self.dense_tq(xt).view(xt.shape[0], xt.shape[1], self.num_heads, self.head_dim)
            kt = self.dense_tk(xt_buffer).view(xt_buffer.shape[0], xt_buffer.shape[1], self.num_heads, self.head_dim)
            vt = self.dense_tv(xt_buffer).view(xt_buffer.shape[0], xt_buffer.shape[1], self.num_heads, self.head_dim)
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
            # 后归一化路径
            x = inputs
            x_buffer = buffer
            xt = torch.unsqueeze(x, dim=1)                  # B, 1, N, D
            xt = rearrange(xt, 'b t o d -> (b o) t d')     # B*N, 1, D
            xt_buffer = rearrange(x_buffer, 'b t o d -> (b o) t d')  # B*N, T, D
            qt = self.dense_tq(xt).view(xt.shape[0], xt.shape[1], self.num_heads, self.head_dim)
            kt = self.dense_tk(xt_buffer).view(xt_buffer.shape[0], xt_buffer.shape[1], self.num_heads, self.head_dim)
            vt = self.dense_tv(xt_buffer).view(xt_buffer.shape[0], xt_buffer.shape[1], self.num_heads, self.head_dim)
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
                 # 最终输出 MLP 配置
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

        # 构建多层空间注意力和时间注意力模块
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
                                   dropout_rate=dropout_rate))
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
                activate_output=True,
                weight_init=weight_init,
            )
            self.p_mlp = MLP(
                input_size=embed_dim,
                hidden_size=2*embed_dim,
                output_size=embed_dim,
                num_hidden_layers=self.out_hidden_layers,
                activate_output=True,
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
      H = sum_i MLP_K(P_i, C) + sum_i MLP_V(Q_i, C) + sum(Linear_I(SelfAtt_I(Q_t; C)))
    其中第一项为动能，第二项为势能，第三项为物体间交互能（自掩码，不计算自交互）。
    '''
    def __init__(self,
                 embed_dim: int,  # P, Q, C 的维度（即 slot_dim // 2）
                 num_heads: int,
                 qkv_size: int,   # attention 中 QKV 的维度
                 mlp_size: int,   # MLP 隐藏层维度
                 num_layers: int = 1,
                 pre_norm: bool = False,
                 weight_init=None,
                 dropout_rate=0.,
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.num_layers = num_layers
        self.pre_norm = pre_norm
        self.weight_init = weight_init
        self.dropout_rate = dropout_rate

        # 动能网络 MLP_K(P_i, C) — 输入为 [p, C] 的拼接 (2*embed_dim)
        self.kinetic_net = MLP(
            input_size=embed_dim * 2,
            hidden_size=mlp_size,
            output_size=1,
            num_hidden_layers=1,
            activate_output=True,
            weight_init=weight_init,
        )

        # 势能网络 MLP_V(Q_i, C) — 输入为 [q, C] 的拼接 (2*embed_dim)
        self.potential_net = MLP(
            input_size=embed_dim * 2,
            hidden_size=mlp_size,
            output_size=1,
            num_hidden_layers=1,
            activate_output=True,
            weight_init=weight_init,
        )

        # 交互能网络：自注意力 SelfAtt_I(Q_t; C) + Linear_I
        # 注意：C 作为 Key/Value 的上下文，Q 作为 Query
        self.interact_attn = GeneralizedDotProductAttention()
        self.dense_iq = nn.Linear(embed_dim, qkv_size)   # Q 投影
        self.dense_ik = nn.Linear(embed_dim, qkv_size)   # C 投影为 Key
        self.dense_iv = nn.Linear(embed_dim, qkv_size)   # C 投影为 Value
        self.dense_io = nn.Linear(qkv_size, 1)           # Linear_I: N -> 1

    def forward(self, q: Array, p: Array, C: Array) -> Array:
        '''
        Args:
            q: 广义坐标 (B, N, D)
            p: 广义动量 (B, N, D)
            C: 静态特征 (B, N, D)
        Returns:
            H: 总能量标量 (B,)
        '''
        # 动能项：对每个物体 i，计算 MLP_K(P_i, C)，然后求和
        kinetic_input = torch.cat([p, C], dim=-1)  # (B, N, 2*D)
        kinetic = self.kinetic_net(kinetic_input).squeeze(-1)  # (B, N)
        kinetic = kinetic.sum(dim=1)  # (B,)

        # 势能项：对每个物体 i，计算 MLP_V(Q_i, C)，然后求和
        potential_input = torch.cat([q, C], dim=-1)  # (B, N, 2*D)
        potential = self.potential_net(potential_input).squeeze(-1)  # (B, N)
        potential = potential.sum(dim=1)  # (B,)

        # 交互能项：自注意力 SelfAtt_I(Q_t; C) — 自掩码，不计算自己和自己的交互能
        B, N, D = q.shape
        # 将 Q 和 C 投影到注意力空间
        q_proj = self.dense_iq(q)  # (B, N, Q)
        k_proj = self.dense_ik(C)  # (B, N, Q)
        v_proj = self.dense_iv(C)  # (B, N, Q)

        # 计算原始注意力分数（未经过 softmax）
        attn_logits = torch.einsum("bqd,bkd->bqk", q_proj, k_proj)  # (B, N, N)
        attn_logits = attn_logits * (q_proj.shape[-1] ** -0.5)

        # 自掩码：在对角线位置设为 -inf，使得 softmax 后为 0（不计算自己与自己的交互）
        diag_mask = torch.eye(N, device=q.device, dtype=torch.bool)
        attn_logits = attn_logits.masked_fill(diag_mask, float('-inf'))

        # Softmax 得到注意力权重（此时 diag 位置为 0）
        attn_weights = F.softmax(attn_logits, dim=-1)

        # 加权的值求和，得到每个物体的交互贡献 SI_t
        interaction = torch.einsum("bqk,bkd->bqd", attn_weights, v_proj)  # (B, N, Q)
        interaction = self.dense_io(interaction).squeeze(-1)  # (B, N)
        # 按 idea.md §1："对于一个物体对，两个物体分别贡献一半交互能"
        # 然后整体的 I_t = Sum(SI_t)
        interaction = interaction.sum(dim=1)  # (B,)

        # 总能量
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
                 pre_norm: bool = False,
                 weight_init=None,
                 dropout_rate: float = 0.1,
                 # q、p 网络配置
                 num_layers: int = 1,
                 # H 网络配置
                 h_layers=1,
                 # 最终 MLP 配置
                 out_mlp=False,
                 out_hidden_layers=1
                 ):
        super().__init__()

        # 通用参数
        self.dropout_rate = dropout_rate
        self.weight_init = weight_init
        self.pre_norm = pre_norm
        self.num_heads = num_heads
        self.mlp_size = mlp_size

        # q、p_Attentions 参数
        self.embed_dim = embed_dim
        self.qkv_size = qkv_size
        self.num_layers = num_layers

        # H_Attention 参数
        self.H_layers = h_layers

        self.out_mlp = out_mlp
        self.out_hidden_layers = out_hidden_layers

        # q-p 网络：从 Slot 中提取位置和动量信息
        self.qp_net = qp_Attentions(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_size=self.qkv_size,
            mlp_size=self.mlp_size,
            pre_norm=self.pre_norm,
            weight_init=self.weight_init,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            out_mlp=self.out_mlp,
            out_hidden_layers=self.out_hidden_layers
        )

        # 哈密顿量网络：输入为 q, p, C，每个维度为 embed_dim
        self.H_net = HamiltonianNet(
            embed_dim=self.embed_dim,  # q, p, C 各自维度
            num_heads=self.num_heads,
            qkv_size=self.qkv_size,
            mlp_size=self.mlp_size,
            num_layers=self.H_layers,
            pre_norm=self.pre_norm,
            weight_init=self.weight_init,
            dropout_rate=self.dropout_rate,
        )
        # 积分器
        self.integrator = Integrator()
        # 输入输出归一化层
        self.mlp_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

    def forward(self, slot: Array, buffer: Array, C: Optional[Array] = None):
        '''
        Args:
            slot: 当前时间步的 Slot, shape (B, N, D)
            buffer: 历史帧缓存, shape (B, T, N, D)
            C: 静态特征上下文, shape (B, N, D)，由 SlotPiModel 计算并传入
        Returns:
            q_next: 下一时间步的 Slot 位置, shape (B, N, D)
            p_next: 下一时间步的 Slot 动量, shape (B, N, D)
        '''
        # 可选预归一化
        if self.pre_norm:
            slot = self.mlp_norm(slot)
            buffer = self.mlp_norm(buffer)

        # 从当前 Slot 中求解 q 和 p
        q, p = self.qp_net(slot, buffer)
        q.requires_grad_(True)
        p.requires_grad_(True)

        # 使用积分器进行一步积分（需要启用梯度以计算哈密顿量的导数）
        with torch.enable_grad():
            q_next, p_next = self.integrator.step(q, p, self.H_net, C=C)

        if not self.pre_norm:
            # 如果没有做预归一化，则在输出端对 q 做归一化
            q_next = self.mlp_norm(q_next)

        return q_next, p_next