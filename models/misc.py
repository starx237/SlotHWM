# misc.py — 通用工具模块
# 包含恒等映射、线性读出层、多层感知机、GRU、全连接层和位置编码等基础组件

import torch
import torch.nn as nn
from typing import Optional, Mapping, Any


def create_coordinate_grid(h, w, device):
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing='ij',
    )
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid


class Identity(nn.Module):
    '''恒等映射层，不对输入做任何变换，直接返回输入值'''
    def forward(self, x):
        return x


class Readout(nn.Module):
    '''线性读出层，将输入特征通过全连接层映射到目标维度'''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, targets=None):
        return self.linear(x)


class MLP(nn.Module):
    '''多层感知机，支持指定隐藏层数量和可选的输出激活函数（Softplus）'''
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=1, activate_output=False,
                 weight_init=None, activation_fn=nn.SiLU):
        super().__init__()
        layers = []
        prev_size = input_size
        # 逐层构建隐藏层
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_fn())
            prev_size = hidden_size
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        if activate_output:
            layers.append(nn.Softplus())  # 使用 Softplus 确保输出为正
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)  # 使用 Xavier 均匀初始化

    def _init_weights(self, module):
        '''Kaiming 均匀初始化权重（适配 SiLU/ReLU），偏置置零'''
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x):
        return self.net(x)


class GRU(nn.Module):
    '''GRU 循环神经网络模块，封装了 PyTorch 的 nn.GRU'''
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, hidden=None):
        return self.gru(x, hidden)


class Dense(nn.Linear):
    '''全连接层的别名，直接继承 nn.Linear'''
    pass


class GRLFunction(torch.autograd.Function):
    '''梯度逆转层自定义算子。前向恒等映射，反向将梯度乘以 -gamma。'''
    @staticmethod
    def forward(ctx, x, gamma):
        ctx.gamma = gamma
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.gamma * grad_output, None


class GradientReversal(nn.Module):
    '''梯度逆转层封装，用于对抗性解耦（idea.md §4 GRL）。
       前向恒等，反向将梯度取反乘 gamma，迫使 S^d 洗去 C 的信息。'''
    def __init__(self):
        super().__init__()
        self.gamma = 0.0

    def set_gamma(self, g):
        self.gamma = g

    def forward(self, x):
        return GRLFunction.apply(x, self.gamma)


class AffineCoupling(nn.Module):
    '''
    交替仿射耦合层（可逆变换），用于实现 f_θ_Z。
    4 层交替，按 static_dim / dynamic_dim 拆分而不是对半。
    层 1,3：固定 S^c（静态部分），变换 S^d（动态部分）
    层 2,4：固定 S^d（动态部分），变换 S^c（静态部分）
    可作为替换元件使用。
    '''
    def __init__(self, static_dim: int, dynamic_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        h = hidden_dim // 2  # s/t MLPs 使用一半 hidden dim

        # 层 1,3：固定 x1（静态），变换 x2（动态）
        self.net_s1 = MLP(static_dim, h, dynamic_dim, num_hidden_layers=1)
        self.net_t1 = MLP(static_dim, h, dynamic_dim, num_hidden_layers=1)
        self.net_s3 = MLP(static_dim, h, dynamic_dim, num_hidden_layers=1)
        self.net_t3 = MLP(static_dim, h, dynamic_dim, num_hidden_layers=1)
        # 层 2,4：固定 x2（动态），变换 x1（静态）
        self.net_s2 = MLP(dynamic_dim, h, static_dim, num_hidden_layers=1)
        self.net_t2 = MLP(dynamic_dim, h, static_dim, num_hidden_layers=1)
        self.net_s4 = MLP(dynamic_dim, h, static_dim, num_hidden_layers=1)
        self.net_t4 = MLP(dynamic_dim, h, static_dim, num_hidden_layers=1)
        # 输出层 zero init，使初始 s≈0, t≈0 → f_z ≈ identity
        for net in [self.net_s1, self.net_t1, self.net_s3, self.net_t3,
                    self.net_s2, self.net_t2, self.net_s4, self.net_t4]:
            nn.init.zeros_(net.net[-1].weight)
            nn.init.zeros_(net.net[-1].bias)

    @staticmethod
    def _bounded_s(s):
        return torch.tanh(s)

    def _layer1_fwd(self, x):
        x1, x2 = x.split([self.static_dim, self.dynamic_dim], dim=-1)
        s = self._bounded_s(self.net_s1(x1))
        t = self.net_t1(x1)
        return torch.cat([x1, x2 * torch.exp(s) + t], dim=-1)

    def _layer1_inv(self, y):
        y1, y2 = y.split([self.static_dim, self.dynamic_dim], dim=-1)
        s = self._bounded_s(self.net_s1(y1))
        t = self.net_t1(y1)
        return torch.cat([y1, (y2 - t) * torch.exp(-s)], dim=-1)

    def _layer2_fwd(self, x):
        x1, x2 = x.split([self.static_dim, self.dynamic_dim], dim=-1)
        s = self._bounded_s(self.net_s2(x2))
        t = self.net_t2(x2)
        return torch.cat([x1 * torch.exp(s) + t, x2], dim=-1)

    def _layer2_inv(self, y):
        y1, y2 = y.split([self.static_dim, self.dynamic_dim], dim=-1)
        s = self._bounded_s(self.net_s2(y2))
        t = self.net_t2(y2)
        return torch.cat([(y1 - t) * torch.exp(-s), y2], dim=-1)

    def _layer3_fwd(self, x):
        return self._layer1_fwd(x)

    def _layer3_inv(self, y):
        return self._layer1_inv(y)

    def _layer4_fwd(self, x):
        return self._layer2_fwd(x)

    def _layer4_inv(self, y):
        return self._layer2_inv(y)

    def forward(self, x):
        x = self._layer1_fwd(x)
        x = self._layer2_fwd(x)
        x = self._layer3_fwd(x)
        x = self._layer4_fwd(x)
        return x

    def inverse(self, y):
        y = self._layer4_inv(y)
        y = self._layer3_inv(y)
        y = self._layer2_inv(y)
        y = self._layer1_inv(y)
        return y


class PositionEmbedding(nn.Module):
    '''位置编码层，支持线性编码和傅里叶（正弦/余弦）编码两种方式'''
    def __init__(self, embedding_type, num_frequencies, embedding_dim):
        super().__init__()
        self.embedding_type = embedding_type
        self.num_frequencies = num_frequencies
        self.embedding_dim = embedding_dim
        if embedding_type == "linear":
            # 线性编码：使用单层线性映射
            self.embed = nn.Linear(1, embedding_dim)
        elif embedding_type == "fourier":
            # 傅里叶编码：使用正弦/余弦基函数，频率为可学习参数（梯度关闭）
            self.freq_bands = nn.Parameter(torch.randn(num_frequencies, embedding_dim // 2) * 2.0, requires_grad=False)

    def forward(self, x):
        if self.embedding_type == "linear":
            # 线性编码：扩展维度后通过全连接层
            return self.embed(x.unsqueeze(-1))
        elif self.embedding_type == "fourier":
            # 傅里叶编码：计算 x * freq * 2*pi 的正弦和余弦值，拼接后返回
            x_proj = x.unsqueeze(-1) * self.freq_bands[None, :, :] * 2.0 * 3.141592653589793
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        raise NotImplementedError(f"未知嵌入类型: {self.embedding_type}")