# misc.py — 通用工具模块
# 包含恒等映射、线性读出层、多层感知机、GRU、全连接层和位置编码等基础组件

import torch
import torch.nn as nn
from typing import Optional, Mapping, Any


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
                 weight_init=None, activation_fn=nn.ReLU):
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
        '''Xavier 均匀初始化权重，偏置置零'''
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
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