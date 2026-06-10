import torch
import torch.nn as nn


class Identity(nn.Module):
    def forward(self, x):
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=1,
                 activate_output=False, weight_init=None, activation_fn=nn.ReLU):
        super().__init__()
        layers = []
        prev = input_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev, hidden_size))
            layers.append(activation_fn())
            prev = hidden_size
        layers.append(nn.Linear(prev, output_size))
        if activate_output:
            layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x):
        return self.net(x)
