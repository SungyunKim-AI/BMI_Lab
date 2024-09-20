import torch
import torch.nn as nn
import itertools as it
from copy import deepcopy as dcopy

class ConvBlock(nn.Module):
    def __init__(self, in_channel=1, out_channel=64, kernel_size=3, stride=1):
        super().__init__()
        layers=[
        nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=1),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ResBlock(nn.Module):
    def __init__(self, in_channel, kernel_size=5):
        super().__init__()
        layers=[
        nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size, stride=1, padding='same'),
        nn.BatchNorm1d(in_channel),
        nn.ReLU(),
        nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size, stride=1, padding='same'),
        nn.BatchNorm1d(in_channel),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def residual_blocks(in_channel, n_repeat):
    return nn.Sequential(*[ResBlock(in_channel) for _ in range(n_repeat)])

class Residual(nn.Module):
    def __init__(self, blocks, activation=None):
        super().__init__()
        if issubclass(type(blocks), nn.Module):
            self.blocks = blocks
        elif issubclass(type(blocks), list):
            self.blocks = nn.ModuleList(blocks)
        else:
            raise Exception('')
        self.activation = activation

    def forward(self, x):
        if self.activation is not None:
            for block in self.blocks:
                x = self.activation(block(x)+x)

        else: # No activation
            for block in self.blocks:
                x = block(x)+x
        return x


class FNN(nn.Module):
    def __init__(self, n_hidden_list, activation_list, n_input=None):
    # def __init__(self, n_input, n_hidden_list, activation_list):
        super().__init__()
        if type(activation_list) is not list:
            activation_list = [dcopy(activation_list)]*len(n_hidden_list)
        assert len(activation_list)==len(n_hidden_list), 'length of layers and activations must match. If you want no activation, use nn.Identity'

        # 1st layer - Select Lazy if n_input is not specified
        if n_input is None:
            layers = [nn.Flatten(1), nn.LazyLinear(n_hidden_list[0]), activation_list[0]]
        else:
            layers = [nn.Flatten(1), nn.Linear(n_input, n_hidden_list[0]), activation_list[0]]
        # Hidden layers ~ Output layer
        for i in range(len(n_hidden_list) - 1):
            layers.extend([nn.Linear(n_hidden_list[i], n_hidden_list[i+1]), activation_list[i+1]])

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        '''x.shape()==(batch_size, feature_dim)'''
        return self.fc(x)
    

class nEMGNet(nn.Module):
    def __init__(self, n_classes, n_channel_list = [64, 128, 256, 512, 1024], n_repeat=2):
        super().__init__()
        _n_channel_list = [1] + n_channel_list

        cnn = nn.Sequential(
            ConvBlock(in_channel=_n_channel_list[0], out_channel=_n_channel_list[1], kernel_size=11, stride=2),
            ConvBlock(in_channel=_n_channel_list[1], out_channel=_n_channel_list[1], kernel_size=7, stride=2),
            ConvBlock(in_channel=_n_channel_list[1], out_channel=_n_channel_list[1], kernel_size=5, stride=2),
            Residual(blocks=residual_blocks(_n_channel_list[1], n_repeat)),
            *it.chain(*[
                [ConvBlock(in_channel=c1, out_channel=c2, kernel_size=3), Residual(blocks=residual_blocks(c2, n_repeat))]
                for c1, c2 in zip(_n_channel_list[1:-1], _n_channel_list[2:])
                ])
            )

        layers = [
            cnn,
            nn.Flatten(1),
            FNN(n_hidden_list=[512,256,64,16,n_classes], activation_list=[nn.LeakyReLU()]*4+[nn.Identity()])
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
