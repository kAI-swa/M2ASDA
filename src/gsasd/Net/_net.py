import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm: bool = True, act: bool = True,
                 use_dropout=False, norm_type=Literal['Batch', 'Instance']):
        super().__init__()
        if norm_type == 'Batch':
            self.linear = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim) if norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity(),
            )
        elif norm_type == 'Instance':
            self.linear = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.InstanceNorm1d(out_dim) if norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity(),
            )

    def forward(self, x):
        x = self.linear(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim=[1024, 512, 256], norm_type='Batch'):
        super().__init__()
        self.down = nn.Sequential(
            LinearBlock(in_dim, out_dim[0], norm_type=norm_type),
            LinearBlock(out_dim[0], out_dim[1], norm_type=norm_type),
            LinearBlock(out_dim[1], out_dim[2], act=False, norm_type=norm_type),
        )
        self.bottleneck = LinearBlock(out_dim[2], out_dim[2], norm_type=norm_type)

    def forward(self, x):
        x = self.down(x)
        return self.bottleneck(x)


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim=[1024, 512, 256], norm_type='Batch'):
        super().__init__()
        self.up = nn.Sequential(
            LinearBlock(out_dim[2], out_dim[1], norm_type=norm_type),
            LinearBlock(out_dim[1], out_dim[0], norm_type=norm_type),
            LinearBlock(out_dim[0], in_dim, norm=False, act=False, norm_type=norm_type),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class SCNetAE(nn.Module):
    def __init__(self, in_dim, out_dim=[1024, 512, 256], norm_type='Batch'):
        super().__init__()
        self.encoder = Encoder(in_dim, out_dim, norm_type)
        self.decoder = Decoder(in_dim, out_dim, norm_type)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))
