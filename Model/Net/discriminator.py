import torch.nn as nn
import torch.nn.utils.spectral_norm as SNorm1d


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act: bool = True):
        super().__init__()
        self.linear = nn.Sequential(
            SNorm1d(nn.Linear(in_dim, out_dim)),
            nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.linear(x)


class Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim=[1024, 512, 1]):
        super().__init__()
        self.model = nn.Sequential(
            LinearBlock(in_dim, out_dim[0]),
            LinearBlock(out_dim[0], out_dim[1]),
            LinearBlock(out_dim[1], out_dim[2], act=False),
        )

    def forward(self, x):
        x = self.model(x)
        return x
