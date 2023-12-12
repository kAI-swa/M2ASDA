import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._unit import MemoryUnit
from ._net import SCNetAE


class Memory_G(nn.Module):
    def __init__(self, in_dim, out_dim=[1024, 512, 256], mem_dim=2048, thres=0.005,
                 temperature=0.5):
        super().__init__()
        self.net = SCNetAE(in_dim, out_dim, 'Batch')
        self.Memory = MemoryUnit(mem_dim, out_dim[2], thres, temperature)

    def forward(self, x):
        real_z = self.net.encode(x)
        mem_z = self.Memory(real_z)
        fake_x = F.relu(self.net.decode(mem_z))
        fake_z = self.net.encode(fake_x)
        return real_z, fake_x, fake_z