import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._unit import MemoryUnit
from ._net import SCNetAE
from ._unit import StyleUnit


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
    
class Align_G(nn.Module):
    def __init__(self, base_cells, input_cells, in_dim, out_dim=[1024, 512, 256]):
        super().__init__()
        self.net = SCNetAE(in_dim, out_dim, 'Batch')
        self.mapping = nn.Parameter(torch.Tensor(base_cells, input_cells))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mapping.size(1))
        self.mapping.data.uniform_(-stdv, stdv)

    def forward(self, x, base):
        z = self.net.encode(x)
        z = F.normalize(z, p=1, dim=1)
        fake_z = torch.mm(F.relu(self.mapping), z)
        z = self.net.encode(base)
        z = F.normalize(z, p=1, dim=1)
        return fake_z, z, F.relu(self.mapping)


class Batch_G(nn.Module):
    def __init__(self, data_n, in_dim, out_dim=[1024, 512, 256]):
        super().__init__()
        self.net = SCNetAE(in_dim, out_dim, 'Instance')
        self.Style = StyleUnit(data_n, out_dim[2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, label):
        z = self.net.encode(x)
        z = self.Style(z, label)
        y = self.net.decode(z)
        return F.relu(x + y)