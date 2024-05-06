import torch
from torch import nn
import math
from torch.nn import functional as F


def hard_shrink_relu(x, lambd=0, epsilon=1e-12):
    '''
    relu based hard shrinkage function, only works for positive values
    '''
    x = (F.relu(x-lambd) * x) / (torch.abs(x - lambd) + epsilon)
    return x


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, z_dim, shrink_thres=0.005, tem=0.5):
        super().__init__()
        self.mem_dim = mem_dim
        self.z_dim = z_dim
        self.shrink_thres = shrink_thres
        self.tem = tem
        self.register_buffer("mem", torch.randn(self.mem_dim, self.z_dim))
        self.register_buffer("mem_ptr", torch.zeros(1, dtype=torch.long))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mem.size(1))
        self.mem.data.uniform_(-stdv, stdv)

    @torch.no_grad()
    def update_mem(self, z):
        batch_size = z.shape[0]  # z, B x C
        ptr = int(self.mem_ptr)
        assert self.mem_dim % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.mem[ptr:ptr + batch_size, :] = z  # mem, M x C
        ptr = (ptr + batch_size) % self.mem_dim  # move pointer

        self.mem_ptr[0] = ptr

    def attention(self, input):
        att_weight = torch.mm(input, self.mem.T)  # input x mem^T, (BxC) x (CxM) = B x M
        att_weight = F.softmax(att_weight/self.tem, dim=1)  # B x M

        # ReLU based shrinkage, hard shrinkage for positive value
        if (self.shrink_thres > 0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        output = torch.mm(att_weight, self.mem)  # AttWeight x mem, (BxM) x (MxC) = B x C
        return output

    def forward(self, x):
        x = self.attention(x)
        return x
    

class StyleUnit(nn.Module):
    def __init__(self, data_n: int, z_dim: int):
        super().__init__()
        self.n = data_n
        self.style = nn.Parameter(torch.Tensor(data_n, z_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.style.size(1))
        self.style.data.uniform_(-stdv, stdv)

    def forward(self, z, label):
        if self.n == 1:
            return z
        else:
            s = torch.mm(label, self.style)
            return s + z