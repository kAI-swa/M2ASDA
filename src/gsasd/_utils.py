import dgl
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import random
import numpy as np


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    dgl.random.seed(seed)


def hard_shrink_relu(x, lambd=0, epsilon=1e-12):
    '''
    relu based hard shrinkage function, only works for positive values
    '''
    x = (F.relu(x-lambd) * x) / (torch.abs(x - lambd) + epsilon)
    return x


def calculate_gradient_penalty(real_data, fake_data, D):
    shapes = [1 if i != 0 else real_data.size(i) for i in range(real_data.dim())]
    eta = torch.FloatTensor(*shapes).uniform_(0, 1)
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        eta = eta.cuda()
    else:
        eta = eta

    interpolated = eta * real_data + ((1 - eta) * fake_data)

    if cuda:
        interpolated = interpolated.cuda()
    else:
        interpolated = interpolated

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(prob_interpolated.size()).cuda()
                              if cuda else torch.ones(prob_interpolated.size()),
                              create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty