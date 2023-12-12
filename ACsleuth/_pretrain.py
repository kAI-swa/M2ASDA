import os
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import anndata as ad
from typing import Literal, Union, List

from Net import SCNetAE
from _utils import seed_everything


def Pretrain_SC(train: ad.AnnData,
                n_epochs: int = 50,
                batch_size: int = 32,
                learning_rate: float = 0.0005,
                GPU: bool = True,
                verbose: bool = True,
                log_interval: int = 10,
                random_state: int = None,
                norm_type: Literal['Batch', 'Instance'] = 'Batch'):
    if GPU:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("GPU isn't available, and use CPU to train NovelGAN.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    if random_state is not None:
        seed_everything(random_state)

    # Initialize dataloader for train data
    train_data = torch.as_tensor(torch.from_numpy(train.X), dtype=torch.float32)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True)
    net = SCNetAE(train.n_vars, norm_type=norm_type).to(device)
    opt_G = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    G_scaler = torch.cuda.amp.GradScaler()
    L1 = nn.L1Loss().to(device)

    net.train()
    for epoch in range(n_epochs):
        for idx, data in enumerate(train_loader):
            data = data.to(device)

            re_data = nn.functional.relu(net(data))
            Loss = L1(data, re_data)
            opt_G.zero_grad()
            G_scaler.scale(Loss).backward()
            G_scaler.step(opt_G)
            G_scaler.update()

        if verbose and ((epoch+1) % log_interval == 0):
            print('Pretrain Epoch: [{:^3}/{:^3} ({:^3.0f}%)]\t\tLoss: {:.6f}'.format(
                   epoch+1, n_epochs, 100.*(epoch+1)/n_epochs, Loss.item()))

    path = './pretrain_weight/'
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(net.state_dict(), os.path.join(path, f'SCNetAE_{norm_type}.pth'))
