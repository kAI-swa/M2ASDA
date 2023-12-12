import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from Net import Memory_G, Discriminator
from _pretrain import Pretrain_SC
from _utils import seed_everything, calculate_gradient_penalty
from typing import Union


class Detect:
    def __init__(self, n_epochs, learning_rate, sample_rate, mem_dim, update_size,
                 shrink_thres, temperature, n_critic, pretrain, GPU, verbose,
                 log_interval, random_state, weight):
        if GPU:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                print("GPU isn't available, and use CPU to train ODBC-GAN.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        if random_state is not None:
            seed_everything(random_state)

        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.sample_rate = sample_rate
        self.mem_dim = mem_dim
        self.update_size = update_size
        self.shrink_thres = shrink_thres
        self.tem = temperature
        self.n_critic = n_critic
        self.pretrain = pretrain
        self.verbose = verbose
        self.log_interval = log_interval
        self.random_state = random_state
        if weight is None:
            self.weight = {'w_rec': 50, 'w_adv': 1, 'w_enc': 1, 'w_gp': 10}
        else:
            self.weight = weight

    def sampling(self, adata):
        idx = adata.obs.index.tolist()
        idx = random.sample(idx, int(len(idx)*self.sample_rate))
        return adata[idx]

    def updating(self, z_all):
        n_obs = z_all.shape[0]
        idx = random.sample([i for i in range(n_obs)], self.update_size)
        return z_all[idx, :]

    def log_print(self, epoch, G_loss, D_loss):
        if (epoch+1) % self.log_interval == 0:
            txt = 'Train Epoch: [{:^4}/{:^4}({:^3.0f}%)]    G_loss: {:.6f}    D_loss: {:.6f}'
            txt = txt.format(epoch+1, self.n_epochs, 100.*(epoch+1)/self.n_epochs,
                             G_loss.item(), D_loss.item())
            print(txt)


class Detect_SC(Detect):
    def __init__(self, n_epochs, learning_rate, sample_rate, mem_dim, update_size,
                 shrink_thres, temperature, n_critic, pretrain, GPU, verbose,
                 log_interval, random_state, weight):
        super().__init__(n_epochs, learning_rate, sample_rate, mem_dim, update_size,
                         shrink_thres, temperature, n_critic, pretrain, GPU, verbose,
                         log_interval, random_state, weight)

    def fit(self, train: Union[ad.AnnData, str]):
        if isinstance(train, str):
            if os.path.exists(train):
                train = sc.read_h5ad(train)
                train = train[train.obs["cell.type"] != "B cells"]  # temp
            else:
                raise FileNotFoundError("File not found Error")
        else:
            train = train

        if self.verbose:
            print('Begin to learn information of normal cells with ODBC-GAN...')

        if self.sample_rate < 1:
            train = self.sampleing(train)


        self.genes = train.var_names

        self.D = Discriminator(in_dim=train.n_vars).to(self.device)
        self.G = Memory_G(in_dim=train.n_vars, thres=self.shrink_thres,
                          mem_dim=self.mem_dim, temperature=self.tem).to(self.device)

        self.opt_D = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        if torch.cuda.is_available():
            self.D_scaler = torch.cuda.amp.GradScaler()
            self.G_scaler = torch.cuda.amp.GradScaler()
        else:
            self.D_scaler = torch.cuda.amp.GradScaler(enabled=False)
            self.G_scaler = torch.cuda.amp.GradScaler(enabled=False)

        self.L1 = nn.L1Loss().to(self.device)
        self.L2 = nn.MSELoss().to(self.device)

        if self.pretrain:
            self.train = train
            self.load_weight()

        self.train = torch.Tensor(train.X).to(self.device)
        # prepare the Memory Bank
        self.prepare()

        self.D.train()
        self.G.train()
        for epoch in range(self.n_epochs):
            for i in range(self.n_critic):
                self.Update_D()
            self.Update_G()
            if self.verbose:
                self.log_print(epoch, self.G_loss, self.D_loss)

        if self.verbose:
            print('Information of normal cells have been learned.\n')

    @torch.no_grad()
    def predict(self, test: Union[ad.AnnData, str]):
        if isinstance(test, str):
            if os.path.exists(test):
                test = sc.read_h5ad(test)
            else:
                raise FileNotFoundError("File not found Error")
        else:
            test = test

        if self.verbose:
            print('Begin to detect outlier cell types with ODBC-GAN...')

        if (test.var_names != self.genes).any():
            raise RuntimeError('Test data and train data have different genes.')

        if (self.G is None or self.D is None):
            raise RuntimeError('Please run Detect_SC.fit first.')

        self.G.eval()
        self.test = torch.Tensor(test.X).to(self.device)
        real_z, fake_x, fake_z = self.G(self.test)
        self.z_x = real_z
        self.res_x = self.test - fake_x
        diff = 1 - F.cosine_similarity(real_z, fake_z).reshape(-1, 1)
        diff = diff.cpu().numpy()
        result = pd.DataFrame({'cell_idx': test.obs_names, 'score': diff.reshape(-1)})

        if self.verbose:
            print('Outlier cell types have been detected.\n')
        return result

    def load_weight(self):
        path = './pretrain_weight/SCNetAE_Batch.pth'
        if not os.path.exists(path):
            Pretrain_SC(self.train)

        # load the pre-trained weights for Encoder and Decoder
        pre_weights = torch.load(path)
        self.G.net.load_state_dict({k: v for k, v in pre_weights.items()})

    @torch.no_grad()
    def prepare(self):
        self.G.eval()
        sum_t = self.mem_dim/self.update_size
        t = 0
        while t < sum_t:
            real_z, _, _ = self.G(self.train)
            self.G.Memory.update_mem(self.updating(real_z))
            t += 1
            if t >= sum_t:
                break

    def Update_G(self):
        real_z, fake_data, fake_z = self.G(self.train)
        fake_d = self.D(fake_data)

        Loss_enc = self.L2(real_z, fake_z)
        Loss_rec = self.L1(self.train, fake_data)
        Loss_adv = -torch.mean(fake_d)

        self.G_loss = (self.weight['w_enc']*Loss_enc +
                       self.weight['w_rec']*Loss_rec +
                       self.weight['w_adv']*Loss_adv)

        self.opt_G.zero_grad()
        self.G_scaler.scale(self.G_loss).backward(retain_graph=True)
        self.G_scaler.step(self.opt_G)
        self.G_scaler.update()

        self.G.Memory.update_mem(self.updating(fake_z))

    def Update_D(self):
        self.opt_D.zero_grad()
        _, fake_data, _ = self.G(self.train)

        real_d = self.D(self.train)
        self.D_scaler.scale(-torch.mean(real_d)).backward()

        fake_d = self.D(fake_data.detach())
        self.D_scaler.scale(torch.mean(fake_d)).backward()

        # Compute W-div gradient penalty
        gp = calculate_gradient_penalty(self.train, fake_data, self.D)
        self.D_scaler.scale(gp*self.weight['w_gp']).backward()

        self.D_loss = -torch.mean(real_d) + torch.mean(fake_d) + gp*self.weight['w_gp']

        self.D_scaler.step(self.opt_D)
        self.D_scaler.update()
