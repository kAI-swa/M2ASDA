import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import numpy as np
import pandas as pd
import anndata as ad
from typing import List, Union, Optional

from .Net import Batch_G, Discriminator
from .align import Align_SC
from ._pretrain import Pretrain_SC
from ._utils import seed_everything, calculate_gradient_penalty


class Correct:
    def __init__(self, n_epochs, learning_rate, sample_rate, n_critic,
                 pretrain, GPU, verbose, log_interval, weight, random_state):
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
        self.n_critic = n_critic
        self.pretrain = pretrain
        self.verbose = verbose
        self.log_interval = log_interval
        self.random_state = random_state
        if weight is None:
            self.weight = {'w_rec': 50, 'w_adv': 1, 'w_gp': 10}
        else:
            self.weight = weight

    def sample_single(self, adata):
        idx = adata.obs.index.tolist()
        idx = random.sample(idx, int(len(idx)*self.sample_rate))
        return adata[idx]

    def sampleing(self, input, base):
        if isinstance(input, List):
            input_sample = []
            for i in range(len(input)):
                adata = input[i]
                adata = self.sample_single(adata)
                input_sample.append(adata)
        else:
            input_sample = self.sample_single(input)
        base_sample = self.sample_single(base)
        return input_sample, base_sample

    def log_print(self, epoch, G_loss, D_loss):
        if (epoch+1) % self.log_interval == 0:
            txt = 'Train Epoch: [{:^4}/{:^4}({:^3.0f}%)]    G_loss: {:.6f}    D_loss: {:.6f}'
            txt = txt.format(epoch+1, self.n_epochs, 100.*(epoch+1)/self.n_epochs,
                             G_loss.item(), D_loss.item())
            print(txt)


class Correct_SC(Correct):
    def __init__(self, n_epochs, learning_rate, sample_rate, n_critic, pretrain, GPU,
                 verbose, log_interval, weight, random_state, fast: bool = False,
                 include: bool = True):
        super().__init__(n_epochs, learning_rate, sample_rate, n_critic, pretrain,
                         GPU, verbose, log_interval, weight, random_state)
        self.fast = fast
        self.include = include

    def fit(self, input: Union[ad.AnnData, List], base: ad.AnnData, idx: Optional[pd.DataFrame] = None):
        base.obs_names_make_unique()
        if self.sample_rate < 1:
            input, base = self.sampleing(input, base)
        if idx is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                input, new_base, label, batch_n = self.data_loader(input, base)
        else:
            if isinstance(input, List):
                batch_n = len(input)
                input_pair = []
                base_pair = []
                for i in range(batch_n):
                    adata = input[i]
                    adata.obs_names_make_unique()
                    adata.obs['batch_new'] = i
                    adata_name = 'input%s_idx' % (i+1)
                    input_pair.append(adata[idx[adata_name]])
                    base_pair.append(base[idx['ref_idx']])
                input = ad.concat(input_pair, merge='same', label="batch")
                new_base = ad.concat(base_pair, merge='same')
                label = np.array(pd.get_dummies(input_pair.obs['batch_new']))
            else:
                batch_n = 1
                input = input[idx['input_idx']]
                new_base = base[idx['ref_idx']]
                label = [1 for i in range(new_base.n_obs)]

        self.input = torch.Tensor(input.X).to(self.device)
        self.base = torch.Tensor(new_base.X).to(self.device)
        self.label = torch.Tensor(label).to(self.device)
        self.D = Discriminator(in_dim=input.n_vars).to(self.device)
        self.G = Batch_G(data_n=batch_n, in_dim=input.n_vars).to(self.device)
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.D_scaler = torch.cuda.amp.GradScaler()
        self.G_scaler = torch.cuda.amp.GradScaler()
        self.L1 = nn.L1Loss().to(self.device)
        if self.pretrain:
            self.train = input
            self.load_weight()

        if self.verbose:
            print('Begin to remove batch effects with ODBC-GAN...')

        self.D.train()
        self.G.train()
        for epoch in range(self.n_epochs):
            self.Update_G()
            for i in range(self.n_critic):
                self.Update_D()
            if self.verbose:
                self.log_print(epoch, self.G_loss, self.D_loss)

        adata = self.test(input, base)

        if self.verbose:
            print('Batch effects have been removed.\n')

        return adata

    def data_loader(self, input, base):
        parameters = {
            'n_epochs': 1000,
            'learning_rate': 1e-3,
            'pretrain': True,
            'GPU': True,
            'verbose': self.verbose,
            'log_interval': 200,
            'weight': None,
            'random_state': self.random_state,
            'fast': self.fast
        }
        align_obs = Align_SC(**parameters)
        if isinstance(input, list):
            idx = align_obs.fit_mult(input, base)
        else:
            idx = align_obs.fit(input, base)
        if isinstance(input, List):
            batch_n = len(input)
            input_pair = []
            base_pair = []
            for i in range(batch_n):
                adata = input[i]
                adata.obs_names_make_unique()
                adata.obs['batch_new'] = i
                adata_name = 'input%s_idx' % (i+1)
                input_pair.append(adata[idx[adata_name]])
                base_pair.append(base[idx['ref_idx']])
            input_pair = ad.concat(input_pair, merge='same', label="batch")
            base_pair = ad.concat(base_pair, merge='same')
            label = np.array(pd.get_dummies(input_pair.obs['batch_new']))
        else:
            batch_n = 1
            input_pair = input[idx['input_idx']]
            base_pair = base[idx['ref_idx']]
            label = [1 for i in range(base_pair.n_obs)]

        return input_pair, base_pair, label, batch_n

    def load_weight(self):
        path = './pretrain_weight/SCNetAE_Instance.pth'
        if not os.path.exists(path):
            Pretrain_SC(self.train, norm_type="Instance")

        # load the pre-trained weights for Encoder and Decoder
        pre_weights = torch.load(path)
        self.G.net.load_state_dict({k: v for k, v in pre_weights.items()})

    def Update_G(self):
        fake_base = self.G(self.input, self.label)
        fake_d = self.D(fake_base)

        Loss_rec = self.L1(fake_base, self.base)
        Loss_adv = -torch.mean(fake_d)
        self.G_loss = (self.weight['w_rec']*Loss_rec +
                       self.weight['w_adv']*Loss_adv)

        self.opt_G.zero_grad()
        self.G_scaler.scale(self.G_loss).backward()
        self.G_scaler.step(self.opt_G)
        self.G_scaler.update()

    def Update_D(self):
        self.opt_D.zero_grad()
        fake_base = self.G(self.input, self.label)

        real_d = self.D(self.base)
        self.D_scaler.scale(-torch.mean(real_d)).backward()

        fake_d = self.D(fake_base.detach())
        self.D_scaler.scale(torch.mean(fake_d)).backward()

        # Compute W-div gradient penalty
        gp = calculate_gradient_penalty(self.base, fake_base, self.D)
        self.D_scaler.scale(gp*self.weight['w_gp']).backward()

        self.D_loss = -torch.mean(real_d) + torch.mean(fake_d) + gp*self.weight['w_gp']

        self.D_scaler.step(self.opt_D)
        self.D_scaler.update()

    @torch.no_grad()
    def test(self, input, base):
        self.G.eval()
        fake_base = self.G(self.input, self.label)
        output = fake_base.cpu().numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            input.X = output
            output = input
            if self.include:
                num_batch = len(output.obs["batch"].unique())
                input_list = []
                for i in range(num_batch):
                    input_list.append(output[output.obs["batch"] == f"{i}"])
                output = ad.concat([base, *input_list], merge='same', label="batch")
            return output

    @torch.no_grad()
    def trans_all(self, input, base):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(input, List):
                for i in range(len(input)):
                    input[i].obs['batch_new'] = i
                input = ad.concat(input, merge='same')
                label = np.array(pd.get_dummies(input.obs['batch_new']))
            else:
                label = [1 for i in range(input.n_obs)]
        self.input = torch.Tensor(input.X).to(self.device)
        self.label = torch.Tensor(label).to(self.device)

        self.G.eval()
        fake_base = self.G(self.input, self.label)
        output = fake_base.cpu().numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            input.X = output
            output = input
            if self.include:
                num_batch = len(input.obs["batch"].unique())
                input_list = []
                for i in range(num_batch):
                    input_list.append(output[output.obs["batch"] == f"{i}"])
                output = ad.concat([base, *input_list], merge='same', label="batch")
            return output
