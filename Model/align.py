import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import anndata as ad
from .Net import Align_G, Discriminator
from ._pretrain import Pretrain_SC
from ._utils import seed_everything, calculate_gradient_penalty


class Align:
    def __init__(self, n_epochs, learning_rate, pretrain, GPU, verbose,
                 log_interval, weight, random_state, fast):
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
        self.pretrain = pretrain
        self.verbose = verbose
        self.log_interval = log_interval
        self.random_state = random_state
        self.fast = fast
        if weight is None:
            self.weight = {'w_rec': 50, 'w_adv': 1, 'w_gp': 10}
        else:
            self.weight = weight

    def log_print(self, epoch, G_loss, D_loss):
        if (epoch+1) % self.log_interval == 0:
            txt = 'Train Epoch: [{:^4}/{:^4}({:^3.0f}%)]    G_loss: {:.6f}    D_loss: {:.6f}'
            txt = txt.format(epoch+1, self.n_epochs, 100.*(epoch+1)/self.n_epochs,
                             G_loss.item(), D_loss.item())
            print(txt)


class Align_SC(Align):
    def __init__(self, n_epochs, learning_rate, pretrain, GPU, verbose, log_interval,
                 weight, random_state, fast):
        super().__init__(n_epochs, learning_rate, pretrain, GPU, verbose, log_interval,
                         weight, random_state, fast)

    def fit(self, input: ad.AnnData, reference: ad.AnnData, log: bool = True):
        if (reference.var_names != input.var_names).any():
            raise RuntimeError('Base data and input data have different genes.')
        if (log and self.verbose):
            print('Begin to find pairs of cells among multiple datasets...')
        if self.fast:
            m = self.align_fast(input, reference)
        else:
            m = self.align_GAN(input, reference)
        idx = pd.DataFrame({'ref_idx': reference.obs.index,
                            'input_idx': input.obs.index[m.argmax(axis=1)]})
        if (log and self.verbose):
            print('Cells have been paired successfully.\n')
        return idx

    def fit_mult(self, inputs: list, reference: ad.AnnData):
        if self.verbose:
            print('Begin to find pairs of cells among multiple datasets...')
        result = [reference.obs.index.tolist()]
        for i in range(len(inputs)):
            idx = self.fit(inputs[i], reference, log=False)
            result.append(idx['input_idx'].tolist())
            if self.verbose:
                txt = 'Datasets: [{:^2}/{:^2}({:^3.0f}%)]'
                txt = txt.format(i+1, len(inputs), 100.*(i+1)/len(inputs))
                print(txt)

        col_name = ['ref_idx'] + ['input%s_idx' % (i+1) for i in range(len(inputs))]
        idx = pd.DataFrame(np.array(result).T, columns=col_name)
        if self.verbose:
            print('Cells have been paired successfully.\n')
        return idx

    def align_fast(self, input, base):
        self.input = torch.Tensor(input.X).to(self.device)
        self.base = torch.Tensor(base.X).to(self.device)
        self.G = Align_G(base.shape[0], input.shape[0], input.n_vars).to(self.device)
        if self.pretrain:
            self.load_weight(input)

        input_z = self.G.net.encode(self.input)
        input_z = F.normalize(input_z, p=1, dim=1)
        base_z = self.G.net.encode(self.base)
        base_z = F.normalize(base_z, p=1, dim=1)

        m = torch.mm(base_z, input_z.T)
        return m.detach().cpu().numpy()

    def align_GAN(self, input, base):
        self.input = torch.Tensor(input.X).to(self.device)
        self.base = torch.Tensor(base.X).to(self.device)
        self.D = Discriminator(in_dim=256, out_dim=[128, 64, 1]).to(self.device)
        self.G = Align_G(base.shape[0], input.shape[0], input.n_vars).to(self.device)
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.D_scaler = torch.cuda.amp.GradScaler()
        self.G_scaler = torch.cuda.amp.GradScaler()
        self.L1 = nn.L1Loss().to(self.device)
        self.L2 = nn.MSELoss().to(self.device)
        if self.pretrain:
            self.load_weight(input)

        self.D.train()
        self.G.train()
        for epoch in range(self.n_epochs):
            self.Update_G()
            self.Update_D()
            if self.verbose:
                self.log_print(epoch, self.G_loss, self.D_loss)

        with torch.no_grad():
            self.G.eval()
            _, _, m = self.G(self.input, self.base)
        return m.detach().cpu().numpy()

    def load_weight(self, adata):
        path = './pretrain_weight/SCNetAE_Batch.pth'
        if not os.path.exists(path):
            Pretrain_SC(adata)

        # load the pre-trained weights for Encoder and Decoder
        pre_weights = torch.load(path)
        self.G.net.load_state_dict({k: v for k, v in pre_weights.items()})

        # freeze the encoder weights
        for name, value in self.G.net.named_parameters():
            value.requires_grad = False

    def Update_G(self):
        fake_z, z, _ = self.G(self.input, self.base)
        fake_d = self.D(fake_z)

        Loss_rec = self.L1(fake_z, z)
        Loss_adv = -torch.mean(fake_d)
        self.G_loss = (self.weight['w_rec']*Loss_rec +
                       self.weight['w_adv']*Loss_adv)

        self.opt_G.zero_grad()
        self.G_scaler.scale(self.G_loss).backward()
        self.G_scaler.step(self.opt_G)
        self.G_scaler.update()

    def Update_D(self):
        self.opt_D.zero_grad()
        fake_z, z, _ = self.G(self.input, self.base)

        real_d = self.D(z)
        self.D_scaler.scale(-torch.mean(real_d)).backward()

        fake_d = self.D(fake_z.detach())
        self.D_scaler.scale(torch.mean(fake_d)).backward()

        # Compute W-div gradient penalty
        gp = calculate_gradient_penalty(z, fake_z, self.D)
        self.D_scaler.scale(gp*self.weight['w_gp']).backward()

        self.D_loss = -torch.mean(real_d) + torch.mean(fake_d) + gp*self.weight['w_gp']

        self.D_scaler.step(self.opt_D)
        self.D_scaler.update()
