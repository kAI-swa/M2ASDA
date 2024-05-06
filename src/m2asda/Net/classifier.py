import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans
from ._net import SCNetAE


class classifier_SC(nn.Module):
    def __init__(self, in_dim=256, res_dim=3000, n_subtypes=3, alpha=1, device='cuda:0'):
        super().__init__()
        self.classes = n_subtypes
        self.alpha = alpha
        self.device = device
        self.net = SCNetAE(res_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim*2, nhead=2)
        self.Trans = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.mu = Parameter(torch.Tensor(self.classes, in_dim*2)).to(self.device)

    def forward(self, new_z):
        x = self.Trans(new_z)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-8)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        self.mu_update(x, q)
        return x, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def mu_init(self, feat):
        kmeans = KMeans(self.classes, n_init=20)
        y_pred = kmeans.fit_predict(feat)
        feat = pd.DataFrame(feat, index=np.arange(0, feat.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, feat.shape[0]), name="Group")
        Mergefeat = pd.concat([feat, Group], axis=1)
        centroid = np.asarray(Mergefeat.groupby("Group").mean())

        self.mu.data.copy_(torch.Tensor(centroid))

    def mu_update(self, feat, q):
        y_pred = torch.argmax(q, axis=1).cpu().numpy()
        feat = pd.DataFrame(feat.cpu().detach().numpy(), index=np.arange(0, feat.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, feat.shape[0]), name="Group")
        Mergefeat = pd.concat([feat, Group], axis=1)
        centroid = np.asarray(Mergefeat.groupby("Group").mean())

        self.mu.data.copy_(torch.Tensor(centroid))

    def fit(self, z, res, learning_rate=1e-4, n_epochs=100, update_interval=3,
            weight_decay=1e-4, verbose=True, log_interval=1):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scaler = torch.cuda.amp.GradScaler()

        new_z = torch.cat([z, self.net.encode(res)], dim=1)
        self.mu_init(self.Trans(new_z).cpu().detach().numpy())

        self.train()
        for epoch in range(n_epochs):
            if epoch % update_interval == 0:
                _, q = self.forward(new_z)
                p = self.target_distribution(q).data

            optimizer.zero_grad()
            _, q = self.forward(new_z)
            loss = self.loss_function(p, q)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ((epoch+1) % log_interval == 0) and (verbose):
                txt = 'Train Epoch: [{:^4}/{:^4}({:^3.0f}%)]    Loss: {:.6f}'
                txt = txt.format(epoch+1, n_epochs, 100.*(epoch+1)/n_epochs, loss)
                print(txt)

        with torch.no_grad():
            self.eval()
            new_z, q = self.forward(new_z)
            return new_z, q