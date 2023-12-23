import torch
from torch.nn import functional as F
import numpy as np
from Net import classifier_SC
from _utils import seed_everything


class Classify:
    def __init__(self,  n_subtypes, n_epochs, learning_rate, weight_decay, alpha,
                 pretrain, GPU, verbose, log_interval, random_state):
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

        self.n_subtypes = n_subtypes
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.pretrain = pretrain
        self.verbose = verbose
        self.log_interval = log_interval


class Classify_SC(Classify):
    def __init__(self, n_subtypes, n_epochs, learning_rate, weight_decay,
                 alpha, pretrain, GPU, verbose, log_interval, random_state):
        super().__init__(n_subtypes, n_epochs, learning_rate, weight_decay, alpha,
                         pretrain, GPU, verbose, log_interval, random_state)

    def fit(self, z: np.ndarray, res: np.ndarray):
        if self.verbose:
            print('Begin to detect subtypes of outlier cells with ODBC-GAN...')

        z = torch.Tensor(z).to(self.device)
        res = torch.Tensor(res).to(self.device)
        model = classifier_SC(n_subtypes=self.n_subtypes, device=self.device,
                              alpha=self.alpha).to(self.device)

        if self.pretrain:
            # load the pre-trained weights for encoder
            pre_weights = torch.load('./ODBCGAN/pretrain_weight/SCNetAE_Batch.pth')
            model.net.load_state_dict({k: v for k, v in pre_weights.items()})
            for name, value in model.net.named_parameters():
                value.requires_grad = False

        new_z, q = model.fit(z, res, learning_rate=self.learning_rate,
                             n_epochs=self.n_epochs, weight_decay=self.weight_decay,
                             verbose=self.verbose, log_interval=self.log_interval)

        pred = torch.argmax(q, axis=1).cpu().numpy()
        self.z = new_z
        self.q = q

        if self.verbose:
            print('Subtypes of outlier cells have been detected.\n')
        return pred