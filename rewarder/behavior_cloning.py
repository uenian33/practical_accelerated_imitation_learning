import numpy as np  # linear algebra

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils import spectral_norm

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import rewarder.ensemble_models as ensemble_models

torch.manual_seed(0)

def weights_init_kaimingUniform(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, a=0, b=1)
            nn.init.constant_(m.bias, val=0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0.)


class Attention_NN(nn.Module):

    def __init__(self, x_dim, y_dim):
        super(Attention_NN, self).__init__()
        self.mask_layers = nn.Sequential(
            spectral_norm(nn.Linear(x_dim, 64)),
            nn.ReLU(),
            spectral_norm(nn.Linear(64, 128)),
            nn.Dropout(p=0.35),
            nn.ReLU(),
            spectral_norm(nn.Linear(128, 64)),
            nn.Dropout(p=0.35),
            nn.ReLU(),
            spectral_norm(nn.Linear(64, x_dim)),
            nn.ReLU(),
            nn.Softmax()
        )

        self.input_mask_l = nn.Linear(64, x_dim)
        self.input_mask_activate = nn.Softmax()
        self.output_mask_l = nn.Linear(64, x_dim),
        self.output_mask_activate = nn.Softmax()

        self.reg_layers = nn.Sequential(
            spectral_norm(nn.Linear(x_dim, 200)),
            nn.ReLU(),
            spectral_norm(nn.Linear(200, 200)),
            nn.Dropout(p=0.35),
            nn.ReLU(),
            spectral_norm(nn.Linear(200, 10)),
            nn.Dropout(p=0.35),
            nn.ReLU(),
            spectral_norm(nn.Linear(10, y_dim)),
            nn.Tanh()
        )
        weights_init_kaimingUniform(self)

    def forward(self, x):
        mask = self.mask_layers(x)
        #weighted_x = torch.mul(mask, x)
        weighted_x = mask * x
        x = self.reg_layers(weighted_x)
       # print(mask)
        return x  # , mask


class BehaviorCloning(object):
    """docstring for  BehaviorCloning"""

    def __init__(self, train_loader, x_dim, y_dim, epochs=100):
        super(BehaviorCloning, self).__init__()
        self.model = ensemble_models.MLP(x_dim, y_dim)
        # init optimizer
        # optimizer = AdaBound(model.parameters(), lr=1e-3, final_lr=0.001)
        # optimizer = RAdam(model.parameters())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        self.epochs = epochs
        self.train_loader = train_loader

    def train_BC(self):

        # optimizer = optim.Adadelta(model.parameters())  # optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.9)

        loss_fn = nn.MSELoss()
        mean_train_losses = []
        mean_valid_losses = []
        valid_acc_list = []
        epochs = self.epochs
        epoch = 0
        while epoch < epochs:  # for epoch in range(epochs):
                    # print(epoch)
            self.model.train()

            train_losses = []
            for i, (x, labels) in enumerate(self.train_loader):
                x = x.float()
                labels = labels.float()

                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            valid_loss = np.mean(train_losses)

            print(valid_loss, epoch, epochs)
            best_valid = 1000
            if valid_loss < best_valid:
                best_valid = valid_loss
                """
                if epoch == None:
                    torch.save(self.model.state_dict(), self.weight_pth)
                else:
                    torch.save(self.model.state_dict(), self.weight_pth)
                """

            epoch += 1
