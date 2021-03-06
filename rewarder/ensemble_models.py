"""Example on regression using YearPredictionMSD."""

import time
import torch
import numbers
import torch.nn as nn
from torch.nn import functional as F
from sklearn.preprocessing import scale
from sklearn.datasets import load_svmlight_file
from torch.utils.data import TensorDataset, DataLoader

from torchensemble.fusion import FusionRegressor
from torchensemble.voting import VotingRegressor
from torchensemble.bagging import BaggingRegressor
from torchensemble.gradient_boosting import GradientBoostingRegressor
from torchensemble.utils import set_logger


from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

torch.manual_seed(0)

def display_records(records, logger):
    msg = (
        "{:<28} | Testing MSE: {:.2f} | Training Time: {:.2f} s |"
        " Evaluating Time: {:.2f} s"
    )

    print("\n")
    for method, training_time, evaluating_time, mse in records:
        logger.info(msg.format(method, mse, training_time, evaluating_time))


class MLP(nn.Module):

    def __init__(self, x_dim, y_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(x_dim, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, y_dim)

    def forward(self, x):
        #x = x.view(x.size()[0], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class EnsembleModels(object):
    """docstring for Temp"""

    def __init__(self, bc_dataset,
                 n_estimators=5,
                 lr=1e-3,
                 weight_decay=5e-4,
                 epochs=50,
                 batch_size=512,
                 cuda=False,
                 n_jobs=1,
                 save_model=True,
                 save_dir=None,
                 value_type='q',
                 dynamic_pair=False):
        super(EnsembleModels, self).__init__()

        if value_type=='q':
            x_dim = bc_dataset.SAs[0].shape[0] 
            if not dynamic_pair:
                y_dim = 1
                self.train_loader = bc_dataset.q_train_loader
            else:
                y_dim = bc_dataset.nextSs[0].shape[0] 
                self.train_loader = bc_dataset.sa_train_loader
        elif value_type=='v':
            x_dim = bc_dataset.xs[0].shape[0] 
            if not dynamic_pair:
                y_dim = 1
                self.train_loader = bc_dataset.v_train_loader
            else:
                y_dim = bc_dataset.ys[0].shape[0] 
                self.train_loader = bc_dataset.train_loader

        # Define the ensemble
        self.model = BaggingRegressor(estimator=MLP,   # class of your base estimator
                                      estimator_args={'x_dim': x_dim, 'y_dim': y_dim},
                                      n_estimators=n_estimators,            # the number of base estimators
                                      cuda=cuda,
                                      n_jobs=n_jobs)


        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_model = save_model
        self.save_dir = save_dir
        self.init_models()

    def init_models(self):
        self.train_models()
        self.generate_threshold()

    def train_models(self):
        torch.manual_seed(0)
        tic = time.time()

        self.model.fit(self.train_loader, self.lr, self.weight_decay, self.epochs,
                       "Adam", save_model=self.save_model, save_dir=self.save_dir)
        toc = time.time()
        training_time = toc - tic

        print('finish ensemble model training in ', training_time)

    def ensemble_predict(self, x):
        outputs = []
        for idx, estimator in enumerate(self.model.estimators_):
            outputs.append(estimator(x)[-1].detach().numpy())
        return outputs

    def predict_class(self, prev_obs, act, obs, type='q'):
        if type=='v':
            x = torch.FloatTensor((np.array([prev_obs])))
        elif type=='q':
            x = torch.FloatTensor((np.array([np.hstack([prev_obs,act]).flatten()])))

        outputs = self.ensemble_predict(x)
        dist = np.max(euclidean_distances(outputs, outputs))
        if dist < self.thres:
            return True
        else:
            return False

    def generate_threshold(self):
        # train_loader, test_loader = load_data(1)
        discs = []
        predicts = None
        for batch_idx, (data, target) in enumerate(self.train_loader):
            outputs = []
            for idx, estimator in enumerate(self.model.estimators_):
                outputs.append(np.squeeze(estimator(data).detach().numpy()))
            outputs = np.array(outputs)
            outputs.resize(outputs.shape[1], outputs.shape[0])
            if predicts is None:
                predicts = outputs
            else:
                predicts = np.vstack((predicts, outputs))
            # discs.append(np.max(euclidean_distances(outputs, outputs)))

        discs = []
        print("calculating ensemble differences for every training pair...")
        for i in range(predicts.shape[0]):
            discs.append(euclidean_distances(predicts[i].reshape(-1, 1), predicts[i].reshape(-1, 1)))
            # print(predicts.shape)
        self.std = np.std(discs)
        self.mean = np.mean(discs)
        self.maxd = np.max(discs)
        self.thres = self.mean + 0  # (self.maxd - self.mean) / self.std * 0.2
        print(self.std, self.mean, self.maxd, self.thres)

