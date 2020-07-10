from utils import *

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch
from os import path

from sklearn.metrics import *
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from sklearn.metrics import *
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    '''
    Implement very simple feed-foward network for binary clasisfication
    '''
    def __init__(self, params):
        super(Net, self).__init__()
        idim = params['i_dim']
        odim = params['o_dim']
        hdim = params['h_dim']
        self._nlayers = params['n_layers']
        self._af = nn.ReLU
        self._of = nn.Sigmoid()

        self.i_layer = nn.Sequential(
            nn.Linear(idim, hdim[0]),
            self._af(inplace=True))

        layers = []
        for i in range(self._nlayers -1):
            layers.append(nn.Linear(hdim[i], hdim[i+1]))
            layers.append(self._af(inplace=True))
        self.h_layers = nn.Sequential(*layers)

        self.o_layer = nn.Linear(hdim[-1], odim, bias = True)

    def forward(self, x):
        if len(x) == 0:
            return None
        o = self.i_layer(x)
        o = self.h_layers(o)
        o = self.o_layer(o)
        return self._of(o), o

    def predict(self, x):
        pred,_ = self.forward(x)

        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return pred.cpu().detach().numpy()

class MLP(Net):

    '''
    implement multiclass feedforward networks
    '''

    def __init__(self, params):
        super(MLP, self).__init__(params)

    def forward(self, x):
        _, o  = super().forward(x)
        return F.log_softmax(o, dim = 1)

    def predict(self, x):
        output = self.forward(x)
        pred = output.argmax(dim=1, keepdim=True)

        return pred.detach().cpu().numpy()


class LRNet(nn.Module):
    def __init__(self, params):
        super(LRNet, self).__init__()
        idim = params['i_dim']
        self.i_layer = nn.Linear(idim, 1)
        self._of = nn.Sigmoid()

    def forward(self, x):
        if len(x) == 0:
            return None
        o = self.i_layer(x)
        return self._of(o), o

    def predict(self, x):
        pred,_ = self.forward(x)

        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return pred.detach().cpu().numpy()


class RegNet(nn.Module):
    def __init__(self, params):
        super(RegNet, self).__init__()
        idim = params['i_dim']
        odim = params['o_dim']
        hdim = params['h_dim']
        self._nlayers = params['n_layers']
        self._af = nn.ReLU

        self.i_layer = nn.Sequential(
            nn.Linear(idim, hdim[0]),
            self._af(inplace=True))

        layers = []
        for i in range(self._nlayers -1):
            layers.append(nn.Linear(hdim[i], hdim[i+1]))
            layers.append(self._af(inplace=True))
        self.h_layers = nn.Sequential(*layers)

        self.o_layer = nn.Linear(hdim[-1], odim, bias = True)

    def forward(self, x):
        if len(x) == 0:
            return None
        o = self.i_layer(x)
        o = self.h_layers(o)
        o = self.o_layer(o)
        return o, None

    def predict(self, x):
        pred,_ = self.forward(x)
        return pred.cpu().detach().numpy()