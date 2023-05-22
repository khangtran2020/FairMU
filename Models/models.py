import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer, dropout=None):
        super(NN, self).__init__()
        self.n_hid = n_layer - 2
        self.in_layer = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        nn.init.kaiming_uniform_(self.in_layer.weight, nonlinearity="relu")
        self.hid_layer = []
        for i in range(self.n_hid):
            layer = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            self.hid_layer.append(layer)
        self.out_layer = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None


    def forward(self, x):
        h = torch.nn.functional.relu(self.in_layer(x))
        for i in range(self.n_hid):
            h = self.dropout(h) if self.dropout is not None else h
            h = torch.nn.functional.relu(self.hid_layer[i](h))
        h = torch.nn.functional.sigmoid(self.out_layer(h))
        return h


class LR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LR, self).__init__()
        self.out_layer = nn.Linear(input_dim, output_dim)
        nn.init.kaiming_uniform_(self.out_layer.weight, nonlinearity="sigmoid")

    def forward(self, x):
        h = torch.nn.functional.sigmoid(self.out_layer(x))
        return h