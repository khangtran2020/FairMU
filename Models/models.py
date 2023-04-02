import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(0, n_layers - 1):
            self.layers.append( nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.n_layers = n_layers
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        h = x
        for i in range(0, self.n_layers):
            h = self.layers[i](h)
            h = self.activation(h)
        h = self.layers[-1](h)
        return h

class Logit(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Logit, self).__init__()
        self.layer_1 = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = self.layer_1(x)
        out = torch.nn.functional.sigmoid(x)
        return out