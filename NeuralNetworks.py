import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
from autograd.misc import flatten
import numpy as np
import autograd.numpy as anp
from autograd import value_and_grad
from numpy import array_split
from numpy.random import choice
from numpy.linalg import solve
import matplotlib.pyplot as plt
import pandas as pd
from autograd import jacobian
import Utils
import Optimizer as ot


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,device):
        super(RNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
      #  self.lsmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        #
      #  out = self.lsmax(out)
        
        return out


def random_initialization_neural_networks(in_size,h_size,n_layers,n_classes,device):
    """
    Initialize the Weights 
    """
    lstm_model = RNN(in_size, h_size, n_layers, n_classes,device).to(device)
    weight = Utils.Flatten_Params(lstm_model)
    torch.manual_seed(0)
    Normal = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.1]))
    weight = Normal.sample((len(weight),))
    lstm_model = Utils.unflatten_param(weight,lstm_model)
    return lstm_model,weight


def squared_loss(NN, inputs, targets,Flattened_Param ,C):
    """
    Regularized squared loss
    """
    Output = NN(inputs)
    L = torch.mean((targets.float().reshape(-1,1)-Output.reshape(-1,1)) ** 2)
    regularizer = C * torch.sum(Flattened_Param**2)
    return L.float() + regularizer.float()


def get_grad(NN, inputs, targets,Flattened_Param ,C):
    """
    Compute the gradient of the loss function
    """
    L = squared_loss(NN,inputs,targets,Flattened_Param,C)
    Gradient = torch.autograd.grad(L,NN.parameters(),retain_graph=True)
    Gradient = Utils.Flatten_Grad(Gradient)
    return L,Gradient

