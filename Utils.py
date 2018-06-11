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


def init_hyper_param():
    """
    Initialize Hyper Parameters
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequence_length = 28
    input_size = 28
    hidden_size = 14
    num_layers = 1
    num_classes = 10
    batch_size = 20
    return device,sequence_length,input_size,hidden_size,num_layers,num_classes,batch_size


def transform_image_data(batch_size):
    """
    Load and Transform Mnist Data Set into Batches
    """
    train_dataset = torchvision.datasets.MNIST(root=os.getcwd(),
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)
    test_dataset = torchvision.datasets.MNIST(root=os.getcwd(),
                                              train=False, 
                                              transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              shuffle=True)
    return train_loader,test_loader



def one_hot_encoding(labels,num_classes=10):
    """
    One Hot Encoding
    """
    temp = np.zeros([len(labels),num_classes])
    for i in range(len(labels)):
        temp[i][int(labels[i])] = 1
    return torch.from_numpy(temp)


def Flatten_Params(NN):
    """
    Flatten the Parameters of a Given Model
    """
    Parameters_Flattened = list(NN.parameters())[0].reshape(-1,1)
    for i in range(1,len(list(NN.parameters()))):
        Parameters_Flattened = torch.cat((Parameters_Flattened,list(NN.parameters())[i].reshape(-1,1)))
    return Parameters_Flattened

def Flatten_Grad(Gradient):
    """
    Flatten the Gradient
    """
    Parameters_Flattened = list(Gradient)[0].reshape(-1,1)
    for i in range(1,len(Gradient)):
        Parameters_Flattened = torch.cat((Parameters_Flattened,Gradient[i].reshape(-1,1)))
    return Parameters_Flattened



def unflatten_param(Flat_Param,NN):
    """
    Update the Model Parameters given by a Flattened Array
    """
    Prev = 0
    for parameter in NN.parameters():
        if len(parameter.data.shape)>1:
            New = Prev + (parameter.data.shape[0] * parameter.data.shape[1] )
            parameter.data = Flat_Param[Prev:New].reshape(parameter.data.shape)
            Prev =New
        else:
            New = Prev + parameter.data.shape[0]
            parameter.data = Flat_Param[Prev:New].reshape(parameter.data.shape)
            Prev = New
    
    return NN


def next_mini_batch(dataset,batch_size,sequence_length,input_size,low,high):
    """
    Next Mini Batch
    """
    
    X_batch = torch.zeros(batch_size,sequence_length,input_size)
    y_batch = torch.zeros(batch_size)
    Key = np.random.randint(low=low,high=high,size=batch_size)
    i = 0
    for element in Key:
        X_batch[i] = dataset[element][0].reshape(sequence_length,input_size)
        y_batch[i] = dataset.train_labels[element]
        i+=1
    return X_batch,y_batch

def Cal_Accuracy(model,data_set,batch_size,sequence_length,input_size):
    """
    Calculate the Accuracy
    """
    X_batch,y_batch = next_mini_batch(data_set,batch_size,sequence_length,input_size,low=50000,high=60000)
    output = model(X_batch)
    output = torch.max(output,1)[1]
    return (np.sum(output.detach().numpy()==y_batch.detach().numpy())/(batch_size)) * 100


def Valid_Accuracy(model,data_set,batch_size,sequence_length,input_size):
    """Calculate the Avergae Accuracy over an iteration of 500 Batches"""
    iters = 500
    accuracy = torch.zeros(iters)
    for i in range(iters):
        accuracy[i] = Cal_Accuracy(model,data_set,batch_size,sequence_length,input_size)
    return np.mean(accuracy.detach().numpy())


