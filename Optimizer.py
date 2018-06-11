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
import NeuralNetworks as Net


def Adam_optimization(input_size, hidden_size, num_layers, num_classes,learning_rate):
    """
    Specify the Loss Function and the Model for Adam Optimization
    """
    model = Net.RNN(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model,criterion,optimizer

def RMSprop_optimization(input_size, hidden_size, num_layers, num_classes,learning_rate):
    """
    Specify the Loss Function and the Model for RMSprop Optimization
    """
    model = Net.RNN(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    return model,criterion,optimizer


def get_Jacobian(NN,inp):
    """
    Calculate Jacobian Matrix
    """
    y_hat = (NN(inp)).reshape(-1,1)
    Tot_Param = np.sum(np.asarray([len(p.view(-1)) for p in NN.parameters()]))
    Jacobian = np.zeros([len(y_hat),Tot_Param])
    for i in range(len(y_hat)):
        Gradient =torch.autograd.grad(y_hat[i], NN.parameters(),retain_graph=True)
        Gradient = Utils.Flatten_Grad(Gradient)
        Jacobian[i,:] = Gradient.reshape(int(Tot_Param))
        
    return Jacobian


def adam(NN,data_set,Flattened_Param,batch_size,sequence_length,input_size,low=0,high=50000 ,max_it=1000, step_size=0.01, b1=0.9, b2=0.999, eps=10**-8,C=0):
   
    
    m = anp.zeros(len(Flattened_Param))
    v = anp.zeros(len(Flattened_Param))
    err = anp.zeros(max_it)
    grad = anp.zeros(max_it)
    
    for t in range(max_it):
        
        images,labels = Utils.next_mini_batch(data_set,batch_size,sequence_length,input_size,low,high)
        labels = Utils.one_hot_encoding(labels)
        
        
        err[t], g = Net.get_grad(NN,images,labels,Flattened_Param,C)
        grad[t] = torch.sum(g**2)
        
        m = ((1 - b1) * g).detach().numpy()      + (b1 * m).reshape(-1,1)  # First  moment estimate.
        v = ((1 - b2) * (g**2)).detach().numpy() + (b2 * v).reshape(-1,1)  # Second moment estimate.
        mhat = m / (1 - b1**(t + 1))    # Bias correction.
        vhat = v / (1 - b2**(t + 1))
        Flattened_Param = Flattened_Param - (torch.from_numpy(step_size*mhat/(anp.sqrt(vhat) + eps))).float()
        #update the parameters in NN model
        NN = Utils.unflatten_param(Flattened_Param,NN)
        
    return NN, err, grad



def sca_ridge(NN,Flattened_Param,data_set,batch_size,sequence_length,input_size,low=0,high=50000,step_size=0.12,step_size_eps=0.0
              ,rho = 0.9,rho_eps=0.0,C=0,blocks=1,virtual_processors=1,tau=0.2,max_it=1000):
    """
    Stochastic Convex Approximation
    """
    message = 'Successful' 
    if blocks < virtual_processors:
        blocks = virtual_processors
        
    err = anp.zeros(max_it)
    grad = anp.zeros(max_it)
    d = anp.zeros(len(Flattened_Param))
    par_idx = array_split(anp.arange(len(Flattened_Param)),blocks)
    par_not_idx = anp.ones((blocks, len(Flattened_Param)), dtype=bool)
    
    for p in range(blocks):
        par_not_idx[p, par_idx[p]] = False
        
        
    
    for t in range(max_it):
        
        blocks_t = choice(blocks, virtual_processors, replace=False)
        
        images,labels = Utils.next_mini_batch(data_set,batch_size,sequence_length,input_size,low,high)
        labels = Utils.one_hot_encoding(labels)
        
        #predict and compute current error
        output = NN(images)
        
        
        J = get_Jacobian(NN,images)
        
        e = (labels.float() - output.float()).reshape(-1,1)
        
        
        
        #compute loss on current iteration
        err[t] = Net.squared_loss(NN,images,labels,Flattened_Param,C)
        #err[t] = torch.mean(e**2) + C * torch.mean(param**2)
        #err[t] = torch.mean(torch.mul(e,e).float() + C * torch.mean(torch.mul(param,param)))
        #torch.mean((targets.float().reshape(-1,1)-Output.reshape(-1,1)) ** 2
        #C * torch.mean(Flattened_Param**2)
        
        #compute gradient on current iteration
        g_loss = -2.0 * torch.mean(e.reshape(-1,1).float() * torch.from_numpy(J).float(),dim=0)
        grad[t] = torch.sum((g_loss + (C * 2.0 * Flattened_Param)) ** 2 )
        
        #compute residuals
        r = e.reshape(-1,1).float() + torch.mm(torch.from_numpy(J).float(),Flattened_Param.float())
        
        for p in range(virtual_processors):
            
            # Get current block and indices
            block = blocks_t[p]
            idx = par_idx[block]
            
            # Compute current A block
            A_rowblock = anp.dot(J[:, idx].T, J)
            
            # Compute J.T times r
            Jr = anp.dot(J[:, idx].T, np.array(r.detach().numpy())).reshape(-1)
            
            #compute A,b matrices
            A = (rho/len(labels))*A_rowblock[:, idx] + (C + tau)*anp.eye(len(idx))
            b = ((rho/len(labels))*Jr - ((1-rho)*0.5)*d[idx] + (tau*Flattened_Param.detach().numpy()[idx]).reshape(len(Flattened_Param),)).reshape(-1,1) -\
                (rho/len(labels))*np.dot(A_rowblock[:, par_not_idx[block]], Flattened_Param.detach().numpy()[par_not_idx[block]])
                
            # Solve surrogate optimization
            try:
                par_hat = solve(A, b)
                # Update auxiliary variables
                d[idx] = (1-rho)*d[idx] + rho*g_loss.detach().numpy()[idx]
                # Update variable
                Flattened_Param[idx] = ((1-step_size)*Flattened_Param[idx]).float() + torch.from_numpy(step_size*par_hat).float()
    
            except np.linalg.LinAlgError as e:
                if 'Singular matrix' in str(e):
                    d[idx] = d[idx]
                    Flattened_Param[idx] = Flattened_Param[idx]
                    message = str(e)
                    break
                    
        
        # Update stepsize and rho
        rho = rho*(1-rho_eps*rho)
        step_size = step_size*(1-step_size_eps*step_size)
        
        #update the parameters in NN model
        NN = Utils.unflatten_param(Flattened_Param,NN)
            
    return NN,err,grad,message
