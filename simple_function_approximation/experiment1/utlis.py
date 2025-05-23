import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import numpy.random as rn
from torch.utils.data import DataLoader, Dataset
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh
import numpy as np
from scipy.sparse import csr_matrix
import functorch
import math
from functorch import make_functional, vmap, vjp, jvp, jacrev,make_functional_with_buffers,grad

def compute_kernel_matrix(x_data):
    # x_data=x_data.cpu().detach().numpy()
    dot_products = torch.matmul(x_data, x_data.T)
    # dot_products = np.matmul(x_data, x_data.T)
    dot_products = torch.clamp(dot_products, min=-1, max=1) # for numerical stability
    K = (1 / (4 * math.pi)) * (dot_products+1) * (math.pi - torch.acos(dot_products))
    return K

def calculate_transformation_matrix(Gram_matrix,xuhao,replace_start,replace_end):
    # eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(Gram_matrix) 
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    # adjusted eigenvalues
    adjusted_eigenvalues=torch.zeros(len(eigenvalues))
    S = torch.eye(Gram_matrix.size(0)).cuda()
    for i in range(replace_start,replace_end):
        if eigenvalues[i]<=0:
            break
        else:
            adjusted_eigenvalues[i]=eigenvalues[xuhao]
            S-=(1-adjusted_eigenvalues[i]/eigenvalues[i])*(eigenvectors[:, i].view(-1, 1) @ eigenvectors[:, i].view(-1, 1).T)
    return S.detach()

def weighted_loss(ground_truth,outputs,S):
    error=(outputs-ground_truth).squeeze(0).detach()
    error=torch.einsum('ij,jk->ik',S,error)
    outputs=outputs.squeeze(0).contiguous().view(-1,1).t()
    error=error.squeeze(0).contiguous().view(-1,1)
    loss=2*torch.matmul(outputs,error)/torch.tensor(error.shape[0])
    return loss

def compute_NTK(inputs,model):
    fmodel,params=make_functional(model,disable_autograd_tracking=True)
    def compute_loss(params,inputs):
        batch = inputs.unsqueeze(0)
        predictions = fmodel(params,batch).sum() #此处的限制为参数必须连续
        return predictions
    inputs=inputs.squeeze(0)
    ft_compute_grad=vmap(grad(compute_loss,argnums=0),(None,0))
    jac1=ft_compute_grad(params,inputs)
    jac1=jac1[:2] # 4 for m=2; 2 for m=1
    jac1 = torch.cat([tensor.flatten(1) if len(tensor.shape) > 2 else tensor for tensor in jac1],dim=1)
    pNTK=torch.matmul(jac1,jac1.T)
    return pNTK 

def compute_NTK_IGA(inputs,model,IGA):
    a=inputs
    a=a.reshape(inputs.shape[0]//IGA,IGA,2)
    a=a[:,IGA//2,:]
    inputs=a
    fmodel,params=make_functional(model,disable_autograd_tracking=True)
    def compute_loss(params,inputs):
        batch = inputs.unsqueeze(0)
        predictions = fmodel(params,batch).sum() #此处的限制为参数必须连续
        return predictions
    inputs=inputs.squeeze(0)
    ft_compute_grad=vmap(grad(compute_loss,argnums=0),(None,0))
    jac1=ft_compute_grad(params,inputs)
    jac1=jac1[:2] # 4 for m=2; 2 for m=1
    jac1 = torch.cat([tensor.flatten(1) if len(tensor.shape) > 2 else tensor for tensor in jac1],dim=1)
    pNTK=torch.matmul(jac1,jac1.T)
    return pNTK

def weighted_loss_IGA(ground_truth,outputs,S,IGA):
    error=(outputs-ground_truth).squeeze(0).detach()
    error=error.reshape(outputs.shape[0]//IGA,IGA,1)
    error=torch.einsum('ij,jkl->ikl',S,error)
    outputs=outputs.squeeze(0).contiguous().view(-1,1).t()
    error=error.squeeze(0).contiguous().view(-1,1)
    loss=2*torch.matmul(outputs,error)/torch.tensor(error.shape[0])
    return loss