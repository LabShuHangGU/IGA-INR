# The following code is adapted from the previous frameworks: FR-INR and WIRE.
# Link to the original frameworks:  https://github.com/LabShuHangGU/FR-INR and https://github.com/vishwa91/wire/tree/main
from scipy import io
from scipy import ndimage
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import copy
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
import math
import lpips
from functorch import make_functional, vmap, vjp, jvp, jacrev,make_functional_with_buffers,grad
import tqdm
import warnings
import argparse
import open3d as o3d
import mcubes
from tensorboardX import SummaryWriter
from utlis import *
from models import *
import sys
import wandb
import random
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

def seed_torch(seed): 
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch(4096) 

def get_3Dgrid(H,W,T,dim=3):
    tensors=(torch.linspace(-1,1,steps=H),torch.linspace(-1,1,steps=W),torch.linspace(-1,1,steps=T))
    mgrid=torch.stack(torch.meshgrid(*tensors),dim=-1)
    return mgrid

def weighted_loss(ground_truth,outputs,S):
    batch_num=ground_truth.shape[0]
    group_size=ground_truth.shape[1]
    ground_truth=ground_truth.reshape(-1,1)
    error=(outputs-ground_truth).squeeze(0).detach()
    error=error.reshape(batch_num,group_size,1)
    error=torch.einsum('ij,jkl->ikl',S,error)
    error=error.reshape(outputs.shape)
    outputs=outputs.squeeze(0).contiguous().view(-1,1).t()
    error=error.squeeze(0).contiguous().view(-1,1)
    loss=2*torch.matmul(outputs,error)/torch.tensor(error.shape[0])
    return loss

def compute_NTK(inputs,model,random_select,patch_number,pixelvalues,target):
    a=inputs # the shape of a: batch_num ,group_size coordinates_dimension
    outputs=pixelvalues.reshape(a.shape[0],a.shape[1],1)
    # error=abs(target-outputs)
    # max_indices = torch.argmax(error.squeeze(2), dim=1, keepdim=True).squeeze(1).cuda() 
    # print(max_indices.shape)
    if random_select:
        b=random.randint(0,inputs.shape[1]-1)
    else:
        b=inputs.shape[1]//2
    a=a[:,b,:] # Here, we simply adopt the fixed points
    # a=a[:,max_indices,:]
    inputs=a
    fmodel,params=make_functional(model,disable_autograd_tracking=True)
    def compute_loss(params,inputs):
        batch = inputs.unsqueeze(0)
        predictions = fmodel(params,batch).sum()
        return predictions
    inputs=inputs.squeeze(0)
    ft_compute_grad=vmap(grad(compute_loss,argnums=0),(None,0))
    jac1=ft_compute_grad(params,inputs)
    jac1 = torch.cat([tensor.flatten(1) if len(tensor.shape) > 2 else tensor for tensor in jac1],dim=1)
    pNTK=torch.matmul(jac1,jac1.T)
    return pNTK 
    
def calculate_precondition_matrix(Gram_matrix,xuhao,replace_start,replace_end):
    # eigenvalues, eigenvectors = torch.linalg.eigh(Gram_matrix)
    S = torch.eye(Gram_matrix.size(0)).cuda()
    # Sometimes, issues with CUDA may arise during feature decomposition on the GPU, 
    # so the CPU is used as a substitute. 
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(Gram_matrix.cpu()) 
    except torch._C._LinAlgError as e:
        mistake=1
        return S.detach(), mistake
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices].cuda()
    eigenvectors = eigenvectors[:, sorted_indices].cuda()
    # adjusted eigenvalues
    adjusted_eigenvalues=torch.zeros(len(eigenvalues))
    for i in range(replace_start,replace_end):
        if eigenvalues[i]<=0:
            break
        else:
            adjusted_eigenvalues[i]=eigenvalues[xuhao]
            S-=(1-adjusted_eigenvalues[i]/eigenvalues[i])*(eigenvectors[:, i].view(-1, 1) @ eigenvectors[:, i].view(-1, 1).T)
    return S.detach()

def get_args():
    parser=argparse.ArgumentParser(description='Hyperparameter settings for training a model.')
    parser.add_argument('--name',type=str,default='arma')
    parser.add_argument('--mode',type=str,default='relu')
    parser.add_argument('--learning_rate',type=float,default=5e-3)
    parser.add_argument('--first_omega0',type=float,default=30.0)
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--gradient_adjust',action='store_true',default=False)
    parser.add_argument('--xuhao',type=int,default=30)
    parser.add_argument('--replace_start',type=int,default=0)
    parser.add_argument('--replace_end',type=int,default=30)  
    parser.add_argument('--random_index',action='store_true',default=False)
    parser.add_argument('--group_num',type=int,default=1024)
    parser.add_argument('--group_size',type=int,default=512)
    parser.add_argument('--patch_size',type=int,default=8)
    parser.add_argument('--alpha',type=float,default=0.01)
    parser.add_argument('--high_freq_num',type=int, default=256)
    parser.add_argument('--low_freq_num',type=int,default=256)
    parser.add_argument('--phi_num',type=int,default=8)
    parser.add_argument('--hidden_layers',type=int,default=2)
    parser.add_argument('--hidden_features',type=int,default=256)
    parser.add_argument('--saving', action="store_true", default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args=get_args()
    print(args.name)
    start_time=time.time()
    data=io.loadmat('/data0/home/shikexuan/exp3/data/'+args.name+'.mat')['hypercube'].astype(np.float32)
    #-----------------------------
    scale=1.0
    mcubes_thres = 0.5  # Threshold for marching cubes
    #-----------------------------
    data=ndimage.zoom(data/data.max(), [scale, scale, scale], order=0)
    print(data.shape)
    # Clip to tightest bound ing box
    hidx, widx, tidx = np.where(data > 0.99)
    data = data[hidx.min():hidx.max(),widx.min():widx.max(),tidx.min():tidx.max()]
    H, W, T = data.shape
    maxpoints=int(1e5) #follow the previous settings
    maxpoints=min(H*W*T, maxpoints)
    dataten=torch.tensor(data).cuda().reshape(H*W*T, 1)
    #-----------------------------
    model=INR(mode=args.mode, in_features=3, hidden_features=args.hidden_features, hidden_layers=args.hidden_layers, out_features=1, outermost_linear=True, 
        high_freq_num=args.high_freq_num, 
        low_freq_num=args.low_freq_num, 
        phi_num=args.phi_num, 
        alpha=args.alpha, 
        first_omega_0=args.first_omega0, 
        hidden_omega_0=args.first_omega0)
    
    model=model.cuda()
    #-----------------------------
    # Optimizer and scheduler
    learning_rate=args.learning_rate #Siren: 5e-4; ReLU: 2e-3 
    #-----------------------------
    optim=torch.optim.Adam(lr=learning_rate, params=model.parameters())

    scheduler=LambdaLR(optim, lambda x: 0.1**min(x/args.epochs, 1))
    # loss function
    criterion=torch.nn.MSELoss()
    # Create inputs
    coords=get_coords(H, W, T)
    mse_array = np.zeros(args.epochs)
    time_array = np.zeros(args.epochs)
    best_mse = float('inf')
    best_results = None
    
    tbar=tqdm.tqdm(range(args.epochs))
    im_estim=torch.zeros((H*W*T, 1), device='cuda')
    tic=time.time()
    print('Running mode %s' %args.mode)
    if args.gradient_adjust:
        print('with Inductive Gradient Adjustment', end=' ')
    if args.random_index:
        print('Random Index')

    print('Partition data simply based on the order of expansion.')
    # define the group size p, referring to our paper
    group_size=args.group_size
    # define the number of group in one iteration
    group_num=args.group_num  

    # the number of the whole groups
    total_group_num=coords.shape[0]//group_size 

    group_coords=coords[0:total_group_num*group_size,:].reshape(total_group_num,group_size,3) 
    remain_coords=coords[total_group_num*group_size:,:].cuda() 

    # Random group indices
    group_indices=torch.randperm(group_coords.shape[0])

    # define the values of the corresponding coords
    group_dataten=dataten[0:total_group_num*group_size,:].reshape(total_group_num,group_size,1)
    remain_dataten=dataten[total_group_num*group_size:,:]

    im_estim=torch.zeros(total_group_num,group_size,1).cuda()
    remain_im_estim=torch.zeros(remain_dataten.shape[0],1).cuda()

    print('group_coords:',group_coords.shape, 'remain_coords:',remain_coords.shape, 'group size:',args.group_size, 'group_num:',args.group_num)
    print('group_dataten:',group_dataten.shape, 'remain_dataten:',remain_dataten.shape)

    print('-------------------')
    if args.gradient_adjust:
        print('xuhao:',args.xuhao,'replace_start:', args.replace_start ,'replace_end:', args.replace_end)
    records = [] 
    for idx in tbar:
        if args.random_index:
            # In each iteration, we randomly generate the indexes to approximate random sampling.
            group_indices=torch.randperm(total_group_num) 
        else:
            group_indices=torch.arange(total_group_num)  

        for b_idx in range(0, total_group_num, group_num):
            b_indices=group_indices[b_idx:min(total_group_num,b_idx+group_num)]
            b_coords=group_coords[b_indices,...].cuda()
            b_coords=b_coords.reshape(-1,3)
            if 'bn' in args.mode:
                pixelvalues=model(b_coords[None, ...].squeeze(0)).squeeze()[:, None] 
            else:
                pixelvalues=model(b_coords[None, ...]).squeeze()[:, None] 
            with torch.no_grad():
                im_estim[b_indices, :] = pixelvalues.reshape(pixelvalues.shape[0]//group_size,group_size,1) 
            loss = criterion(pixelvalues, group_dataten[b_indices, :].reshape(-1,1))

            if args.gradient_adjust:
                pNTK=compute_NTK(group_coords[b_indices,...].cuda(),model,False,30,pixelvalues,group_dataten[b_indices, :])
                S=calculate_precondition_matrix(pNTK,args.xuhao,args.replace_start,args.replace_end)
                new_loss=weighted_loss(group_dataten[b_indices, :],pixelvalues,S)
                optim.zero_grad()
                new_loss.backward()
            else:
                optim.zero_grad()
                loss.backward() 
            optim.step()
            lossval=loss.item()
        if remain_coords.shape[0]==1:
            remain_pixel_values=torch.zeros(remain_dataten.shape).cuda()
            remian_im_estim=remain_pixel_values
        else:
            # For remaining data, we don't use the gradient adjustment.
            if 'bn' in args.mode:
                remain_pixel_values=model(remain_coords[None, ...].squeeze(0)).squeeze()[:, None]
            else:
                remain_pixel_values=model(remain_coords[None, ...]).squeeze()[:, None] 

            with torch.no_grad():
                remian_im_estim=remain_pixel_values

            loss=criterion(remain_pixel_values,remain_dataten)
            optim.zero_grad()
            loss.backward()
            optim.step()
        scheduler.step()

        mse_array[idx] = get_IoU(torch.cat((im_estim.reshape(-1,1),remian_im_estim),dim=0),torch.cat((group_dataten.reshape(-1,1),remain_dataten),dim=0),mcubes_thres)
        if mse_array[idx]>=max(mse_array):
            best_results=torch.cat((im_estim.reshape(-1,1),remian_im_estim),dim=0)
            best_model_state = model.state_dict() #Store the best model parameters.
        time_array[idx]=time.time()
        end_time=time.time()
        step_time=start_time-end_time
        records.append((step_time, mse_array[idx]))
        tbar.set_description('%.4e'%mse_array[idx])
        tbar.refresh()
    df = pd.DataFrame(records)
    if args.gradient_adjust:
        savename = args.name+'_'+args.mode+'_IGA_'+str(args.xuhao)+'_'+str(args.learning_rate)+'_'+str(args.hidden_layers)+'_epochs_'+str(args.epochs)
    else:
        savename = args.name+'_'+args.mode+'_'+str(args.learning_rate)+'_'+str(args.hidden_layers)+'_epochs_'+str(args.epochs+500)
    df.to_csv('iou_'+savename+'.csv',index=False,header=False)
    total_time=time.time()-tic
    print('max_IOU',max(mse_array))
    print('final learning rate: {}'.format(scheduler.get_last_lr())) 

str_max_iou= f"{max(mse_array):.5f}"

# if you want to visualize the ground truth, 
# please uncomment the following lines and don't forget to modify the savename.
# best_results=torch.cat((group_dataten.reshape(-1,1),remain_dataten),dim=0).reshape(H,W,T).detach().cpu().numpy()

# The best result
# best_results=best_results.reshape(H,W,T).detach().cpu().numpy()
if args.saving:
    if args.gradient_adjust:
        savename = 'results/%s.dae'%(args.name+'_'+args.mode+'_IGA_'+str(args.xuhao)+'_'+str_max_iou+'_'+str(args.learning_rate)+'_'+str(args.hidden_layers)+'_'+str(args.first_omega0))
        march_and_save(best_results, mcubes_thres, savename, True)
        torch.save(best_model_state, 'models/'+args.name+'_'+args.mode+'_IGA_'+str(args.xuhao)+'_'+str_max_iou+'_'+str(args.learning_rate)+'_'+str(args.hidden_layers)+'_'+str(args.first_omega0)+'.pth')
    else:
        savename ='results/%s.dae'%(args.name+'_'+args.mode+'_'+str_max_iou+'_'+str(args.learning_rate)+'_'+str(args.hidden_layers)+'_'+str(args.first_omega0))
        march_and_save(best_results, mcubes_thres, savename, True)
    torch.save(best_model_state, 'models/'+args.name+'_'+args.mode+'_'+str_max_iou+'_'+str(args.learning_rate)+'_'+str(args.hidden_layers)+'_'+str(args.first_omega0)+'.pth')


















