import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import numpy as np
import pandas as pd
import time
import math
import lpips
import warnings
import random
import sys
from skimage import transform
import functorch
from functorch import make_functional, vmap, vjp, jvp, jacrev,make_functional_with_buffers,grad
from tqdm.autonotebook import tqdm
import random
import argparse
from models import *
# Suppress warnings of a specific category.
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

def get_mgrid(H,W,dim=2):
    tensors=(torch.linspace(-INPUT_RANGE,INPUT_RANGE,steps=H),torch.linspace(-INPUT_RANGE,INPUT_RANGE,steps=W))
    mgrid=torch.stack(torch.meshgrid(*tensors),dim=-1)
    return mgrid

def get_image_tensor(fig):
    img = Image.fromarray(fig)
    transform = Compose([
        ToTensor()
    ])
    img = transform(img)
    return img


# This dataset includes the step for image patches.
class ImageFitting(Dataset):
    def __init__(self,fig,patch_size):
        super().__init__()
        img=get_image_tensor(fig) #[3,512,768]
        self.coords=get_mgrid(img.shape[1],img.shape[2],2) #[512,768,2]
        self.origin_coords=self.coords
        self.pixels=img
        print(self.pixels.shape)
        self.coords=self.coords.unfold(0,patch_size,patch_size).unfold(1,patch_size,patch_size).permute(0,1,3,4,2)
        self.pixels=self.pixels.unfold(1,patch_size,patch_size).unfold(2,patch_size,patch_size).permute(1,2,3,4,0)
        self.coords=self.coords.reshape(self.coords.size(0)*self.coords.size(1),self.coords.size(2),self.coords.size(3),self.coords.size(4)) #[256,patch_size,patch_size,2]
        self.pixels=self.pixels.reshape(self.pixels.size(0)*self.pixels.size(1),self.pixels.size(2),self.pixels.size(3),self.pixels.size(4))# [256,patch_size,patch_size,1]
        # print(self.coords.shape) #coords: [384,patch_size,patch_size,2]
        # print(self.pixels.shape) #pixels: [384,patch_size,patch_size,3]
   
        self.coords=self.coords.permute(0,3,1,2).reshape(self.coords.shape[0],self.coords.shape[3],-1).permute(0,2,1)
        self.pixels=self.pixels.permute(0,3,1,2).reshape(self.pixels.shape[0],self.pixels.shape[3],-1).permute(0,2,1) 
        # print(self.coords.shape) 
        # print(self.pixels.shape)
        self.patch_number=self.coords.shape[0]
        self.patch_size=self.coords.shape[1]
        self.out_dim=self.pixels.shape[-1]
        self.in_dim=self.coords.shape[-1]
        self.H=img.shape[1]
        self.W=img.shape[2]

        self.coords=self.coords.reshape(-1,2)
        self.pixels=self.pixels.reshape(-1,3) 
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        if idx > 0: raise IndexError
        return self.coords, self.pixels

    
def compute_NTK(inputs,model,random_select,photo,model_output,ground_truth):
    a=inputs.reshape(photo.patch_number,photo.patch_size,photo.in_dim)
    output=model_output.reshape(photo.patch_number,photo.patch_size,photo.out_dim)
    ground_truth=ground_truth.reshape(photo.patch_number,photo.patch_size,photo.out_dim)
    tensor=torch.sum(abs(output-ground_truth),dim=2,keepdim=True) #[384,1024,1]
    max_indices = torch.argmax(tensor.squeeze(2), dim=1, keepdim=True).squeeze(1) # [384,1]
    if random_select:
        b=random.randint(0,photo.patch_size-1)
    else:
        b=photo.patch_size//2 
    a=a[:,max_indices,:] # Use the point of large error
    # a=a[:,b,:] # Simply select the fixed points
    inputs=a
    # Use a new model to compute the pNTK, which can be found in our Implementation Details "Multidimensional Output Approximation".
    new_model=nn.Sequential(model,nn.Linear(photo.out_dim,1,bias=False)).cuda()
    nn.init.constant_(new_model[-1].weight,torch.tensor(1.0/torch.sqrt(torch.tensor(photo.out_dim))))
    for param in new_model[-1].parameters():
        param.requires_grad=False
    
    fmodel,params=make_functional(new_model,disable_autograd_tracking=True)
    def compute_loss(params,inputs):
        batch = inputs.unsqueeze(0)
        predictions = fmodel(params,batch).sum() 
        return predictions
    inputs=inputs.squeeze(0)
    ft_compute_grad=vmap(grad(compute_loss,argnums=0),(None,0))
    jac1=ft_compute_grad(params,inputs)
    jac1=list(jac1)
    jac1.pop() # Remove the gradients of the additional layer in "new_model" of parameters.
    jac1 = torch.cat([tensor.flatten(1) if len(tensor.shape) > 2 else tensor for tensor in jac1],dim=1)
    pNTK=torch.matmul(jac1,jac1.T)
    return pNTK 

# Construction Strategy
def calculate_precondition_matrix(Gram_matrix,xuhao,replace_start,replace_end):
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


# Apply the transformation to the loss function
def weighted_loss(ground_truth,outputs,S,photo):
    error=(outputs-ground_truth).squeeze(0).detach()
    error=error.reshape(photo.patch_number,photo.patch_size,photo.out_dim)
    error=torch.einsum('ij,jkl->ikl',S,error)
    error=error.reshape(outputs.shape)
    outputs=outputs.squeeze(0).contiguous().view(-1,1).t()
    error=error.squeeze(0).contiguous().view(-1,1)
    loss=2*torch.matmul(outputs,error)/torch.tensor(error.shape[0])
    return loss

def get_args():
    parser = argparse.ArgumentParser(description='Hyperparameter settings for training a model.')

    parser.add_argument('--photo_name',type=str,help='photo name for saving')
    parser.add_argument('--photo_address', type=str, default='/data0/home/shikexuan/kodim_photo/kodim01.png', help='')
    parser.add_argument('--mode',type=str,default='relu')
    parser.add_argument('--patch_size',type=int,default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs')
    parser.add_argument('--step_size',type=int,default=3000)
    parser.add_argument('--gradient_adjust',action="store_true",default=False)
    parser.add_argument('--random_select',action="store_true",default=False)
    parser.add_argument('--xuhao',type=int,default=30)
    parser.add_argument('--replace_start',type=int,default=0)
    parser.add_argument('--replace_end',type=int,default=30)
    parser.add_argument('--step_if',action="store_true",default=False)
    parser.add_argument('--saving',action="store_true",default=False)
    parser.add_argument('--high_freq',type=int,default=128)
    parser.add_argument('--low_freq',type=int,default=128)
    parser.add_argument('--phi_num',type=int,default=32)
    parser.add_argument('--alpha',type=float,default=0.01)
    parser.add_argument('--hidden_features',type=int,default=256)
    parser.add_argument('--hidden_layers',type=int,default=2)
    args = parser.parse_args()
    return args

args = get_args()
if args.gradient_adjust:
    print('Using IGA','patch size',args.patch_size,'xuhao',args.xuhao)
print('current_photo',args.photo_name,'current_mode',args.mode,'learning rate',args.learning_rate)
start_time=time.time()
INPUT_RANGE=1
photo=skimage.io.imread(args.photo_address+args.photo_name+'.png')
photo=ImageFitting(photo,args.patch_size)
dataloader=DataLoader(photo,batch_size=1,pin_memory=True,num_workers=1)

model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

if args.mode=='wire_finer':
    model=WF(in_features=2, out_features=3, hidden_layers=args.hidden_layers, hidden_features=args.hidden_features)
elif args.mode=='gauss_finer':
    model=GF(in_features=2, out_features=3, hidden_layers=args.hidden_layers, hidden_features=args.hidden_features)
else:
    model=INR(args.mode,in_features=2,hidden_features=args.hidden_features,hidden_layers=args.hidden_layers,out_features=3,outermost_linear=True,high_freq_num=args.high_freq,low_freq_num=args.low_freq,phi_num=args.phi_num,
            alpha=args.alpha,first_omega_0=30.0,hidden_omega_0=30.0)
model.cuda()

optim=torch.optim.Adam(lr=args.learning_rate,params=model.parameters())
scheduler=StepLR(optim,step_size=args.step_size,gamma=0.1)

max_psnr=1.0
best_model_state = None 
records = [] 
with tqdm(args.epochs) as pbar:
    for step in range(args.epochs):
        model_output=model(model_input[0,:,:]) # for bn
        # model_output=model(model_input) 

        loss=((model_output-ground_truth)**2).mean()
        psnr=10*np.log10(1.0**2/loss.item())
        if args.gradient_adjust:
            pNTK=compute_NTK(model_input,model,args.random_select,photo,model_output,ground_truth)
            S=calculate_precondition_matrix(pNTK,args.xuhao,args.replace_start,args.replace_end)
            new_loss=weighted_loss(ground_truth,model_output,S,photo)
            optim.zero_grad()
            new_loss.backward()
        else:
            optim.zero_grad()
            loss.backward()
        optim.step()
        if psnr>= max_psnr:
            max_psnr=psnr
            best_model_state = model.state_dict() #Store the best model parameters.
        if args.step_if==True and step <=3100:
            scheduler.step()
        pbar.set_description(f'Loss: {loss.item():.4f} | PSNR: {psnr:.2f}')
        pbar.update(1)
        final_psnr=psnr
        step_end_time=time.time()
        iteration_time=step_end_time-start_time
        records.append((iteration_time, psnr))
    final_model_state=model.state_dict()
df = pd.DataFrame(records, columns=['Time (s)', 'PSNR'])
if args.gradient_adjust:
    name=args.photo_name+'_'+args.mode+'_'+str(args.hidden_layers+1)+'layers_IGA_xuhao_'+str(args.xuhao)+'_'+str(args.replace_end)+'_'+str(args.patch_size)+'_psnr_'+str(max_psnr)
else:
    name=args.photo_name+'_'+args.mode+'_'+str(args.hidden_layers+1)+'layers_psnr_'+str(max_psnr)+'_lr_'+str(args.learning_rate)
df.to_csv(name+"_psnr_log.csv", index=False)
end_time = time.time()  

print('Current learning rate: {}'.format(scheduler.get_last_lr()),'max_psnr',max_psnr)
str_max_psnr=f"{max_psnr:.3f}"
if args.gradient_adjust:
    name=args.photo_name+'_'+args.mode+'_'+str(args.hidden_layers+1)+'layers_IGA_xuhao_'+str(args.xuhao)+'_'+str(args.replace_end)+'_'+str(args.patch_size)+'_psnr_'+str_max_psnr
else:
    name=args.photo_name+'_'+args.mode+'_'+str(args.hidden_layers+1)+'layers_psnr_'+str(max_psnr)+'_lr_'+str(args.learning_rate)

if best_model_state is not None:
    torch.save(best_model_state, 'models/best_model_'+name+'.pth')

if args.saving:
    model.load_state_dict(best_model_state)
    model.eval()
    # Inference the photo.
    #for bn
    best_photo=model(photo.origin_coords.cuda().view(-1,2)).cpu().view(photo.H,photo.W,3).detach().numpy() 
    # best_photo=model(photo.origin_coords.cuda()).cpu().view(photo.H,photo.W,3).detach().numpy()
    best_photo=np.clip(best_photo,0,1)
    best_photo = (best_photo * 255).astype(np.uint8)  
    best_photo = Image.fromarray(best_photo)
    best_photo.save('results/'+name+'.png')