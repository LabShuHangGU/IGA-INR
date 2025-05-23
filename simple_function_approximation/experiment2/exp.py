import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
import math
import numpy.random as rn
from torch.utils.data import DataLoader, Dataset
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utlis
from utlis import compute_kernel_matrix, calculate_precondition_matrix, weighted_loss,compute_NTK,compute_NTK_IGA,weighted_loss_IGA
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
import argparse

start_time=time.time()
parser = argparse.ArgumentParser(description='Your program description')
parser.add_argument('--name', type=str, default='test',help='save_name')
parser.add_argument('--k_of_target', type=float, nargs='+',help='frequency list')
parser.add_argument('--optimizer', type=str, default='adam',help='optimizer')
parser.add_argument('--mode', type=str, default='relu',help='save_name')
parser.add_argument('--in_features', type=int, default=1)
parser.add_argument('--hidden_features', type=int, default=256)
parser.add_argument('--hidden_layers', type=int, default=3)
parser.add_argument('--out_features', type=int, default=1)
parser.add_argument('--epochs', type=int, default=20000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gradient_adjust',action='store_true',default=False)
parser.add_argument('--IGA',type=int,default=8)
parser.add_argument('--enable_IGA', action='store_true', help='Enable the IGA')
parser.add_argument('--N', type=int, default=1024)
parser.add_argument('--xuhao',type=int,default=15)
parser.add_argument('--replace_start',type=int,default=0)
parser.add_argument('--replace_end',type=int,default=15)
parser.add_argument('--i_freq',type=int,default=100,help='the freq of print')
parser.add_argument('--min_val',type=float,default=-1.0, help='The left side of the data sampling interval')
parser.add_argument('--max_val',type=float,default=1.0,  help='The right side of the data sampling interval')
args = parser.parse_args()
print(args)

def seed_torch(seed): 
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class INR(nn.Module):
    def __init__(self,mode,in_features,hidden_features,hidden_layers,out_features,outermost_linear=True,first_omega_0=5.0,hidden_omega_0=5.0):
        super().__init__()
        self.net=[]
        self.mode=mode

        self.net=[]

        if mode=='relu':
            self.net.append(nn.Linear(in_features,hidden_features))
            self.net.append(nn.ReLU())
            for i in range(hidden_layers):
                self.net.append(nn.Linear(hidden_features,hidden_features))
                self.net.append(nn.ReLU())

        if mode=='sin':
            self.net.append(SineLayer(in_features, hidden_features,is_first=True, omega_0=first_omega_0))
            for i in range(hidden_layers):
                self.net.append(SineLayer(hidden_features, hidden_features,is_first=False, omega_0=hidden_omega_0))
        
        #末端初始化这边还是需要修改
        if outermost_linear==True:
            final_linear = nn.Linear(hidden_features, out_features)
            if mode=='sin':
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6/ hidden_features)/hidden_omega_0,np.sqrt(6 / hidden_features)/hidden_omega_0)
            else:
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6/ hidden_features),np.sqrt(6 / hidden_features)) 
            self.net.append(final_linear)
        else:
            if mode=='relu':
                final_linear=nn.Linear(hidden_features,out_features)
                self.net.append(final_linear)
                self.net.append(nn.ReLU())
            if mode=='sin':
                self.net.append(SineLayer(hidden_features, out_features,is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        output=self.net(x)
        return output

class CustomDataset(Dataset):
    def __init__(self,cfg):
        cfg
        self.num_samples = cfg['n_train']
        self.x=torch.linspace(cfg['min_val'],cfg['max_val'],cfg['n_train'])
        self.x=torch.reshape(self.x,(cfg['n_train'],1))
        self.y=torch.zeros((cfg['n_train'],1))
        for k in cfg['ks']:
            self.y+=torch.sin(k*2*math.pi*self.x)
    def __len__(self):
        return self.num_samples

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

cfg = {
    'ks': args.k_of_target, 
    'n_train': args.N,
    'min_val': args.min_val,
    'max_val':args.max_val
}

# 创建训练数据集
train_dataset = CustomDataset(cfg)
dataloader=DataLoader(train_dataset,batch_size=cfg['n_train'],pin_memory=True,num_workers=1)

# seed_list=[16,32,64,128,256,512,1024,2048,4096,8192]
seed_list=[16,32]
all_losses={f'loss_seed{seed_idx}': [] for seed_idx in range(len(seed_list))}

for freq_idx in range(len(args.k_of_target)):
    all_losses.update({f'freq{freq_idx+1}_seed{seed_idx}': [] for seed_idx in range(len(seed_list))})

print(args.name)
for seed_idx,seed in enumerate(seed_list):
    print(seed_idx)
    seed_torch(seed=seed) #该函数一般放在main()函数开头第一行进行固定种子最好
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    model = INR(args.mode,args.in_features,args.hidden_features,args.hidden_layers,args.out_features).cuda()
    criterion = nn.MSELoss()
    xuhao=args.xuhao
    replace_start=args.replace_start
    replace_end=args.replace_end

    if args.optimizer=='adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer=optim.SGD(model.parameters(),lr=args.lr)

    losses = []
    state_list = [] 
    grad_list=[]
    preds_freqs_all_epochs=[]

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(model_input) #[1024,1]
        if args.gradient_adjust: #adjust gradient
            if args.enable_IGA:
                loss = criterion(outputs, ground_truth)
                K_t=compute_NTK_IGA(model_input,model,args.IGA)
                S_t=calculate_precondition_matrix(K_t,xuhao,replace_start,replace_end)
                weight_loss=weighted_loss_IGA(ground_truth,outputs,S_t,args.IGA)
                weight_loss.backward()
            else:
                loss = criterion(outputs, ground_truth)
                K_t=compute_NTK(model_input,model)
                S_t=calculate_precondition_matrix(K_t,xuhao,replace_start,replace_end)
                weight_loss=weighted_loss(ground_truth,outputs,S_t)
                weight_loss.backward()
        else: #original gradient
            loss = criterion(outputs, ground_truth)
            loss.backward() 
        optimizer.step()

        if epoch+1 ==1 or (epoch+1) % args.i_freq == 0:
            with torch.no_grad():
                predictions = model(model_input).cpu().numpy().flatten()
                fft_preds=np.fft.rfft(predictions)
                fft_gt=np.fft.rfft(ground_truth.cpu().numpy().flatten())
                sorted_indices=np.argsort(np.abs(fft_gt))
                fft_data=np.abs(fft_gt)
                pd.DataFrame(fft_data).to_csv('fft_results.csv', index=False)
                top_indices=sorted_indices[-len(args.k_of_target):][::-1]
                freq_indices=np.sort(top_indices)
                theta = np.linspace(0, 2 * np.pi, cfg['n_train'])
                freqs=np.fft.rfftfreq(cfg['n_train'],d=(theta[1]-theta[0])) 
                percentage=np.abs(fft_gt-fft_preds)/np.abs(fft_gt) 
                percentage=percentage[freq_indices]
                all_losses[f'loss_seed{seed_idx}'].append(loss.item())
                for i, p in enumerate(percentage):
                    all_losses[f'freq{i+1}_seed{seed_idx}'].append(p.item())
        if (epoch+1) % args.i_freq == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.8f}')
    end_time=time.time()
    print('avg time',(end_time-start_time)/args.epochs)
        
losses_df = pd.DataFrame(all_losses)
losses_df.to_csv(f'/data0/home/shikexuan/IGA-INR/simple_function_approximation/experiment2/all_seeds/{args.name}_all_seeds.csv', index=False)
# 保存参数信息
args_dict = vars(args)
args_list = [(key, str(value)) for key, value in args_dict.items()]
args_array = np.array(args_list, dtype=str)
np.savetxt(f'/data0/home/shikexuan/IGA-INR/simple_function_approximation/experiment2/all_seeds/{args.name}.txt', args_array, fmt='%s', delimiter=',', header='Argument,Value', comments='')

