import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math
import numpy.random as rn
from torch.utils.data import DataLoader, Dataset
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utlis
from utlis import compute_kernel_matrix, calculate_transformation_matrix, weighted_loss,compute_NTK,compute_NTK_IGA,weighted_loss_IGA
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
import argparse

# fix the random seed
def seed_torch(seed): 
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed) 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# Define a toy network which have been introduced in Section 5: Experiment 1 of our Paper.
class SimpleNN(nn.Module):
    def __init__(self, m):
        super(SimpleNN, self).__init__()
        self.m = m
        self.linear1 = nn.Linear(2, m, bias=True)
        self.activation = nn.ReLU()  
        self.linear2 = nn.Linear(m, 1, bias=False)
        self.init_weights()

    def init_weights(self):
        # Follow the initialization scheme of Paper: 
        # "The Convergence Rate of Neural Networks for Learned Functions of Different Frequencies"
        nn.init.normal_(self.linear1.weight, mean=0, std=1)
        nn.init.zeros_(self.linear1.bias)
        shape = self.linear2.weight.shape
        indices = torch.randint(0, 2, shape, dtype=torch.long)
        # As a_r ∼ Uniform{−1, 1}
        self.linear2.weight.data = torch.where(indices == 0, torch.tensor(-1.0), torch.tensor(1.0))
        # a_r are fixed. 
        self.linear2.weight.requires_grad = False

    def forward(self, x):
        # As we have bias, we need to divide by the square root of 2 before activation.
        # The detailed explanation can be found in the Section 5 of our paper.
        x = (1/torch.sqrt(torch.tensor(2.0)))*self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)/(torch.sqrt(torch.tensor(self.m)))
        return x


# This class references the open-source code from previous work:
# https://github.com/ykasten/Convergence-Rate-NN-Different-Frequencies
class AroraDataset(Dataset):
    def __init__(self, cfg):
        n = cfg['n_train']
        self.x, self.theta = self.gen_x_arora(n)
        self.y = self.gen_y_arora(self.theta, cfg['ks'])
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x, self.y

    def gen_x_arora(self, n):
        theta = np.linspace(0, 2 * np.pi, n)
        x = np.stack([np.cos(theta), np.sin(theta)]).T
        return x, theta
    def gen_y_arora(self, theta, ks):
        y = np.zeros(len(theta))
        # adopt the frequence k in the pre-defined frequence set ks. 
        for k in ks:
            y += np.sin(k*2*math.pi* theta)
        return y

parser = argparse.ArgumentParser(description='Experiment_1')
parser.add_argument('--name', type=str, default='test',help='save_name')
parser.add_argument('--k_of_target', type=float, nargs='+',help='frequency list')
parser.add_argument('--optimizer', type=str, default='adam',help='optimizer')
parser.add_argument('--width', type=int, default=256, help='number of hidden neurons') 
parser.add_argument('--epochs', type=int, default=20000)
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--estimation',type=str,help='ntk,entk') # two estimation methods: ntk and entk of linear dynmaics models
parser.add_argument('--enable-adjust',action='store_true', help='Enable the adjust') # open the gradient adjustment
parser.add_argument('--disable-adjust', action='store_false', dest='enable_adjust', help='Disable the adjust') # close the gradient adjustment
parser.add_argument('--IGA',type=int,default=8, help='the interval length of Indcutive Gradient Adjustment (IGA)') 
parser.add_argument('--enable-IGA', action='store_true', help='Adopt the Indcutive Gradient Adjustment')
parser.add_argument('--disable-IGA', action='store_false', dest='enable_IGA', help='Disable the IGA')
parser.add_argument('--N', type=int, default=1024, help='sampling number or the size of data set')
parser.add_argument('--xuhao',type=int,default=15, help='The indices of the eigenvalues used for replacement.')
parser.add_argument('--replace_start',type=int,default=0, help='start, which can be found in the Equation (5) of our Paper.')
parser.add_argument('--replace_end',type=int,default=15, help='end, which can be found in the Equation (5) of our Paper.')
parser.add_argument('--i_freq',type=int,default=100,help='the interval for printing training')
args = parser.parse_args()

# cfg: capture the information of dataset.
cfg = {'n_train': args.N,'ks': args.k_of_target,  'gen_x_func': 'gen_x_arora','gen_y_func': 'gen_y_arora'}
train_dataset = AroraDataset(cfg)
dataloader=DataLoader(train_dataset,batch_size=1,pin_memory=True,num_workers=1)
print(args)

# ten random seeds used for paper. 
# seed_list=[16,32,64,128,256,512,1024,2048,4096,8192]
seed_list=[16,32]
# Initialize a dictionary to store training information under different random seeds.
all_losses = {f'loss_seed{seed_idx}': [] for seed_idx in range(len(seed_list))}
for freq_idx in range(len(args.k_of_target)):
    all_losses.update({f'freq{freq_idx+1}_seed{seed_idx}': [] for seed_idx in range(len(seed_list))})

for seed_idx, seed in enumerate(seed_list):
    print(seed_idx)
    # fix the random seed
    seed_torch(seed) 

    # load data
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input[0,:,:].cuda(), ground_truth[0,:].unsqueeze(1).cuda()

    # Initialize the network, loss function, and optimizer.
    model = SimpleNN(m=args.width).cuda()
    criterion = nn.MSELoss()
    xuhao=args.xuhao
    replace_start=args.replace_start
    replace_end=args.replace_end
    K=compute_kernel_matrix(model_input)
    S=calculate_transformation_matrix(K,xuhao,replace_start,replace_end)

    # validate on different optimizers.
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
        if args.enable_adjust: #adjust gradient
            if args.estimation=='ntk': #accurate ntk
                loss = criterion(outputs, ground_truth)
                weight_loss=weighted_loss(ground_truth,outputs,S)
                weight_loss.backward()
            if args.estimation=='entk':
                if args.enable_IGA:
                    loss = criterion(outputs, ground_truth)
                    # compute the dynamics model of sampled set defined by args.IGA
                    K_t=compute_NTK_IGA(model_input,model,args.IGA)
                    # compute the trasnforamtion_matrix
                    S_t=calculate_transformation_matrix(K_t,xuhao,replace_start,replace_end)
                    # Adjust the loss function to apply modifications.
                    weight_loss=weighted_loss_IGA(ground_truth,outputs,S_t,args.IGA)
                    # Get the adjusted gradient
                    weight_loss.backward()
                else:
                    # no iga, i.e., dynamics model of whole data set.
                    loss = criterion(outputs, ground_truth)
                    K_t=compute_NTK(model_input,model)
                    S_t=calculate_transformation_matrix(K_t,xuhao,replace_start,replace_end)
                    weight_loss=weighted_loss(ground_truth,outputs,S_t)
                    weight_loss.backward()
        else: #original gradient
            loss = criterion(outputs, ground_truth)
            loss.backward() 
        optimizer.step()
    
        if (epoch+1) % 1 == 0:
            with torch.no_grad():
                predictions = model(model_input).cpu().numpy().flatten()
                fft_preds=np.fft.rfft(predictions)
                fft_gt=np.fft.rfft(ground_truth.cpu().numpy().flatten())
                sorted_indices=np.argsort(np.abs(fft_gt))
                top_indices=sorted_indices[-len(args.k_of_target):][::-1]
                freq_indices=np.sort(top_indices)
                theta = np.linspace(0, 2 * np.pi, cfg['n_train'])
                freqs=np.fft.rfftfreq(cfg['n_train'],d=(theta[1]-theta[0])) 
                percentage=np.abs(fft_gt-fft_preds)/np.abs(fft_gt) #Select the corresponding frequencies using freq_indices
                percentage=percentage[freq_indices]
                # # Record the loss and frequency error for each seed
                all_losses[f'loss_seed{seed_idx}'].append(loss.item())
                for i, p in enumerate(percentage):
                    all_losses[f'freq{i+1}_seed{seed_idx}'].append(p.item())
        
        # print information of training
        if (epoch+1) % args.i_freq == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.8f}')
            
losses_df = pd.DataFrame(all_losses)
losses_df.to_csv(f'/data0/home/shikexuan/IGA-INR/simple_function_approximation/experiment1/all_seeds/{args.name}_all_seeds.csv', index=False)
# save parameter information
args_dict = vars(args)
args_list = [(key, str(value)) for key, value in args_dict.items()]
args_array = np.array(args_list, dtype=str)
np.savetxt(f'IGA-INR/simple_function_approximation/experiment1/all_seeds/{args.name}.txt', args_array, fmt='%s', delimiter=',', header='Argument,Value', comments='')

