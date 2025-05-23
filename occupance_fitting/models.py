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
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
import math
import lpips
from torch.nn import init
import warnings

class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)

class Fourier_reparam_linear(nn.Module):
    def __init__(self,in_features,out_features,high_freq_num,low_freq_num,phi_num,alpha):
        super(Fourier_reparam_linear,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.high_freq_num =high_freq_num
        self.low_freq_num = low_freq_num
        self.phi_num = phi_num
        self.alpha=alpha
        self.bases=self.init_bases()
        self.lamb=self.init_lamb()
        self.bias=nn.Parameter(torch.Tensor(self.out_features,1),requires_grad=True)
        self.init_bias()

    def init_bases(self):
        phi_set=np.array([2*math.pi*i/self.phi_num for i in range(self.phi_num)])
        high_freq=np.array([i+1 for i in range(self.high_freq_num)])
        low_freq=np.array([(i+1)/self.low_freq_num for i in range(self.low_freq_num)])
        if len(low_freq)!=0:
            T_max=2*math.pi/low_freq[0]
        else:
            T_max=2*math.pi/min(high_freq) 
        points=np.linspace(-T_max/2,T_max/2,self.in_features)
        bases=torch.Tensor((self.high_freq_num+self.low_freq_num)*self.phi_num,self.in_features)
        i=0
        for freq in low_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        for freq in high_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        bases=self.alpha*bases
        bases=nn.Parameter(bases,requires_grad=False)
        return bases

    
    def init_lamb(self):
        self.lamb=torch.Tensor(self.out_features,(self.high_freq_num+self.low_freq_num)*self.phi_num)
        with torch.no_grad():
            m=(self.low_freq_num+self.high_freq_num)*self.phi_num
            for i in range(m):
                dominator=torch.norm(self.bases[i,:],p=2)
                self.lamb[:,i]=nn.init.uniform_(self.lamb[:,i],-np.sqrt(6/m)/dominator,np.sqrt(6/m)/dominator)
        self.lamb=nn.Parameter(self.lamb,requires_grad=True)
        return self.lamb

    def init_bias(self):
        with torch.no_grad():
            nn.init.zeros_(self.bias)
        
    def forward(self,x):
        weight=torch.matmul(self.lamb,self.bases)
        output=torch.matmul(x,weight.transpose(0,1))
        output=output+self.bias.T
        return output

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
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class sin_fr_layer(nn.Module):
    def __init__(self, in_features, out_features, high_freq_num,low_freq_num,phi_num,alpha,omega_0=30.0):
        super().__init__()
        super(sin_fr_layer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.high_freq_num =high_freq_num
        self.low_freq_num = low_freq_num
        self.phi_num = phi_num
        self.alpha=alpha
        self.omega_0=omega_0
        self.bases=self.init_bases()
        self.lamb=self.init_lamb()
        self.bias=nn.Parameter(torch.Tensor(self.out_features,1),requires_grad=True)
        self.init_bias()

    def init_bases(self):
        phi_set=np.array([2*math.pi*i/self.phi_num for i in range(self.phi_num)])
        high_freq=np.array([i+1 for i in range(self.high_freq_num)])
        low_freq=np.array([(i+1)/self.low_freq_num for i in range(self.low_freq_num)])
        if len(low_freq)!=0:
            T_max=2*math.pi/low_freq[0]
        else:
            T_max=2*math.pi/min(high_freq) # 取最大周期作为取点区间
        points=np.linspace(-T_max/2,T_max/2,self.in_features)
        bases=torch.Tensor((self.high_freq_num+self.low_freq_num)*self.phi_num,self.in_features)
        i=0
        for freq in low_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        for freq in high_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        bases=self.alpha*bases
        bases=nn.Parameter(bases,requires_grad=False)
        return bases

    
    def init_lamb(self):
        self.lamb=torch.Tensor(self.out_features,(self.high_freq_num+self.low_freq_num)*self.phi_num)
        with torch.no_grad():
            m=(self.low_freq_num+self.high_freq_num)*self.phi_num
            for i in range(m):
                dominator=torch.norm(self.bases[i,:],p=2)
                self.lamb[:,i]=nn.init.uniform_(self.lamb[:,i],-np.sqrt(6/m)/dominator/self.omega_0,np.sqrt(6/m)/dominator/self.omega_0)
        self.lamb=nn.Parameter(self.lamb,requires_grad=True)
        return self.lamb

    def init_bias(self):
        with torch.no_grad():
            nn.init.zeros_(self.bias)
        
    def forward(self,x):
        weight=torch.matmul(self.lamb,self.bases)
        output=torch.matmul(x,weight.transpose(0,1))
        output=output+self.bias.T
        return torch.sin(self.omega_0*output)

class SineLayer_bn(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm=nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.norm(self.omega_0 * self.linear(input)))

class INR(nn.Module):
    def __init__(self,mode,in_features,hidden_features,hidden_layers,out_features,outermost_linear,high_freq_num,low_freq_num,
    phi_num,alpha,first_omega_0,hidden_omega_0):
        super().__init__()
        self.net=[]
        self.mode=mode
        if 'pe' in mode:
            self.positional_encoding = PosEncodingNeRF(in_features=in_features,sidelength=256,fn_samples=None,use_nyquist=True)
            in_features=self.positional_encoding.out_dim
        self.net=[]
        # if mode=='finer':
        #     self.net.append(FinerLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, first_bias_scale=None, scale_req_grad=False))
        #     for i in range(hidden_layers):
        #         self.net.append(FinerLayer(hidden_features, hidden_features, omega_0=hidden_omega_0, scale_req_grad=False))

        if mode=='relu':
            self.net.append(nn.Linear(in_features,hidden_features))
            self.net.append(nn.ReLU())
            for i in range(hidden_layers):
                self.net.append(nn.Linear(hidden_features,hidden_features))
                self.net.append(nn.ReLU())
        
        if mode=='relu+fr':
            self.net.append(nn.Linear(in_features,hidden_features))
            self.net.append(nn.ReLU())
            for i in range(hidden_layers):
                self.net.append(Fourier_reparam_linear(hidden_features,hidden_features,high_freq_num,low_freq_num,phi_num,alpha))
                self.net.append(nn.ReLU())
        
        if mode=='relu+pe':
            self.net.append(nn.Linear(in_features,hidden_features))
            self.net.append(nn.ReLU())
            for i in range(hidden_layers):
                self.net.append(nn.Linear(hidden_features,hidden_features))
                self.net.append(nn.ReLU())
        
        if mode=='relu+pe+fr':
            self.net.append(nn.Linear(in_features,hidden_features))
            self.net.append(nn.ReLU())
            for i in range(hidden_layers):
                self.net.append(Fourier_reparam_linear(hidden_features,hidden_features,high_freq_num,low_freq_num,phi_num,alpha))
                self.net.append(nn.ReLU())

        if mode=='sin':
            self.net.append(SineLayer(in_features, hidden_features,is_first=True, omega_0=first_omega_0))
            for i in range(hidden_layers):
                self.net.append(SineLayer(hidden_features, hidden_features,is_first=False, omega_0=hidden_omega_0))
        
        # Follow the latest version of the open source code from Paper: Batch Normalization Alleviates the Spectral Bias in Coordinate Networks.
        # In occupance tasks, we found that BN after activation is better than BN before activation.
        if mode=='relu+bn':
            self.net.append(nn.Linear(in_features,hidden_features))
            self.net.append(nn.ReLU())
            self.net.append(nn.BatchNorm1d(hidden_features))
            for i in range(hidden_layers):
                self.net.append(nn.Linear(hidden_features,hidden_features))
                self.net.append(nn.ReLU())
                self.net.append(nn.BatchNorm1d(hidden_features))
        
        if mode=='relu+pe+bn':
            self.net.append(nn.Linear(in_features,hidden_features))
            self.net.append(nn.BatchNorm1d(hidden_features))
            self.net.append(nn.ReLU())

            for i in range(hidden_layers):
                self.net.append(nn.Linear(hidden_features,hidden_features))
                self.net.append(nn.BatchNorm1d(hidden_features))
                self.net.append(nn.ReLU())

        
        if mode=='sin+bn':
            self.net.append(SineLayer_bn(in_features, hidden_features,is_first=True, omega_0=first_omega_0))
            for i in range(hidden_layers):
                self.net.append(SineLayer_bn(hidden_features, hidden_features,is_first=False, omega_0=hidden_omega_0))
        
        if mode=='sin+fr':
            self.net.append(SineLayer(in_features, hidden_features,is_first=True, omega_0=first_omega_0))
            for i in range(hidden_layers):
                self.net.append(sin_fr_layer(hidden_features,hidden_features,high_freq_num,low_freq_num,phi_num,alpha,hidden_omega_0))

        if outermost_linear==True:
            final_linear = nn.Linear(hidden_features, out_features)
            if 'sin' in mode or 'finer' in mode:
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6/ hidden_features)/hidden_omega_0,np.sqrt(6 / hidden_features)/hidden_omega_0)
            else:
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6/ hidden_features),np.sqrt(6 / hidden_features)) 
            self.net.append(final_linear)
        else:
            if 'relu' in mode:
                final_linear=nn.Linear(hidden_features,out_features)
                self.net.append(final_linear)
                self.net.append(nn.ReLU())
            else:
                self.net.append(SineLayer(hidden_features, out_features,is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        if 'bn' in self.mode:
            x=torch.squeeze(x)
        if 'pe' in self.mode:
            x=self.positional_encoding(x)
            x=torch.squeeze(x) # for bn
        output=self.net(x)
        return output
class GaborLayer(nn.Module):
    def __init__(self, in_dim, out_dim, padding, alpha, beta=1.0, bias=False):
        super(GaborLayer, self).__init__()

        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim, )))
        self.linear = torch.nn.Linear(in_dim, out_dim)
        #self.padding = padding

        self.linear.weight.data *= 128. * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)

        # Bias parameters start in zeros
        #self.bias = nn.Parameter(torch.zeros(self.responses)) if bias else None

    def forward(self, input):
        return torch.sin(self.linear(input))
class mfn_INR(nn.Module):
    def __init__(self, in_features=2, hidden_features=256,
                 hidden_layers=4, out_features=1, 
                 outermost_linear=True, first_omega_0=0,
                 hidden_omega_0=0, scale=1, pos_encode=False,
                 sidelength=1, fn_samples=None, use_nyquist=None):
        super(mfn_INR, self).__init__()

        self.k = hidden_layers+1
        self.gabon_filters = nn.ModuleList([GaborLayer(in_features, hidden_features, 0, alpha=6.0 / self.k) for _ in range(self.k)])
        self.linear = nn.ModuleList(
            [torch.nn.Linear(hidden_features, hidden_features) for _ in range(self.k - 1)] + [torch.nn.Linear(hidden_features, out_features)])

        for lin in self.linear[:self.k - 1]:
            lin.weight.data.uniform_(-np.sqrt(1.0 / hidden_features), np.sqrt(1.0 / hidden_features))

    def forward(self, x):

        # Recursion - Equation 3
        zi = self.gabon_filters[0](x)  # Eq 3.a
        for i in range(self.k - 1):
            zi = self.linear[i](zi) * self.gabon_filters[i + 1](x)
            # Eq 3.b

        return self.linear[self.k - 1](zi)[None, ...]  # Eq 3.c


# Init weights for Finer, Finer+Gauss, Finer+WIRE
def init_weights(m, omega=1, c=1, is_first=False): # Default: Pytorch initialization
    if hasattr(m, 'weight'):
        fan_in = m.weight.size(-1)
        if is_first:
            bound = 1 / fan_in # SIREN
        else:
            bound = math.sqrt(c / fan_in) / omega
        init.uniform_(m.weight, -bound, bound)
    
def init_weights_kaiming(m):
    if hasattr(m, 'weight'):
        init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

def init_bias(m, k):
    if hasattr(m, 'bias'):
        init.uniform_(m.bias, -k, k)

'''Used for SIREN, FINER, Gauss, Wire, etc.'''
def init_weights_cond(init_method, linear, omega=1, c=1, is_first=False):
    init_method = init_method.lower()
    if init_method == 'sine':
        init_weights(linear, omega, 6, is_first)    # SIREN initialization
    ## Default: Pytorch initialization

def init_bias_cond(linear, fbs=None, is_first=True):
    if is_first and fbs != None:
        init_bias(linear, fbs)
    ## Default: Pytorch initialization
def generate_alpha(x, alphaType=None, alphaReqGrad=False):
    """
    if alphaType == ...:
        return ...
    """
    with torch.no_grad():
        return torch.abs(x) + 1
def finer_activation(x, omega=1, alphaType=None, alphaReqGrad=False):
    return torch.sin(omega * generate_alpha(x, alphaType, alphaReqGrad) * x)

'''
    Gauss. & GF(FINER++Gauss.) activation
'''
def wire_activation(x, scale, omega_w):
    # return torch.exp(1j*omega_w*x - torch.abs(scale*x)**2)
    return torch.cos(omega_w*x)*torch.exp(- torch.abs(scale*x)**2)
def gauss_activation(x, scale):
    return torch.exp(-(scale*x)**2)

def gauss_finer_activation(x, scale, omega, alphaType=None, alphaReqGrad=False):
    return gauss_activation(finer_activation(x, omega, alphaType, alphaReqGrad), scale)

def wire_finer_activation(x, scale, omega_w, omega, alphaType=None, alphaReqGrad=False):
    if x.is_complex():
        return wire_activation(finer_activation_complex_sep_real_imag(x, omega), scale, omega_w)
    else:
        return wire_activation(finer_activation(x, omega), scale, omega_w)

class GFLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=3, omega=1,
                 is_first=False, is_last=False, 
                 init_method='Pytorch', init_gain=1, fbs=None, hbs=None,
                 alphaType=None, alphaReqGrad=False,
                 norm_activation=False):
        super().__init__()
        self.scale = scale
        self.omega = omega
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm_activation = norm_activation
        
        # init weights
        init_weights_cond(init_method, self.linear, omega, init_gain, is_first)
            
        # init bias 
        init_bias_cond(self.linear, fbs, is_first)
    
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return gauss_finer_activation(wx_b, self.scale, self.omega)
        return wx_b # is_last==True

class GFbnLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=3, omega=1,
                 is_first=False, is_last=False, 
                 init_method='Pytorch', init_gain=1, fbs=None, hbs=None,
                 alphaType=None, alphaReqGrad=False,
                 norm_activation=False):
        super().__init__()
        self.scale = scale
        self.omega = omega
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.batchnorm=nn.BatchNorm1d(out_features)
        self.norm_activation = norm_activation
        
        # init weights
        init_weights_cond(init_method, self.linear, omega, init_gain, is_first)
            
        # init bias 
        init_bias_cond(self.linear, fbs, is_first)
    
    def forward(self, input):
        wx_b = self.linear(input) 
        wx_b=self.batchnorm(wx_b)
        if not self.is_last:
            return gauss_finer_activation(wx_b, self.scale, self.omega)
        return wx_b # is_last==True
class GFfrLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=3, omega=1,
                 is_first=False, is_last=False, 
                 init_method='Pytorch', init_gain=1, fbs=None, hbs=None,
                 alphaType=None, alphaReqGrad=False,
                 norm_activation=False):
        super().__init__()
        self.scale = scale
        self.omega = omega
        self.is_last = is_last
        self.linear = Fourier_reparam_linear(in_features, out_features, 32,32,8,0.01)
        self.norm_activation = norm_activation
        
        # init weights
        init_weights_cond(init_method, self.linear, omega, init_gain, is_first)
            
        # init bias 
        init_bias_cond(self.linear, fbs, is_first)
    
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return gauss_finer_activation(wx_b, self.scale, self.omega)
        return wx_b # is_last==True


# Gauss_Finer
class GF(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, hidden_features, 
                 scale=3, omega=10, 
                 init_method='sine', init_gain=1, fbs=1, hbs=None, 
                 alphaType=None, alphaReqGrad=False,
                 norm_activation=False):
        super().__init__()
        self.net = []
        self.net.append(GFLayer(in_features, hidden_features, is_first=True, 
                                scale=scale, omega=omega, 
                                init_method=init_method, init_gain=init_gain, fbs=fbs,
                                alphaType=alphaType, alphaReqGrad=alphaReqGrad, 
                                norm_activation=norm_activation))
        
        for i in range(hidden_layers):
            self.net.append(GFLayer(hidden_features, hidden_features, 
                                     scale=scale, omega=omega, 
                                     init_method=init_method, init_gain=init_gain, hbs=hbs,
                                     alphaType=alphaType, alphaReqGrad=alphaReqGrad,
                                     norm_activation=norm_activation))
         
        self.net.append(GFLayer(hidden_features, out_features, is_last=True, 
                                scale=scale, omega=omega, 
                                init_method=init_method, init_gain=init_gain, hbs=hbs,
                                norm_activation=norm_activation)) # omega: For weight init
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output

# GF_BN
class GF_BN(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, hidden_features, 
                 scale=3, omega=10, 
                 init_method='sine', init_gain=1, fbs=1, hbs=None, 
                 alphaType=None, alphaReqGrad=False,
                 norm_activation=False):
        super().__init__()
        self.net = []
        self.net.append(GFbnLayer(in_features, hidden_features, is_first=True, 
                                scale=scale, omega=omega, 
                                init_method=init_method, init_gain=init_gain, fbs=fbs,
                                alphaType=alphaType, alphaReqGrad=alphaReqGrad, 
                                norm_activation=norm_activation))
        # self.net.append(nn.BatchNorm1d(hidden_features))
        
        for i in range(hidden_layers):
            self.net.append(GFbnLayer(hidden_features, hidden_features, 
                                     scale=scale, omega=omega, 
                                     init_method=init_method, init_gain=init_gain, hbs=hbs,
                                     alphaType=alphaType, alphaReqGrad=alphaReqGrad,
                                     norm_activation=norm_activation))
            # self.net.append(nn.BatchNorm1d(hidden_features))
         
        self.net.append(GFLayer(hidden_features, out_features, is_last=True, 
                                scale=scale, omega=omega, 
                                init_method=init_method, init_gain=init_gain, hbs=hbs,
                                norm_activation=norm_activation)) # omega: For weight init
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output

# WF_BN
class WF_BN(nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_layers=3, hidden_features=256, 
                 scale=2, omega_w=4, omega=5,
                 init_method='sine', init_gain=1, fbs=1, hbs=None, 
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        hidden_features = int(hidden_features / np.sqrt(2))
        
        self.net = []
        self.net.append(WFLayer(in_features, hidden_features, is_first=True,
                                    omega=omega, scale=scale, omega_w=omega_w, 
                                    init_method=init_method, init_gain=init_gain, fbs=fbs,
                                    alphaType=alphaType, alphaReqGrad=alphaReqGrad))
        self.net.append(nn.BatchNorm1d(hidden_features))

        for i in range(hidden_layers):
            self.net.append(WFLayer(hidden_features, hidden_features, 
                                        omega=omega, scale=scale, omega_w=omega_w,
                                        init_method=init_method, init_gain=init_gain, hbs=hbs,
                                        alphaType=alphaType, alphaReqGrad=alphaReqGrad))
            self.net.append(nn.BatchNorm1d(hidden_features))

        self.net.append(WFLayer(hidden_features, out_features, is_last=True, 
                                    omega=omega, scale=scale, omega_w=omega_w,
                                    init_method=init_method, init_gain=init_gain, hbs=hbs)) # omega: For weight init
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output.real

# Gauss_Finer_fr
class GF_fr(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, hidden_features, 
                 scale=3, omega=10, 
                 init_method='sine', init_gain=1, fbs=1, hbs=None, 
                 alphaType=None, alphaReqGrad=False,
                 norm_activation=False):
        super().__init__()
        self.net = []
        self.net.append(GFLayer(in_features, hidden_features, is_first=True, 
                                scale=scale, omega=omega, 
                                init_method=init_method, init_gain=init_gain, fbs=fbs,
                                alphaType=alphaType, alphaReqGrad=alphaReqGrad, 
                                norm_activation=norm_activation))
        
        for i in range(hidden_layers):
            self.net.append(GFfrLayer(hidden_features, hidden_features, 
                                     scale=scale, omega=omega, 
                                     init_method=init_method, init_gain=init_gain, hbs=hbs,
                                     alphaType=alphaType, alphaReqGrad=alphaReqGrad,
                                     norm_activation=norm_activation))
         
        self.net.append(GFLayer(hidden_features, out_features, is_last=True, 
                                scale=scale, omega=omega, 
                                init_method=init_method, init_gain=init_gain, hbs=hbs,
                                norm_activation=norm_activation)) # omega: For weight init
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output

class WFLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=10, omega_w=20, omega=1,
                 is_first=False, is_last=False, 
                 init_method='Pytorch', init_gain=1, fbs=None, hbs=None,
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.scale = scale
        self.omega_w = omega_w
        self.omega = omega
        self.is_last = is_last ## no activation
        # dtype = torch.float if is_first else torch.cfloat
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # init weights
        init_weights_cond(init_method, self.linear, omega*omega_w, init_gain, is_first)
        # init bias 
        init_bias_cond(self.linear, fbs, is_first)
        
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return wire_finer_activation(wx_b, self.scale, self.omega_w, self.omega)
        return wx_b # is_last==True

# WIRE_Finer
# --scale 2 --omega_w 4 --omega 5 --fbs 1 \
#     # --init_method sine \
class WF(nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_layers=3, hidden_features=256, 
                 scale=2, omega_w=4, omega=5,
                 init_method='sine', init_gain=1, fbs=1, hbs=None, 
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        hidden_features = int(hidden_features / np.sqrt(2))
        
        self.net = []
        self.net.append(WFLayer(in_features, hidden_features, is_first=True,
                                    omega=omega, scale=scale, omega_w=omega_w, 
                                    init_method=init_method, init_gain=init_gain, fbs=fbs,
                                    alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        for i in range(hidden_layers):
            self.net.append(WFLayer(hidden_features, hidden_features, 
                                        omega=omega, scale=scale, omega_w=omega_w,
                                        init_method=init_method, init_gain=init_gain, hbs=hbs,
                                        alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        self.net.append(WFLayer(hidden_features, out_features, is_last=True, 
                                    omega=omega, scale=scale, omega_w=omega_w,
                                    init_method=init_method, init_gain=init_gain, hbs=hbs)) # omega: For weight init
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output.real


## FINER 
class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega=30, 
                 is_first=False, is_last=False, 
                 init_method='sine', init_gain=1, fbs=1, hbs=1, 
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.omega = omega
        self.is_last = is_last ## no activation
        self.alphaType = alphaType
        self.alphaReqGrad = alphaReqGrad
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # init weights
        init_weights_cond(init_method, self.linear, omega, init_gain, is_first)
        # init bias
        init_bias_cond(self.linear, fbs, is_first)
    
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return finer_activation(wx_b, self.omega)
        return wx_b # is_last==True
      
class Finer(nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_layers=3, hidden_features=256, 
                 first_omega=30, hidden_omega=30, 
                 init_method='sine', init_gain=1, fbs=1, hbs=1, 
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.net = []
        self.net.append(FinerLayer(in_features, hidden_features, is_first=True, 
                                   omega=first_omega, 
                                   init_method=init_method, init_gain=init_gain, fbs=fbs,
                                   alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        for i in range(hidden_layers):
            self.net.append(FinerLayer(hidden_features, hidden_features, 
                                       omega=hidden_omega, 
                                       init_method=init_method, init_gain=init_gain, hbs=hbs,
                                       alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        self.net.append(FinerLayer(hidden_features, out_features, is_last=True, 
                                   omega=hidden_omega, 
                                   init_method=init_method, init_gain=init_gain, hbs=hbs)) # omega: For weight init
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)