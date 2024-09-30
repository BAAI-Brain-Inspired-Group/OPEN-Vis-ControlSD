import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import random

import torchvision

from einops import rearrange, repeat,reduce
from einops.layers.torch import Rearrange
# from utils import *


class Lgn_ende(nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.arg = arg
        self.channel = arg.channel
        self.n_vector = arg.n_vector
        self.vector_length = arg.vector_length
        self.lgn_kernel_size = arg.lgn_kernel_size
        self.lgn_stride = arg.lgn_stride
        self.lgn_padding = arg.lgn_padding
        self.activate_fn = arg.activate_fn

        self.conv = nn.Conv2d(self.channel,self.n_vector*self.vector_length,self.lgn_kernel_size,self.lgn_stride,self.lgn_padding,bias=False)
        # self.bn = nn.BatchNorm2d(arg.n_vector*arg.vector_length,momentum=0.1,affine=False)
        self.to_column = nn.Identity()
        self.to_channel = nn.Identity()
        if self.activate_fn is None:
            self.nonlinear = nn.Identity()
        elif self.activate_fn == 'gelu':
            self.nonlinear = nn.GELU()
        elif self.activate_fn == 'relu':
            self.nonlinear = nn.ReLU()
        # self.nonlinear = nn.Identity()

        # self.bn = nn.BatchNorm2d(arg.n_vector,momentum=0.1,affine=False)
        # self.to_column = Rearrange('b (nv vl) h w -> b nv (vl h) w',vl=arg.vector_length)
        # self.to_channel = Rearrange('b nv (vl h) w -> b (nv vl) h w',vl=arg.vector_length)
        self.bn = nn.Identity()
        
        # self.apply(weights_init)

    def forward(self,x):
        return self.nonlinear(self.to_channel(self.bn(self.to_column(self.conv(x)))))
        # return F.gelu(self.bn(self.conv(x)))
        # return self.conv(x)
    
    def deconv(self,f):
        return F.conv_transpose2d(f,self.conv.weight,stride=self.lgn_stride,padding=self.lgn_padding)
        # # print('Lgn_ende deconv running_var:',self.bn.running_var.size(),', running_mean:',self.bn.running_mean.size())
        # return F.conv_transpose2d(self.to_channel(self.to_column(f)*self.bn.running_var.view(1,-1,1,1)+self.bn.running_mean.view(1,-1,1,1)),self.conv.weight,stride=self.arg.lgn_stride,padding=self.arg.lgn_padding)

    def deconv_group(self,f):
        img_group = F.conv_transpose2d(f,self.conv.weight,stride=self.lgn_stride,padding=self.lgn_padding,groups=self.n_vector)
        img = reduce(img_group,'b (nv c) h w -> b c h w','sum',nv=self.n_vector)
        return img_group,img
    
    def fullconv(self,x):
        return self.nonlinear(self.to_channel(self.bn(self.to_column(F.conv2d(x,self.conv.weight,padding=self.lgn_padding)))))
    
    def fulldeconv(self,f):
        return F.conv_transpose2d(f,self.conv.weight,padding=self.lgn_padding)/self.lgn_stride**2
    
    def fulldeconv_group(self,f):
        return F.conv_transpose2d(f,self.conv.weight,padding=self.lgn_padding,groups=self.n_vector)/self.lgn_stride**2
    
    def conv_stride(self,x):
        return self.conv(x)
    
class Gonlin(nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.arg = arg
        self.chl = arg.channel
        self.nv = arg.n_vector
        self.vl = arg.vector_length
        self.eye = nn.Parameter(torch.eye(self.nv*self.vl).unsqueeze(2).unsqueeze(3), requires_grad=False)
        # self.eye = torch.eye(self.nv*self.vl).unsqueeze(2).unsqueeze(3)

class Bipolar(Gonlin):
    def __init__(self,arg):
        super().__init__(arg)
        self.c1_1 = nn.Conv2d(self.chl,self.nv,5,2,2,bias=False)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
        self.c1_2_1 = nn.Conv2d(self.chl,self.nv,5,2,2,bias=False)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
        self.c1_2_2 = nn.Conv2d(self.nv,self.nv,5,1,2,bias=False,groups=self.nv) # w:[nv,1,5,5] input:[b,nv,h,w] output:[b,nv,h,w]
        self.c2 = nn.Conv2d(self.nv,self.nv*self.vl,5,2,2,bias=False,groups=self.nv) # w:[nvvl,1,5,5] input:[b,nv,h,w] output:[b,nvvl,h,w]

        self.get_weight()
    
    def get_weight(self):
        out = F.conv_transpose2d(self.eye,self.c2.weight,stride=self.c2.stride,padding=0,groups=self.c2.groups)
        out = F.pad(out,(2,2,2,2))
        out1 = F.conv_transpose2d(out,self.c1_1.weight,stride=self.c1_1.stride,padding=0,groups=self.c1_1.groups)
        out2 = F.conv_transpose2d(out,self.c1_2_2.weight,stride=self.c1_2_2.stride,padding=2,groups=self.c1_2_2.groups)
        out2 = F.conv_transpose2d(out2,self.c1_2_1.weight,stride=self.c1_2_1.stride,padding=0,groups=self.c1_2_1.groups)
        self.weight = out1 + out2

        return self.weight
    
class Bipolar_ks3(Gonlin):
    def __init__(self,arg):
        super().__init__(arg)
        print('Bipolar:',self.chl,self.nv)
        self.c1_1 = nn.Conv2d(self.chl,self.nv,3,2,2,bias=False)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
        self.c1_2_1 = nn.Conv2d(self.chl,self.nv,3,2,2,bias=False)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
        self.c1_2_2 = nn.Conv2d(self.nv,self.nv,3,1,2,bias=False,groups=self.nv) # w:[nv,1,5,5] input:[b,nv,h,w] output:[b,nv,h,w]
        self.c2 = nn.Conv2d(self.nv,self.nv*self.vl,5,2,2,bias=False,groups=self.nv) # w:[nvvl,1,5,5] input:[b,nv,h,w] output:[b,nvvl,h,w]

        self.get_weight()
    
    def get_weight(self):
        out = F.conv_transpose2d(self.eye,self.c2.weight,stride=self.c2.stride,padding=0,groups=self.c2.groups)
        out = F.pad(out,(1,1,1,1))
        out1 = F.conv_transpose2d(out,self.c1_1.weight,stride=self.c1_1.stride,padding=0,groups=self.c1_1.groups)
        out2 = F.conv_transpose2d(out,self.c1_2_2.weight,stride=self.c1_2_2.stride,padding=1,groups=self.c1_2_2.groups)
        out2 = F.conv_transpose2d(out2,self.c1_2_1.weight,stride=self.c1_2_1.stride,padding=0,groups=self.c1_2_1.groups)
        self.weight = out1 + out2

        return self.weight
    
# class Bipolar_1x1(Gonlin):
#     def __init__(self,arg):
#         super().__init__(arg)
#         self.c0_1 = nn.Conv2d(self.chl,self.nv,1,1,0,bias=False)
#         self.c0_2 = nn.Conv2d(self.chl,self.nv,1,1,0,bias=False)
#         self.c1_1 = nn.Conv2d(self.nv,self.nv,5,2,2,bias=False,groups=self.nv)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
#         self.c1_2_1 = nn.Conv2d(self.nv,self.nv,5,2,2,bias=False,groups=self.nv) # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
#         self.c1_2_2 = nn.Conv2d(self.nv,self.nv,5,1,2,bias=False,groups=self.nv) # w:[nv,1,5,5] input:[b,nv,h,w] output:[b,nv,h,w]
#         self.c2 = nn.Conv2d(self.nv,self.nv*self.vl,5,2,2,bias=False,groups=self.nv) # w:[nvvl,1,5,5] input:[b,nv,h,w] output:[b,nvvl,h,w]

#         self.get_weight()
    
#     def get_weight(self):
#         out = F.conv_transpose2d(self.eye,self.c2.weight,stride=self.c2.stride,padding=0,groups=self.c2.groups)
#         out = F.pad(out,(2,2,2,2))
#         out1 = F.conv_transpose2d(out,self.c1_1.weight,stride=self.c1_1.stride,padding=0,groups=self.c1_1.groups)
#         out1 = F.conv_transpose2d(out1,self.c0_1.weight,stride=self.c0_1.stride,padding=0,groups=self.c0_1.groups)
#         out2 = F.conv_transpose2d(out,self.c1_2_2.weight,stride=self.c1_2_2.stride,padding=2,groups=self.c1_2_2.groups)
#         out2 = F.conv_transpose2d(out2,self.c1_2_1.weight,stride=self.c1_2_1.stride,padding=0,groups=self.c1_2_1.groups)
#         out2 = F.conv_transpose2d(out2,self.c0_2.weight,stride=self.c0_2.stride,padding=0,groups=self.c0_2.groups)
#         self.weight = out1 + out2

#         return self.weight
    
class Bipolar_1x1(Gonlin):
    def __init__(self,arg):
        super().__init__(arg)
        self.c0_1 = nn.Conv2d(self.chl,self.nv,1,1,0,bias=False)
        self.c0_2 = nn.Conv2d(self.chl,self.nv,1,1,0,bias=False)
        self.c1_1 = nn.Conv2d(self.nv,self.nv,5,2,2,bias=False,groups=self.nv)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
        self.c1_2_1 = nn.Conv2d(self.nv,self.nv,5,2,2,bias=False,groups=self.nv)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
        self.c1_2_2 = nn.Conv2d(self.nv,self.nv,5,1,2,bias=False,groups=self.nv) # w:[nv,1,5,5] input:[b,nv,h,w] output:[b,nv,h,w]
        self.c2 = nn.Conv2d(self.nv,self.nv*self.vl,5,2,2,bias=False,groups=self.nv) # w:[nvvl,1,5,5] input:[b,nv,h,w] output:[b,nvvl,h,w]

        self.get_weight()
    
    def get_weight(self):
        out = F.conv_transpose2d(self.eye,self.c2.weight,stride=self.c2.stride,padding=0,groups=self.c2.groups)
        out = F.pad(out,(2,2,2,2))
        out1 = F.conv_transpose2d(out,self.c1_1.weight,stride=self.c1_1.stride,padding=0,groups=self.c1_1.groups)
        out1 = F.conv_transpose2d(out1,self.c0_1.weight,stride=self.c0_1.stride,padding=0,groups=self.c0_1.groups)
        out2 = F.conv_transpose2d(out,self.c1_2_2.weight,stride=self.c1_2_2.stride,padding=2,groups=self.c1_2_2.groups)
        out2 = F.conv_transpose2d(out2,self.c1_2_1.weight,stride=self.c1_2_1.stride,padding=0,groups=self.c1_2_1.groups)
        out2 = F.conv_transpose2d(out2,self.c0_2.weight,stride=self.c0_2.stride,padding=0,groups=self.c0_2.groups)
        self.weight = out1 + out2

        return self.weight
    
class Vanilla(Gonlin):
    def __init__(self,arg):
        super().__init__(arg)
        self.c1 = nn.Conv2d(self.chl,self.nv*self.vl,arg.lgn_kernel_size,arg.lgn_stride,arg.lgn_padding,bias=False)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]

        self.get_weight()
    
    def get_weight(self):
        out = F.conv_transpose2d(self.eye,self.c1.weight,stride=self.c1.stride,padding=0,groups=self.c1.groups)
        self.weight = out

        return self.weight
    
class Vanilla_2layer(Gonlin):
    def __init__(self,arg):
        super().__init__(arg)
        self.c1 = nn.Conv2d(self.chl,self.nv,5,2,2,bias=False)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
        self.c2 = nn.Conv2d(self.nv,self.nv*self.vl,5,2,2,bias=False,groups=self.nv)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]

        self.get_weight()
    
    def get_weight(self):
        out = F.conv_transpose2d(self.eye,self.c2.weight,stride=self.c2.stride,padding=0,groups=self.c2.groups)
        out = F.conv_transpose2d(out,self.c1.weight,stride=self.c1.stride,padding=0,groups=self.c1.groups)
        self.weight = out

        return self.weight
    

class Vanilla_2layer_nogroup(Gonlin):
    def __init__(self,arg):
        super().__init__(arg)
        self.c1 = nn.Conv2d(self.chl,self.nv,5,2,2,bias=False)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
        self.c2 = nn.Conv2d(self.nv,self.nv*self.vl,5,2,2,bias=False,groups=1)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]

        self.get_weight()
    
    def get_weight(self):
        out = F.conv_transpose2d(self.eye,self.c2.weight,stride=self.c2.stride,padding=0,groups=self.c2.groups)
        out = F.conv_transpose2d(out,self.c1.weight,stride=self.c1.stride,padding=0,groups=self.c1.groups)
        self.weight = out

        return self.weight

class Lgn_ende_multi(nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.arg = arg
        self.channel = arg.channel
        self.n_vector = arg.n_vector
        self.vector_length = arg.vector_length
        self.lgn_kernel_size = arg.lgn_kernel_size
        self.lgn_stride = arg.lgn_stride
        self.activate_fn = arg.activate_fn

        self.conv = eval(arg.gonlin)(arg)
        print('weight_size',self.conv.weight.size())
        arg.lgn_padding = int((self.conv.weight.size(2)-1)/2)
        self.lgn_padding = arg.lgn_padding
        print('gangling cell padding:',self.lgn_padding)

        if self.activate_fn is None:
            self.nonlinear = nn.Identity()
        elif self.activate_fn == 'gelu':
            self.nonlinear = nn.GELU()
        elif self.activate_fn == 'relu':
            self.nonlinear = nn.ReLU()
        

    def forward(self,x):
        w = self.conv.get_weight()
        return self.nonlinear(F.conv2d(x,w,stride=self.lgn_stride,padding=self.lgn_padding))
    
    def deconv(self,f):
        w = self.conv.get_weight()
        return F.conv_transpose2d(f,w,stride=self.lgn_stride,padding=self.lgn_padding)

    def deconv_group(self,f):
        w = self.conv.get_weight()
        img_group = F.conv_transpose2d(f,w,stride=self.lgn_stride,padding=self.lgn_padding,groups=self.n_vector)
        img = reduce(img_group,'b (nv c) h w -> b c h w','sum',nv=self.n_vector)
        return img_group,img
    
    def fullconv(self,x):
        w = self.conv.get_weight()
        return self.nonlinear(F.conv2d(x,w,padding=self.lgn_padding))
    
    def fulldeconv(self,f):
        w = self.conv.get_weight()
        return F.conv_transpose2d(f,w,padding=self.lgn_padding)/self.lgn_stride**2
    
    def fulldeconv_group(self,f):
        w = self.conv.get_weight()
        return F.conv_transpose2d(f,w,padding=self.lgn_padding,groups=self.n_vector)/self.lgn_stride**2
    

    
