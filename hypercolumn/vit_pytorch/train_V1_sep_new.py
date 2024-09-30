import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
# from vit_pytorch.V1 import *
from hypercolumn.vit_pytorch.V1_sep import Lgn_ende_multi as Lgn_ende
import copy
    

        
class Column_trans_rot_lgn(nn.Module):
    def __init__(
        self,
        arg,
        masking_ratio = 0.75,
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        self.arg = arg
        arg_trans = self.get_trans_arg()
        arg_rot = self.get_rot_arg()

        self.lgn_ende = nn.ModuleList([Lgn_ende(arg) for arg in arg_rot])

        print('len(self.lgn_ende):',len(self.lgn_ende))

    def forward(self,img):
        if 'mnist' or 'cifar' in self.arg.dataset:
            if self.arg.ensemble == 'vanilla' or self.arg.ensemble == 'single' or self.arg.ensemble == 'double':
                lgn = [img]
            elif self.arg.ensemble == 'moe':
                lgn = [lgn_ende.fullconv(img) for lgn_ende in self.lgn_ende]
            elif self.arg.ensemble == 'mix':
                lgn = [lgn_ende.fullconv(img) for lgn_ende in self.lgn_ende]
            elif self.arg.ensemble == 'ensemble':
                lgn = [lgn_ende.fullconv(img) for lgn_ende in self.lgn_ende]
                lgn = [img] + lgn + lgn

        else:
            if self.arg.ensemble == 'vanilla' or self.arg.ensemble == 'single' or self.arg.ensemble == 'double':
                lgn = [img]
            elif self.arg.ensemble == 'moe':
                lgn = [lgn_ende(img) for lgn_ende in self.lgn_ende]
            elif self.arg.ensemble == 'mix':
                lgn = [lgn_ende(img) for lgn_ende in self.lgn_ende]
            elif self.arg.ensemble == 'ensemble':
                lgn = [lgn_ende(img) for lgn_ende in self.lgn_ende]
                lgn = [img] + lgn + lgn

        return lgn

    def get_rot_arg(self):
        arg = []
        # for nv,vl,rn in self.arg.n_vector,self.arg.vector_length,self.arg.rot_num:
        for i,nv in enumerate(self.arg.n_vector):
            nv,vl,rn = self.arg.n_vector[i],self.arg.vector_length[i],self.arg.rot_num[i]
            #16 2 4; 64 1 8
            arg_rot = copy.deepcopy(self.arg)
            arg_rot.n_vector = int(nv/rn)    #4
            arg_rot.vector_length = int(vl*rn)     #8
            arg.append(arg_rot)

        return arg

    def get_trans_arg(self):
        arg = []
        # for nv,vl,rn in self.arg.n_vector,self.arg.vector_length,self.arg.rot_num:
        for i,nv in enumerate(self.arg.n_vector):
            nv,vl,rn = self.arg.n_vector[i],self.arg.vector_length[i],self.arg.rot_num[i]
            arg_trans = copy.deepcopy(self.arg)
            arg_trans.n_vector = nv
            arg_trans.vector_length = vl
            arg.append(arg_trans)
        
        return arg