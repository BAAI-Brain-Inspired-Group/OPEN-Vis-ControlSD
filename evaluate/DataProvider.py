from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import glob
import cv2
from typing import Union
from einops import rearrange


import clip

# class LoadFolder(Dataset):
#     def __init__(self,dir:str='',size:Union[tuple,list,callable]=(299,299)):

#         # scan files in dir with .png
#         self.data = glob.glob(dir+'/*.png')
#         print('LoadFloder:',len(self.data),self.data)
#         self.size = size
        
#         # normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
#         # self.trans = transforms.Compose([
#         #     transforms.ToTensor(),
#         #     normalize,
#         # ])

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx:int) -> torch.Tensor:
#         # load data
#         # print('LoadFolder __getitem start')
#         item = self.data[idx]
#         img_bgr = cv2.imread(item)
        
#         # print('img_bgr:',img_bgr.shape)

#         # don't forget images in cv2 is bgr style.
#         img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
#         # print('img_rgb:',img_rgb.shape)

#         img = cv2.resize(img_rgb,self.size)
        
#         img = (img.astype(np.float32) / 127.5)-1.
#         img = rearrange(img,'h w c -> c h w')

#         return img

class LoadFolder(Dataset):
    def __init__(self,dir:str='',size:Union[tuple,list,callable]=(299,299),color="RGB"):

        # scan files in dir with .png
        self.data = glob.glob(dir+'/*.png')
        # print('LoadFloder:',len(self.data),self.data)

        self.size = size
        self.color = color
        if color=='L':
            normalize = transforms.Normalize(mean=[0.5],std=[0.5])
        else:
            normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        self.trans = transforms.Compose([
            # transforms.Resize(size),
            transforms.ToTensor(),
            normalize,
        ])
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx:int) -> torch.Tensor:
        # load data
        # print('LoadFolder __getitem start')
        item = self.data[idx]
        # print('item:',item)
        img = self.trans(Image.open(item).convert(self.color))
        # print('img:',img.size())
        # print('img:',img.size())
        

        return img

class LoadFolderCLIP(Dataset):
    def __init__(self,dir:str='',size:Union[tuple,list,callable]=(299,299)):

        # scan files in dir with .png
        self.data = glob.glob(dir+'/*.png')
        # print('LoadFloder:',len(self.data),self.data)

        
        _, self.preprocess = clip.load("ViT-B/32", device='cpu')
        self.size = size
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx:int) -> torch.Tensor:
        # load data
        # print('LoadFolder __getitem start')
        item = self.data[idx]
        # print('item:',item)
        img = self.preprocess(Image.open(item))
        # print('img:',img.size())
        

        return img
    
def data_provider(root:str, batch_size:int=128, n_threads:int=4,dstype='CLIP',color='RGB',**args):

    if dstype is 'CLIP':
        dataset = LoadFolderCLIP(dir=root)
    elif dstype is 'SSIM':
        dataset = LoadFolder(dir=root,color=color)

    # set shuffle False
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=n_threads)

    return dataloader