# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:59:18 2020

@author: ssk
"""
import torch as t
import numpy as np
import os 
import torchvision
import torch.nn as nn
from collections import OrderedDict

#resnet模块
class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels=256,alpha=0.1):
        super(ResBlock, self).__init__()
        m = OrderedDict()
        m['conv1']=nn.Conv2d(in_channels, out_channels,kernel_size=3,bias=False,stride=1,padding=1)
        m['relu1']=nn.ReLU(True)
        m['conv2']=nn.Conv2d(out_channels,out_channels, kernel_size=3,bias=False,stride=1,padding=1)
        self.net = nn.Sequential(m)
        self.relu= nn.Sequential(nn.ReLU(True))
        self.alpha = alpha
    #out = 256+2*0-1*(3-1)-1 +1 = 256
    def forward(self,x):
        out =self.net(x)
        out = self.alpha*out + x
        return out
    


class DSen2_CR(nn.Module):
    def __init__(self,in_channels,out_channels,alpha=0.1,num_layers = 16 , feature_sizes = 256):
        super(DSen2_CR,self).__init__()
        m= []
        m.append(nn.Conv2d(in_channels,out_channels=feature_sizes,kernel_size=3,bias=True,stride = 1 ,padding=1))
        m.append(nn.ReLU(True))
        for i in range(num_layers):
            m.append(ResBlock(feature_sizes,feature_sizes,alpha))
        m.append(nn.Conv2d(feature_sizes,out_channels,kernel_size=3, bias=True,stride=1,padding=1))
        self.net = nn.Sequential(*m)
    
    def forward(self, x):
        #print(x.shape)
        return x[:,2:,:,:]+self.net(x)
    