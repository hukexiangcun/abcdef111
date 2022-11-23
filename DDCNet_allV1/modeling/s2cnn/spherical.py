""" Spherical harmonics transforms/inverses and spherical convolution. """


import functools
import numpy as np
import torch
import torch.nn as nn
import itertools
from functools import reduce

import torch.nn.functional as F
import math
from math import cos,atan
from .dis_convolutional import dis_conv

CCD_length = 7.7
CCD_width = 5.5
ox = CCD_width/2
oy = CCD_length/2
a = 8.3
k = 4
dx = 0.00859375
dy = 0.00859375
f = 8
#a0 = 0.1
# cache outputs; 2050 > 32*64

class dis_convolution(nn.Module):
    def __init__(self,  batch_size):
        super(dis_convolution, self).__init__()
        
        # self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1,
                                # bias=False)
        
        # self.bn1 = nn.BatchNorm2d(3)
        # self.relu = nn.ReLU()
        
        # self._init_weight()
        
        w0=512
        h0=128
        self.bn = np.zeros((h0,w0))
        for i in range(h0):
            for j in range(w0):
                a0 = atan(dx*((j)-ox)/f)
                
                self.bn[i,j] = int(i-((cos(a0)-1)*((i)-oy)))
               
                
                if self.bn[i,j]<0:
                    self.bn[i,j] = 0
                if self.bn[i,j]>=h0:
                    self.bn[i,j] = h0-1
                    
        
    def forward(self, x):
        
        input = x
        
        input = input.cuda()
        shape1 = input.shape
        
        
        feature1 = torch.zeros((shape1[0],shape1[1],shape1[2],(shape1[3])))
        feature1 = input
        for j in range(shape1[3]):
            
            feature1[:,:,:,j] = input[:,:,self.bn[:,j],j]
                        
        out = feature1
        
        out = out.cuda()
        # out = self.conv1(out)
        # out = self.bn1(out)
        # out = self.relu(out)
        return out
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            

   