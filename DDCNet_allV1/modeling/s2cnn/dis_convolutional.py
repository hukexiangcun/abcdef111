import functools
import numpy as np
import torch
import torch.nn as nn
import itertools
from functools import reduce

import torch.nn.functional as F
import math
from math import cos,atan

#CCD相机参数
CCD_length = 7.7
CCD_width = 5.5
ox = CCD_width/2
oy = CCD_length/2
a = 8.3
k = 4
dx = 0.00859375
dy = 0.00859375
f = 8

class dis_conv(nn.Module):
    def __init__(self, w, h, batch_size):
        super(dis_conv,self).__init__()
        w0, h0 = w, h
        batch_size0 = batch_size
        self.bn = np.zeros((h0,w0))
        for i in range(h0):
            for j in range(w0):
                a0 = atan(dx*(j-ox)/f)
                self.bn[i,j] = math.floor((cos(a0)-1)*(i-oy))
        
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=0,
                                bias=False)
        
    def forward(self,input):
        input = input.cuda()
        shape1 = input.shape
        #print(shape1)
        
        feature1 = torch.zeros((shape1[0],shape1[1],shape1[2],(shape1[3])))
       
        
        matrix1 = torch.zeros((shape1[0],shape1[1],3,3)).cuda()
        
        tb = torch.zeros((9),dtype=torch.int)
        for i in range(shape1[2]):
            for j in range(shape1[3]):
                if i+2<shape1[2] and j+2 < (shape1[3]//2):
                    if i+2+self.bn[i+2,j+2] < shape1[2]:
                        for k in range(9):
                            if j >= 256:
                                j1 = j%256
                                tb[k] = int(i+int(k/3)+self.bn[int(i+k/3),int(j1+k%3)])
                            else:
                                tb[k] = int(i+int(k/3)+self.bn[int(i+k/3),int(j+k%3)])
                            
                            matrix1[:,:,int(k/3),int(k%3)] = input[:,:,tb[k],int(k%3)]
                            
                       
                        feature1[:,:,i,j] = self.conv1(matrix1).reshape(shape1[0],shape1[1])
                        
        out = feature1
        
        out = out.cuda()
        return out
        
        
                