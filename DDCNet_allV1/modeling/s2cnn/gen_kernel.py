# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 21:36:36 2019

@author: Administrator
"""

import numpy as np
import math
from math import pi
from numpy import cos,sin
from PIL import Image
import torch.nn as nn
import torch
#u1_min=int(3/2) 
#u1_max=int(17/9)     
#u1_gap=90
#x1_func='(0+6*cos(u))*cos(7*pi/4)'
#y1_func='(0+6*cos(u))*sin(7*pi/4)'
#z1_func='6*sin(u)+6'
#u2_min=int(10/9+90)  
#u2_max=int(3/2)         
#u2_gap=90
#x2_func='(8*2^0.5+6*cos(u))*cos(7*pi/4)'
#y2_func='(8*2^0.5+6*cos(u))*sin(7*pi/4)'
#z2_func='6*sin(u)+6'
#u1_sample=range(u1_max)[u1_min::u1_gap]
#u2_sample=range(u2_max)[u2_min::u2_gap]
#u_sample=[u1_sample, u2_sample]
#[u_row,u_column]=np.array(u_sample).shape
##print(u_row,u_column)
#u_column=u_row

#for n in range(1,u_column):
#    u=u_sample[n]
#    if n<=np.ceil(u_column/2):
#        x_sample_orig[n]=eval(x1_func)
#        print()
#        y_sample_orig[n]=eval(y1_func)
#        z_sample_orig[n]=eval(z1_func)
#    else:
#        x_sample_orig[n]=eval(x2_func)
#        y_sample_orig[n]=eval(y2_func)
#        z_sample_orig[n]=eval(z2_func)
#
#
CCD_length = 8.8
CCD_width = 6.6
ox = CCD_width/2
oy = CCD_length/2
b = 77.27
a = 6.78 
k = 5
def generator1(data,f):
    
    #print(f)
    kernel = torch.FloatTensor(data)
    weight1 = kernel#nn.Parameter(data=kernel, requires_grad=False)
    shape1 = kernel.shape
    kernel2 = np.zeros(shape=(3,3,7,9))
    kernel2=torch.FloatTensor(kernel2)
    for a in range(shape1[0]):
        for i in range(shape1[1]):
            for j in range(shape1[2]):
                for k in range(shape1[2]):
                    #if k != int(np.sqrt((7*j-j*j)/3)):
                        kernel2[a,i,j,k+int(j*(7-j)*(1/6))] = weight1[a,i,j,k]
                    
    #print(kernel2)
    kernel2 = nn.Parameter(data=kernel2, requires_grad=True)
    
    return kernel2

def generator2(data,f):
    
    #print(f)
    kernel = torch.FloatTensor(data)
    weight1 = kernel#nn.Parameter(data=kernel, requires_grad=False)
    shape1 = kernel.shape
    kernel2 = np.zeros(shape=(3,3,7,9))
    kernel2=torch.FloatTensor(kernel2)
    for a in range(shape1[0]):
        for i in range(shape1[1]):
            for j in range(shape1[2]):
                for k in range(shape1[2]):
                    #if k != int(np.sqrt((7*j-j*j)/3)):
                        kernel2[a,i,j,k+int((j-7)*j*(1/6))] = weight1[a,i,j,k]
                    
    #print(kernel2)
    kernel2 = nn.Parameter(data=kernel2, requires_grad=True)
    
    return kernel2
    
def gen_kernel(weight1):
    weight3 = torch.randn((64,3,3,3)).cuda()
    the_m = np.random.rand(3,3)
    for m in range(3):
          for n in range(3):
            the_m[m,n] =  ((2*m-oy)*cos(abs(m-k/2))+oy-b)/a-(2*m)
            #print(the_m)
            the_m[the_m>2] = 2
            the_m[the_m<-2] = -2
            if m+int(the_m[m,n]) <= 3:
              weight3[:,:,m+int(the_m[m,n]),n] = weight1[:,:,m,n]
    weight =  nn.Parameter(data=weight3, requires_grad=True)
    return weight