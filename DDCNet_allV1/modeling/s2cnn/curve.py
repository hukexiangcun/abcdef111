import numpy as np
import math
from math import pi
from numpy import cos,sin
from PIL import Image
from ..distortion.Camera_Model import Camera
def up_down():
    up,down = Camera()
    up_c = up#curve(up)
    down_c = down #curve(down)
    return up_c,down_c

def curve(img):
    #img = np.array(Image.open('3.jpg').convert('L'))
    #np.savetxt('1.txt',img)
    #img1 = img[img != 255]
    img = np.array(img)
    shape1 = img.shape
    img = img.reshape(shape1[1],shape1[0])
    #print(shape1)
    x =img[0]
    y=img[1]
#    for i in range(int(shape1[0]/10)):
#        for j in range(int(shape1[1]/10)):
#            if img[i*10][j*10] != 255 and img[i*10][j*10] != 0:
#                x.append(i*10)
#                y.append(j*10)
    s = len(x)
    x_sample_orig = x + (np.random.random_integers(5, size=(s)) - 1) / 4
    y_sample_orig = y + (np.random.random_integers(5, size=(s)) - 1) / 4
    #np.savetxt('1.txt',[x,y])
    #print(x_sample_orig)
    #print(y_sample_orig)
    u_number = len(x)
    print('u_number is:', u_number)
    u_column = u_number


    x_noise=np.zeros(shape=u_column)
    y_noise=np.zeros(shape=u_column)
    dx_discrete=np.zeros(shape=u_column)
    dy_discrete=np.zeros(shape=u_column)
    dax_discrete=np.zeros(shape=u_column)
    day_discrete=np.zeros(shape=u_column)
    d_sum=0
    for n in range(1,(u_column-1)):
        distance=((x_sample_orig[n+1]-x_sample_orig[n])**2+(y_sample_orig[n+1]-y_sample_orig[n])**2)**0.5
        d_sum=d_sum+distance



    E1=2
    E2=8
    #u_number=u_column
    d_ave=d_sum/(u_column-1)
    for nd in range(1):
        if nd ==0:
            sd=0*d_ave
            window=2
            window0=1
            window1=1
        if nd ==1:
            sd=0.3*d_ave
            window=6
            window0=math.ceil(window/2)
            window1=window-window0
       
        x_sample=x_sample_orig
        y_sample=y_sample_orig
        
        #z_sample=z_sample_orig+z_noise*sd

        x_st=np.zeros(shape=u_column)
        y_st=np.zeros(shape=u_column)
        dr_st=np.zeros(shape=u_column)
        t_st=np.zeros(shape=u_column)
        sum_st=0
        for n in range(2,u_number):
            x_st[n]=x_sample[n]-x_sample[n-1]
            y_st[n]=y_sample[n]-y_sample[n-1]
            #z_st[n]=z_sample[n]-z_sample[n-1]
            dr_st[n]=(x_st[n]**2+y_st[n]**2)**0.5
            sum_st=sum_st+dr_st[n]
            t_st[n]=sum_st
        t_sample=t_st/sum_st
        t_discrete=t_st
        t_change=dr_st
        
        for n in range(1,u_number):
            n_start=n-window0
            n_end=n+window0
            if n_start<1:
                if n_end>u_number:
                    n_start=1
                    n_end=u_number        
                else:
                    n_start=1
                    n_end=n+window0
                
            else:
                if n_end>u_number:
                    n_start=n-window0
                    n_end=u_number
                else:
                    n_start=n-window0
                    n_end=n+window0
                        
            sum_numerator=0
            sum_denominator=0
            for cal_n in range(n_start,n_end):
                sum_numerator=sum_numerator+(t_sample[cal_n]-t_sample[n])*(x_sample[cal_n]-x_sample[n])
                
                sum_denominator=sum_denominator+(t_sample[cal_n]-t_sample[n])**2
            
            dx_discrete[n]=sum_numerator/sum_denominator
       
        
        
        for n in range(1,u_number):
            n_start=n-window0
            n_end=n+window0
            if n_start<1:
                if n_end>u_number:
                    n_start=1
                    n_end=u_number        
                else:
                    n_start=1
                    n_end=n+window0
                
            else:
                if n_end>u_number:
                    n_start=n-window0
                    n_end=u_number
                else:
                    n_start=n-window0
                    n_end=n+window0
               
            sum_numerator=0
            sum_denominator=0
            for cal_n in range(n_start,n_end):
                sum_numerator=sum_numerator+(t_sample[cal_n]-t_sample[n])*(y_sample[cal_n]-y_sample[n])
                sum_denominator=sum_denominator+(t_sample[cal_n]-t_sample[n])**2
            
            dy_discrete[n]=sum_numerator/sum_denominator
        
    #    for n in range(1,u_number):
    #        n_start=n-window0
    #        n_end=n+window0
    #        if n_start<1:
    #            if n_end>u_number:
    #                n_start=1
    #                n_end=u_number        
    #            else:
    #                n_start=1
    #                n_end=n+window0
    #        else:
    #            if n_end>u_number:
    #                n_start=n-window0
    #                n_end=u_number
    #            else:
    #                n_start=n-window0
    #                n_end=n+window0
    #            
    #        sum_numerator=0
    #        sum_denominator=0
    #        for cal_n in range(n_start,n_end):
    #            sum_numerator=sum_numerator+(t_sample[cal_n]-t_sample[n])*(z_sample[cal_n]-z_sample[n])
    #            sum_denominator=sum_denominator+(t_sample[cal_n]-t_sample[n])^2
    #        dz_discrete[n]=sum_numerator/sum_denominator
        
        dr_discrete=(dx_discrete**2+dy_discrete**2)**0.5
        ax_discrete=dx_discrete/dr_discrete                               
        ay_discrete=dy_discrete/dr_discrete
        
        for n in range(1,u_number):
            n_start=n-window1
            n_end=n+window1
            if n_start<1:
                if n_end>u_number:
                    n_start=1
                    n_end=u_number     
                else:
                    n_start=1
                    n_end=n+window1
                
            else:
                if n_end>u_number:
                    n_start=n-window1
                    n_end=u_number
                else:
                    n_start=n-window1
                    n_end=n+window1
             
            sum_numerator=0
            sum_denominator=0
            for cal_n in range(n_start,n_end):
                sum_numerator=sum_numerator+(t_sample[cal_n]-t_sample[n])*(ax_discrete[cal_n]-ax_discrete[n])
                sum_denominator=sum_denominator+(t_sample[cal_n]-t_sample[n])**2
           
            dax_discrete[n]=sum_numerator/sum_denominator
        
        
        for n in range(1,u_number):
            n_start=n-window1
            n_end=n+window1
            if n_start<1:
                if n_end>u_number:
                    n_start=1
                    n_end=u_number       
                else:
                    n_start=1
                    n_end=n+window1
            
            else:
                if n_end>u_number:
                    n_start=n-window1
                    n_end=u_number
                else:
                    n_start=n-window1
                    n_end=n+window1
                            
            
            sum_numerator=0
            sum_denominator=0
            for cal_n in range(n_start,n_end):
                sum_numerator=sum_numerator+(t_sample[cal_n]-t_sample[n])*(ay_discrete[cal_n]-ay_discrete[n])
                sum_denominator=sum_denominator+(t_sample[cal_n]-t_sample[n])**2
          
            day_discrete[n]=sum_numerator/sum_denominator
        
    #    for n in range(1,u_number):
    #        n_start=n-window1
    #        n_end=n+window1
    #        if n_start<1:
    #            if n_end>u_number:
    #                n_start=1
    #                n_end=u_number        
    #            else:
    #                n_start=1
    #                n_end=n+window1
    #            
    #        else:
    #            if n_end>u_number:
    #                n_start=n-window1
    #                n_end=u_number
    #            else:
    #                n_start=n-window1
    #                n_end=n+window1
    #            
    #        sum_numerator=0
    #        sum_denominator=0
    #        for cal_n in range(n_start,n_end):
    #            sum_numerator=sum_numerator+(t_sample[cal_n]-t_sample[n])*(az_discrete[cal_n]-az_discrete[n])
    #            sum_denominator=sum_denominator+(t_sample[cal_n]-t_sample[n])**2
    #        daz_discrete[n]=sum_numerator/sum_denominator
        
        
        daxs_discrete=dax_discrete/dr_discrete
        #print(dr_discrete)
        
        days_discrete=day_discrete/dr_discrete
        #dazs_discrete=daz_discrete/dr_discrete
        drs_discrete=(daxs_discrete**2+days_discrete**2)**0.5
        k_discrete=drs_discrete 
        #print(k_discrete)
    np.savetxt('110.txt',k_discrete)
    k= k_discrete
    k = k[4:u_number]
    k_mean=np.mean(k)
    k_max=np.max(k)
    k1 = np.sum(k)
    k2 = np.abs(k-k1)
    k3 = np.mean(k2)
    p=8
    f = p*(k3)
    return f