
import torch
from torch.nn import functional as f
import numpy as np 
x = torch.arange(0, 10 * 3 * 128 * 256).float()

x = x.view(10, 3, 128, 256)
 
x1 = f.unfold(x, kernel_size=3, stride=1,padding=1)
print(x1.shape)
x=torch.arange(0, 1 * 3*3*3*4).float()
x=x.view(1,3*3*3,4)
# print(x)
# x2 = f.fold(x,(3,6),kernel_size=3,stride=1)
# print(x2.shape)
