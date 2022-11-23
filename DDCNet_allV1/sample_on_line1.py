import torch
import torch.nn.functional as F
inputs = torch.arange(1, 21).reshape(1, 2, 2, 5)
filters = torch.arange(1, 7).reshape(2, 1, 1, 3)
print(inputs)
print(filters)
res = F.conv2d(input=inputs.float(), weight=filters.float(), stride=(1, 1), groups=2)
print(res.shape)
