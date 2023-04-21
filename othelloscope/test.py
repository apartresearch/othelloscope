import torch
from torch.nn import functional

x = torch.tensor([[2, 10], [2, 10], [2, 10]], dtype=torch.float32)
print(x.shape)
print(functional.normalize(x, dim=0))
