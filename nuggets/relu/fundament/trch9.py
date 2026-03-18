from torch.utils.data import Dataset, DataLoader
import torch


cuda = torch.cuda.is_available()
device = 'cuda' if cuda else "cpu"

template = torch.randn(2, 3)
new = torch.randn_like(template)

print(template)
print(template.to(device))

print(new)
print(new.dtype)
print(new.ndim)
print(new.device)

#miscy

# dim = 0, dim = 1 , axis = 1, on reduction functions

#. torch gather, is like mask?, or where with mask?