import torch
import numpy as np

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

print(device)

# normal tensor

tnsr1 = torch.tensor([1, 2, 3, 4 ,5])
print(tnsr1)

# from numpy

np1 = np.arange(1, 10, 1)
print(torch.from_numpy(np1))


# random tensor

g1 = torch.Generator().manual_seed(12)
print(torch.randn((3, 5), generator=g1))

# grab a whole row

rowsy = torch.linspace(1, 50, 20)
print(rowsy.shape)
rowsyd = rowsy.reshape(2,2,-1)
print(rowsyd)
print(rowsyd[1, 1, 1].item())
print(rowsyd[1, 1, :])


# resahpe vs copy

mock_tensor = torch.arange(0, 100, 1).reshape((5, 2 ,10))
print(mock_tensor.shape)
print(mock_tensor)
print(mock_tensor.view(20, -1))

# multiplication 

mocked1 = torch.tensor([[3, 2, 3], [3, 2, 0]], dtype=torch.int16)
mocked2 = torch.tensor([[1, 2, 3], [1, 1, 0]], dtype=torch.int16)

print(f"matrix 1 {mocked1.shape} and matrix 2. {mocked2.shape}")
print(mocked1 @ mocked2.T) # matrix multiplication
print(mocked1 * mocked2) # element wise
print(torch.mul(mocked1, mocked2)) # mul
print(mocked1.dtype)


# type changing

np2 = np.array([2, 4, 6, 8, 10, 12])
print(np2.dtype)
trch1 = torch.tensor([1, 2, 3])
print(trch1.dtype)

print(np2.astype(np.float32).dtype)
print(trch1.to(torch.float16).dtype)
print(trch1.float().dtype)


# print(trch1.to(device='cuda'))
