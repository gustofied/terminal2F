# [:, None] -> unsqueze(1)
# [None, :] -> unsqueze(0)

import torch
from torch.utils.data import Dataset, DataLoader

class TinyDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        self.y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

    def __len__(self):
        return len(self.x)



dataset = TinyDataset()
print(dataset[0])   # one sample

loader = DataLoader(dataset, batch_size=2, shuffle=True)

for xb, yb in loader:
    print("xb:", xb)
    print("yb:", yb)


from torchvision import datasets
from torch.utils.data import DataLoader

train_data = datasets.MNIST(root="data", train=True, download=True)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


# Dataset   -> sample-level access
# DataLoader -> batch-level iteration