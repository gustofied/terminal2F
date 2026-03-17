# mnist
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train_data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
testing_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

data = list(train_loader)
print(len(data))
print(len(data[0][-1]))
print(data[1][0].shape) # understanding thi shape is important

print(data[0][0][0]) # why tripple 0 in here?

image = data[0][0][0]

label = data[0][1][0]

plt.imshow(image[0], cmap='gray')
plt.title(str(label))
plt.show()

