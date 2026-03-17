import torch
import numpy as np

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
device_count = torch.cuda.device_count()
# device_name = torch.cuda.get_device_name(0)

print(cuda)
print(device)
print(device_count)
# print(device_name)

# Tensors

tensor = torch.Tensor([[3, 2 ,1], [3, 2, 2]])
print(tensor)

print(tensor * 5) # elementwise, utiilsing broadcasting
print(tensor.sum()) 

np1 = np.arange(1, 12, 2)
print(np1)

torched = torch.from_numpy(np1)
print(torched)

reshaped = torched.reshape((2,3))
print(reshaped)
print(reshaped.ndim)
print(reshaped.shape)
print(reshaped.itemsize)
print(reshaped.device)

# torch random

random_tensor = torch.rand(3, 2)
print(random_tensor)

print("- - - - - - - - - -")
random_tensor2 = torch.randn(3, 3, 3)
print(random_tensor2)


# let's create two random tensors

random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)

torch.manual_seed(222)
random_tensor_A = torch.rand(3, 4)
torch.manual_seed(222)
random_tensor_B = torch.rand(3, 4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)

# better do make a generator

g1 = torch.Generator().manual_seed(22)
g2 = torch.Generator().manual_seed(22)

print(torch.rand(3, 4, generator=g1))
print(torch.rand(3, 4, generator=g2))

print("- - - - - -")

# autograd

# In this toy example, a and b are just variables we want gradients for. f is the objective function. 
# Autograd computes how changing a or b would change f. In a real neural network, the same idea is used with the loss and the model parameters, 
# and the optimizer uses those gradients to update the parameters to reduce the loss.

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

f = 3 * a**3 - b**2
print(f)
f.backward(gradient=torch.tensor([1., 1.]))
print(a.grad)
print(b.grad)


