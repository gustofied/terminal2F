import torch
import torch.nn.functional as F

x = torch.tensor(2.0)
target = torch.tensor(1.0)

w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

z = w * x + b
pred = F.relu(z)
loss = (pred - target) ** 2

print("pred:", pred.item())
print("loss:", loss.item())

loss.backward()

print("w.grad:", w.grad)
print("b.grad:", b.grad)