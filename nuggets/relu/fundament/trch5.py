import torch

x = torch.tensor([3.0], dtype=torch.float32, requires_grad=True)

y = x**2 + 4 
y.backward()

print(x)
print(x.grad)

# Example 2

x = torch.tensor(3.0)
y = torch.tensor(7.0)

x.requires_grad_(True)
y.requires_grad_(True)

z = x**y

z.backward()
print(x.grad, y.grad)
