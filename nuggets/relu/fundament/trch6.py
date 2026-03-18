import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Go through 1

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x**2
y.sum().backward()
print("x.grad:", x.grad)

xs = torch.linspace(-4, 4, 200, requires_grad=True)
ys = xs**2
ys.sum().backward()

plt.figure(figsize=(6,4))
plt.title("Function and gradient")
plt.show()
plt.plot(xs.detach().numpy(), ys.detach().numpy(), label="y = x^2")
plt.plot(xs.detach().numpy(), xs.grad.detach().numpy(), label="dy/dx")  # ty:ignore[unresolved-attribute]
plt.legend()
plt.title("heeyee")
plt.xlabel("x")
plt.ylabel("value")
plt.show()

# simple matploty

x = [0, 1, 2, 3, 4]
y = [0, 2, 3, 9, 16]
z = [10, 2, 3, 9, 16]

plt.figure(figsize=(8, 4))
plt.xlim(0, 10)   # x-axis from 0 to 10
plt.ylim(0, 20)   # y-axis from 0 to 20
plt.plot(x, y)
plt.plot(x, z)
plt.title("y = x^2")
plt.xlabel("x values")
plt.ylabel("y values")
plt.show()


# accumulate

x = torch.tensor(3.0, requires_grad=True)

y = x**2
y.backward()
print("after first backward:", x.grad)   # 6

y = x**2
y.backward()
print("after second backward:", x.grad)  # 12, accumulated

# you can do grad_fn to look at the gradient

# .

# torch.Relu()