import math
import numpy as np
import torch
from matplotlib import pyplot as plt

x = torch.linspace(0, 5, 100, requires_grad=True)
y = (x**2).cos()
s = y.sum()

[dydx] = torch.autograd.grad(s, [x])

plt.plot(x.detach(), y.detach(), label='y')
plt.plot(x.detach(), dydx, label='dy/dx')
plt.legend()
plt.show()