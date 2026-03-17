# constructig tensors
import torch

print(torch.tensor([1, 2, 3]))
print(torch.tensor([1, 2, 3]).dtype)
print(torch.tensor([1, 2, 3], dtype=torch.float32).dtype)
print(torch.tensor([1, 2, 3]).to(torch.float16).dtype)

trch1 = torch.ones((2, 4))
trch2 = torch.full((3, 3), 3)
trch3 = torch.zeros((3, 3))
trch4 = torch.rand((3, 3)).to(torch.float32)


print(trch1)
print(trch2)
print(trch3)
print(trch4)
print(trch4.dtype)

# operations on tensors

trcha = torch.tensor([3, 6, 9])
trchb = torch.tensor([4, 8, 12])

print(trchb + trcha)
print(trchb - trcha)
print(trchb * trcha)

print(torch.add(trchb, trcha))
print(torch.sub(trchb, trcha))
print(torch.multiply(trchb, trcha))
print(torch.sum(trcha))

trchoutput = torch.tensor([]) # hm. hmm

print(torch.add(trchb, trcha, alpha=5, out=trchoutput))
print(torch.sub(trchb, trcha, alpha=5,  out=trchoutput))
print(torch.multiply(trchb, trcha))
print("-")
print(trchoutput)