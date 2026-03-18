# dot product
# np.T transposing
# np.where
import numpy as np

np1 = np.array([1, 2, 3, 4, 5, 6])
print(np.where(np1 > 3, np1, 0))
# [0, 0, 0, 4, 5, 6]

# dot prodcut

arrayen = np.array([2, 4, 7, 10, 12])
arrayen2 = np.array([2, 4, 7, 10, 12])

print(np.dot(arrayen, arrayen2))
print(arrayen @ arrayen2)
