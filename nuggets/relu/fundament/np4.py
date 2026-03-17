# dot product
# np.T transposing
# np.where
import numpy as np

np1 = np.array([1, 2, 3, 4, 5, 6])
print(np.where(np1 > 3, np1, 0))
# [0, 0, 0, 4, 5, 6]