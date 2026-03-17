import numpy as np

# appending to an array
# base array
vector = np.array([1, 2, 3, 4, 5, 6])
print(vector)

# the thing here is that size is fixed, so this creates a copy..

vector_2 = np.append(vector, [10, 20])
print(vector_2)

# and yeh when slicing you are returned a view, hence when modified you will then also modify the same data
# this example showcases this, copy on python lists

vector_3 = np.array([2, 4, 6, 8])
vector_4 = vector_3[::2]
print(vector_4)
vector_4[0] = 1
print(vector_3)

# meh
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix[0, 2])
print()
print(matrix.ndim) # 2
print(matrix.itemsize) # 8, aways bytes buddy, not bits
print(matrix.shape) # (3,3)

# nice to do
npzeros = np.zeros(4).astype(np.int16)
npfulls = np.full(3,3)
npones = np.ones(2)
nprangee = np.arange(0, 10, 3) # not inclusive ofc this would print 0, 3, 6, 9
nplin = np.linspace(0, 50, 11) # nice to do odd number here, n + 1 often
npemtpy = np.empty(4)
print(npzeros)
print(npfulls)
print(npones)
print(nprangee)
print(nplin)
print(npemtpy)

# add, remove, sort

vector_5 = np.array([4, 8, 2, 4, 6, 1, 10, 3, 45])
vector_5.sort()
print(vector_5)

# concatenate
# always been an issue with understanding
# then look at hstack, vstack, stack, columnstack

a = np.array([[1, 2, 3], [1, 2, 3]])
b = np.array([[4, 5, 6], [4, 5, 6]])
ab = np.concatenate((a, b), axis=1) # to do se mer på det her
print(ab)


# reshape 
print("Reshape")
np1 = np.arange(0, 11, 3)
respahed = np1.reshape((2,2))
print(np1)
print(np1.shape)
print(respahed)
print(respahed.shape)

print("reshape inferred")
np1 = np.linspace(0, 10, 30)
respahed = np1.reshape((2,-1))
print(respahed)



# newaxis

print("newaxis")
np2 = np.arange(0, 10, 2)
print(np2)
print(np2[:, np.newaxis])
print(np2[np.newaxis, :])
print(np2[None, np.newaxis])
print(np2[np.newaxis, None])
# print(np2[:, 1])

print("along an axis")

x = np.array([
    [[1, 2, 3],
    [50, 60, 70],
    [0, 1, 0]], 
     [[1, 2, 3],
    [50, 60, 70],
    [0, 1, 0]], 
])
print(x.shape)
result = np.sum(x, axis=0)
print(result)


# expandimensions TO:DO

print("conditionals")

z = np.arange(0, 20, 2)
mask = z < 10
print(z[mask])


# nonzero
print("nonzero")
np3 = np.array([0, 5, 3, 2, 10, 3, 213, 0, 456, 786, 342, 0])
print(np3)

print(np.nonzero(np3))

# np.reshape(-1), flatten, ravel
# the diff between them

print("reshape, flatten, ravel")

np4 = np.linspace(0, 100, 50).reshape(5, 2, -1)
print(np4.shape)
print(np4.reshape(-1)) # reshaped back
print(np4.flatten()) # copy
print(np4.ravel()) # view



print("insert")

np5 = np.array([1, 2, 3, 4, 5, 6])
np.insert(np5, 3, 3)



print("roll")

np5 = np.array([1, 2, 3, 4, 5, 6])
np.roll(np5, 2)
print(np5)

arr = np.arange(3)
print(arr.shape)


print("- - - - -")
# Broadcasting is how NumPy makes different shapes compatible, often by stretching dimensions of size 1.
# Element-wise means operations happen on corresponding elements.
# Ufuncs are NumPy functions like sqrt, add, and sin that usually work element-wise.
# Reductions like sum and mean combine many values into one result. 


print("- - - - -") # random
rng = np.random.default_rng(seed=122)
print(rng.integers(10,22))


vector_6 = np.arange(0, 10, 2)
print(vector_6[vector_6 <= 2])

