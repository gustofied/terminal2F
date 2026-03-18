import numpy as np

np1 = np.array([1, 2, 3, 4, 5, 6])
np2 = np.array([7, 8, 9, 10, 11, 12])

np12 = np.concatenate((np1, np2))
print(np12)

np11 = np1.reshape((3, 2))
np22 = np2.reshape((3, 2))
print(np11)
print(np22)

np12 = np.concatenate((np11, np22), axis=0)  # vstack
print("---")

np122 = np.concatenate((np11, np22), axis=1)  # hstack

print(np12)
print(np122)

print(np.vstack((np11, np22)))
print(np.hstack((np11, np22)))


# new dimension

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.stack((a, b)))

print(np.stack((a, b), axis=0))  # shape (2, 3)
print(np.stack((a, b), axis=1))  # shape (3, 2)


# column stack 

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.column_stack((a, b)))
# [[1 4]
#  [2 5]
#  [3 6]]