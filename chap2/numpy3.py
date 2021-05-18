import numpy as np

arr = np.arange(10)
print('\n',arr[5])
print('\n',arr[5:8])
arr[5:8] = 12
print('\n',arr)


arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('\n', arr[:2])
print('\n', arr[:2, 1:])
