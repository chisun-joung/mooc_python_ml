import numpy as np 

arr = np.arange(8)
arr2 = arr.reshape((4, 2))
print(arr2)

arr3 = arr.reshape((4, 2)).reshape((2, 4))
print('\n',arr3)
