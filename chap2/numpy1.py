import numpy as np

data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
print(arr1,'\n')
print(data1,'\n')

data2 = [[1,2,3,4], [5,6,7,8]]
arr2 = np.array(data2)
print(arr2,'\n')

print(arr2.ndim,'\n')
print(arr2.shape,'\n')
print(arr2.dtype,'\n')

print(np.arange(14))

arr = np.array([1,2,3,4,5], dtype=np.int32)
print(arr.dtype, '\n')

float_arr = arr.astype(np.float32);
print(float_arr.dtype,'\n')



