import numpy as np
np.random.seed(12345)

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
print('\n',names,"\n")
print('\n',data)

print('\n',names == 'Bob')
print('\n',data[names == 'Bob'])


print('\n',data[names == 'Bob', 2:])
print('\n',data[names == 'Bob', 3])
