import pandas as pd

df = pd.read_csv('ex1.csv')
print(df)

print('\n',pd.read_csv('ex2.csv', header=None),'\n')
print('\n',pd.read_csv('ex2.csv', names=['a','b','c','d','message']),'\n')

names=['a','b','c','d','message']
print('\n',pd.read_csv('ex2.csv', names=names, index_col='message'),'\n')

parsed = pd.read_csv('csv_mindex.csv', index_col=['key1','key2'])
print('\n',parsed,'\n')

print('\n',pd.read_csv('ex4.csv', skiprows=[0,2,3]),'\n')



