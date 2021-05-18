import os
import mglearn
import pandas as pd

data = pd.read_csv(
    os.path.join(mglearn.datasets.DATA_PATH, "adult.data"), header=None, index_col=False,
    names=['age', 'workclass', 'fnlwgt', 'education',  'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income'])
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week','occupation', 'income']]
print(data.head())

print(data.gender.value_counts())

print("\noriginal attr:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("get_dummies attr:\n", list(data_dummies.columns))

print('\n', data_dummies.iloc[0])

