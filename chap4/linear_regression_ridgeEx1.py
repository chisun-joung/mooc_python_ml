from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import mglearn

x, y = mglearn.datasets.load_extended_boston()
x_train, x_test, y_train, y_test = train_test_split(x, y , random_state=0)
ridge = Ridge(alpha=10).fit(x_train, y_train)
print('\ntrain score:{:.2f}'.format(ridge.score(x_train, y_train)))
print('test score:{:.2f}'.format(ridge.score(x_test, y_test)))
