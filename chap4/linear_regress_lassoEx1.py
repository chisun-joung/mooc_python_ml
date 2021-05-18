from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import mglearn

x, y = mglearn.datasets.load_extended_boston()
x_train, x_test, y_train, y_test = train_test_split(x, y , random_state=0)
lasso = Lasso(alpha=0.01, max_iter=100000).fit(x_train, y_train)
print('\ntrain score:{:.2f}'.format(lasso.score(x_train, y_train)))
print('test score:{:.2f}'.format(lasso.score(x_test, y_test)))
