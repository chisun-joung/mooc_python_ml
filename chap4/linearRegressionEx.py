from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mglearn
#x, y = mglearn.datasets.make_wave(n_samples=60)
x, y = mglearn.datasets.load_extended_boston()
x_train, x_test, y_train, y_test = train_test_split(x, y , random_state=0)
lrModel = LinearRegression()
trainModel = lrModel.fit(x_train, y_train) #y = wx + b
print("weight(coefficient):{}".format(trainModel.coef_))
print('intercept:{}'.format(trainModel.intercept_))

print('\ntrain score:{:.2f}'.format(trainModel.score(x_train, y_train)))
print('test score:{:.2f}'.format(trainModel.score(x_test, y_test)))



