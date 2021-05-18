from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import mglearn
import matplotlib.pyplot as plt
import numpy as np

X, y = mglearn.datasets.make_wave(n_samples=120)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

bins = np.linspace(-3, 3, 11)
print('bins:{}'.format(bins))

whitch_bin = np.digitize(X, bins=bins)
print('\ndata point:\n', X[:5])
print('\ndata point bins:\n', whitch_bin[:5])

encoder = OneHotEncoder(sparse=False)
encoder.fit(whitch_bin)
x_binned = encoder.transform(whitch_bin)
print('\n',x_binned[:5],'\n')
print("X_binned.shape: {}".format(x_binned.shape))

line_binned = encoder.transform(np.digitize(line, bins=bins))

reg = LinearRegression().fit(x_binned, y)
plt.plot(line, reg.predict(line_binned), label='binning linear regression')

reg = DecisionTreeRegressor(min_samples_split=3).fit(x_binned, y)
plt.plot(line, reg.predict(line_binned), '--', label='binning decisiontree')
plt.plot(X[:, 0], y, 'o', c='k')
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.legend(loc="best")
plt.ylabel("linear output")
plt.xlabel("input attr0")
plt.show()

