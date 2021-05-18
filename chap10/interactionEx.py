from sklearn.linear_model import LinearRegression
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

#X_combined = np.hstack([X, x_binned])
X_product = np.hstack([x_binned, X * x_binned])

line_binned = encoder.transform(np.digitize(line, bins=bins))

'''
reg = LinearRegression().fit(X_combined, y)
line_combined = np.hstack([line, line_binned])
plt.plot(line, reg.predict(line_combined), label='linear regression')

'''
reg = LinearRegression().fit(X_product, y)
line_product = np.hstack([line_binned, line * line_binned])
plt.plot(line, reg.predict(line_product), label='linear regression2')


for bin in bins:
    plt.plot([bin, bin], [-3, 3], ':', c='k', linewidth=1)
plt.legend(loc="best")
plt.ylabel("linear output")
plt.xlabel("input attr")
plt.plot(X[:, 0], y, 'o', c='k')
plt.show()
