from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import mglearn
import matplotlib.pyplot as plt
import numpy as np

X, y = mglearn.datasets.make_wave(n_samples=120)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

bins = np.linspace(-3, 3, 11)
whitch_bin = np.digitize(X, bins=bins)

encoder = OneHotEncoder(sparse=False)
encoder.fit(whitch_bin)
x_binned = encoder.transform(whitch_bin)

poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)
print("\nX_poly.shape: {}".format(X_poly.shape))
print("\nX:\n{}".format(X[:5]))
print("\nX_poly:\n{}".format(X_poly[:5]))
print("\nfeature names:\n{}".format(poly.get_feature_names()))


line_binned = encoder.transform(np.digitize(line, bins=bins))

reg = LinearRegression().fit(X_poly, y)
line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly), label='poly linear')
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("linear output")
plt.xlabel("input attr")
plt.legend(loc="best")
plt.show()

