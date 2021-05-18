from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 10*np.sin(x)
x = np.arange(-10, 10, 0.1)
plt.plot(x, f(x))
plt.show()

result = optimize.minimize(f, 4)
x0 = result['x']
plt.plot(x, f(x));
plt.hold(True)
plt.scatter(x0, f(x0), s=200)
plt.show()