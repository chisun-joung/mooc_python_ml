import scipy as sp
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
x = np.linspace(-3, 3, 1000)
y1 = sp.special.erf(x)
a = plt.subplot(211)
plt.plot(x, y1)
plt.title("erf func")
a.xaxis.set_ticklabels([])
y2 = sp.special.expit(x)
plt.subplot(212)
plt.plot(x, y2)
plt.title("logistic")
plt.show()

