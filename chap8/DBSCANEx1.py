from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import mglearn
import matplotlib.pyplot as plt

x, y = make_blobs(random_state=0, n_samples=12)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(x)
print('cluster label:\n{}\n'.format(clusters))

mglearn.plots.plot_dbscan()
plt.show()
