from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import mglearn
import matplotlib.pyplot as plt

x, y = make_blobs(random_state=1)
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(x)

mglearn.discrete_scatter(x[:, 0], x[:, 1], assignment)
plt.legend(['cluster 0', 'cluster1', 'cluster2'], loc='best')
plt.xlabel('attr 0')
plt.ylabel('attr 1')
plt.show()
