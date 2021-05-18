from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import mglearn

cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
x_scaled = scaler.transform(cancer.data)

pca = PCA(n_components=2)
pca.fit(x_scaled)
x_pca = pca.transform(x_scaled)
print('real data shape:{}'.format(str(x_scaled.shape)))
print('reduced data shape:{}'.format(str(x_pca.shape)))

plt.figure(figsize=(8,8))
mglearn.discrete_scatter(x_pca[:, 0], x_pca[:, 1], cancer.target)
plt.legend(['positive','negative'], loc='best')
plt.gca().set_aspect('equal')
plt.xlabel('first principal component')
plt.ylabel('second pricipal component')
plt.show()
