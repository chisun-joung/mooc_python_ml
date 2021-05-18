from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import mglearn
import matplotlib.pyplot as plt
import numpy as np

S = mglearn.datasets.make_signals()
plt.figure(figsize=(6, 1))
plt.plot(S, '-')
plt.xlabel("time")
plt.ylabel("signal")
plt.margins(0)
plt.show()

A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)
print("measured signal shape:", X.shape)

nmf = NMF(n_components=3, random_state=42)
S_ = nmf.fit_transform(X)
print("restored signal shape:", S_.shape)

pca = PCA(n_components=3)
H = pca.fit_transform(X)

models = [X, S, S_, H]
names = ['measured signal',
         'original signal',
         'signal with NMF',
         'signal with PCA']

fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hspace': .5},
                         subplot_kw={'xticks': (), 'yticks': ()})

for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:, :3], '-')
    ax.margins(0)
plt.show()
