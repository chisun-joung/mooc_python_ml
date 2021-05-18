from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression()

scores = cross_val_score(logreg, iris.data, iris.target)
print("\ncross validation score:", scores)

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("\ncross validation score:", scores)

print("\ncross validation mean score:{:.2f}".format(scores.mean()))
