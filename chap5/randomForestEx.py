from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target , random_state=42)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(x_train, y_train)

print('\ntrain score:{:.3f}'.format(forest.score(x_train, y_train)))
print('test score:{:.3f}'.format(forest.score(x_test, y_test)))
