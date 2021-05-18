from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target , random_state=0)
gb = GradientBoostingClassifier(max_depth=1, random_state=0)
gb.fit(x_train, y_train)

print('\ntrain score:{:.3f}'.format(gb.score(x_train, y_train)))
print('test score:{:.3f}'.format(gb.score(x_test, y_test)))
