from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target , random_state=0)

svm = SVC(kernel='rbf', C=10, gamma=0.1)
svm.fit(x_train, y_train)

print('\ntrain score:{:.3f}'.format(svm.score(x_train, y_train)))
print('test score:{:.3f}'.format(svm.score(x_test, y_test)))


import matplotlib.pyplot as plt
plt.boxplot(x_train, manage_ticks=False)
plt.yscale('symlog')
plt.xlabel('attr list')
plt.ylabel('attr size')
plt.show()

min_on_traing = x_train.min(axis=0)
range_on_traing = (x_train - min_on_traing).max(axis=0)
x_train_scaled = (x_train - min_on_traing) / range_on_traing
x_test_scaled = (x_test - min_on_traing) / range_on_traing
svc = SVC(kernel='rbf', C=10, gamma=0.1)
svc.fit(x_train_scaled, y_train)

print('\ntrain score:{:.3f}'.format(svc.score(x_train_scaled, y_train)))
print('test score:{:.3f}'.format(svc.score(x_test_scaled, y_test)))

