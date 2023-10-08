from SoftmaxRegression import SoftmaxRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression
import numpy as np

X = np.genfromtxt(r'C:\Users\Rui\Desktop\Deeplearning\BasicAlgorithm\LogisticAndSoftmaxRE\archive.ics.uci.edu_ml_machine-learning-databases_wine_wine.data.txt',
                  delimiter=',', usecols=range(1, 14))
y = np.genfromtxt(r'C:\Users\Rui\Desktop\Deeplearning\BasicAlgorithm\LogisticAndSoftmaxRE\archive.ics.uci.edu_ml_machine-learning-databases_wine_wine.data.txt', delimiter=',', usecols=0)


# logistic
idx = y != 3  # remove class 3

X = X[idx]
y = y[idx]

y -= 1

clf = LogisticRegression(n_iter=2000, eta=0.01, tol=0.0001)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# 梯度计算前，先将特征值范围缩小，使其均一
ss = StandardScaler()
ss.fit(X_train)  # 计算每一列均值和方差

X_train_std = ss.transform(X_train)
X_test_std = ss.transform(X_test)

clf.train(X_train_std, y_train)

y_pred = clf.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# softmax
X = np.genfromtxt(r'C:\Users\Rui\Desktop\Deeplearning\BasicAlgorithm\LogisticAndSoftmaxRE\archive.ics.uci.edu_ml_machine-learning-databases_wine_wine.data.txt',
                  delimiter=',', usecols=range(1, 14))
y = np.genfromtxt(r'C:\Users\Rui\Desktop\Deeplearning\BasicAlgorithm\LogisticAndSoftmaxRE\archive.ics.uci.edu_ml_machine-learning-databases_wine_wine.data.txt', delimiter=',', usecols=0)

y -= 1

clf = SoftmaxRegression(n_iter=2000, eta=0.01, tol=0.0001)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# 梯度计算前，先将特征值范围缩小，使其均一
ss = StandardScaler()
ss.fit(X_train)  # 计算每一列均值和方差

X_train_std = ss.transform(X_train)
X_test_std = ss.transform(X_test)

clf.train(X_train_std, y_train)

y_pred = clf.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
