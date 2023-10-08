from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from OLSLinearRegression import OLSLinearRegression
from GDLinearRegression import GDLinearRegression
import numpy as np
# Prepare Data
data = np.genfromtxt(
    r'C:\Users\Rui\Desktop\Deeplearning\BasicAlgorithm\LinearRegression\winequality-red.csv', delimiter=';', skip_header=True)
X = data[:, :-1]
y = data[:, -1]

# OLSmodel
ols_lr = OLSLinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

ols_lr.train(X_train, y_train)
y_pred = ols_lr.predict(X_test)
print(y_pred)
# MSE
mse = mean_squared_error(y_test, y_pred)
print(mse)
# MAE
mae = mean_absolute_error(y_test, y_pred)
print(mae)

# GDmodel
gd_lr = GDLinearRegression(n_iter=3000, eta=0.001, tol=0.00001)

gd_lr.train(X_train, y_train)
# Loss doesn't decrase, we need decrease eta
# gd_lr = GDLinearRegression(n_iter=3000, eta=0.0001, tol=0.00001)

# gd_lr.train(X_train, y_train)

# different diagram has different dimension, we need change them to fit one eta
ss = StandardScaler()
ss.fit(X_train)

X_train_std = ss.transform(X_train)
X_test_std = ss.transform(X_test)
gd_lr = GDLinearRegression(n_iter=5000, eta=0.001, tol=0.00001)
gd_lr.train(X_train_std, y_train)

y_pred = gd_lr.predict(X_test_std)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(mse)
print(mae)
