import numpy as np


class OLSLinearRegression:

    def _ols(self, X, y):
        tmp = np.linalg.inv(np.matmul(X.T, X))
        tmp = np.matmul(tmp, X.T)
        return np.matmul(tmp, y)

    def _preprocess_data_X(self, X):
        m, n = X.shape
        X_ = np.empty((m, n+1))
        X_[:, 0] = 1
        X_[:, 1:] = X

        return X_

    def train(self, X_train, y_train):
        X_train = self._preprocess_data_X(X_train)

        self.w = self._ols(X_train, y_train)

    def predict(self, X):
        X = self._preprocess_data_X(X)
        return np.matmul(X, self.w)
