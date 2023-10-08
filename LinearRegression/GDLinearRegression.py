import numpy as np


class GDLinearRegression:
    def __init__(self, n_iter=200, eta=0.12, tol=None):
        self.n_iter = n_iter
        self.eta = eta
        self.tol = tol
        self.w = None

    def _loss(self, y, y_pred):
        return np.sum((y_pred - y)**2) / y.size

    def _gradient(self, X, y, y_pred):
        return np.matmul(y_pred - y, X) / y.size

    def _predict(self, X, w):
        return np.matmul(X, w)

    def _gradient_descent(self, w, X, y):
        if self.tol is not None:
            loss_old = np.inf

        for step_i in range(self.n_iter):
            y_pred = self._predict(X, w)

            loss = self._loss(y, y_pred)
            print('%4i Loss: %s' % (step_i, loss))

            if self.tol is not None:
                if loss_old - loss < self.tol:
                    break
                loss_old = loss

            grad = self._gradient(X, y, y_pred)
            w -= self.eta * grad

    def _preprocess_data_X(self, X):
        m, n = X.shape
        X_ = np.empty((m, n+1))
        X_[:, 0] = 1
        X_[:, 1:] = X

        return X_

    def train(self, X_train, y_train):
        X_train = self._preprocess_data_X(X_train)

        _, n = X_train.shape
        self.w = np.random.random(n) * 0.05

        self._gradient_descent(self.w, X_train, y_train)

    def predict(self, X):
        return self._predict(self._preprocess_data_X(X), self.w)
