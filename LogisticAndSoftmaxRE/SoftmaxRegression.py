import numpy as np


class SoftmaxRegression:
    def __init__(self, n_iter=200, eta=0.12, tol=None):
        self.n_iter = n_iter
        self.eta = eta
        self.tol = tol
        self.W = None

    def _z(self, X, W):
        if X.ndim == 1:
            return np.dot(W, X)
        return np.matmul(X, W.T)

    def _softmax(self, Z):
        E = np.exp(Z)
        if Z.ndim == 1:
            return E / np.sum(E)
        return E / np.sum(E, axis=1, keepdims=True)

    def _predict_proba(self, X, W):
        Z = self._z(X, W)
        return self._softmax(Z)

    def _loss(self, y, y_proba):
        m = y.size
        p = y_proba[range(m), 0]
        return -np.sum(np.log(p)) / m

    def _gradient(self, xi, yi, yi_proba):
        yi = int(yi)
        K = yi_proba.size
        y_bin = np.zeros(K)
        y_bin[yi] = 1

        return (yi_proba - y_bin)[:, None] * xi

    def _stochastic_gradient_descent(self, W, X, y):
        if self.tol is not None:
            loss_old = np.inf
            end_count = 0

        m = y.size
        idx = np.arange(m)
        for step_i in range(self.n_iter):
            y_proba = self._predict_proba(X, W)
            loss = self._loss(y, y_proba)
            print('%4i Loss: %s' % (step_i, loss))

            if self.tol is not None:
                if loss_old - loss < self.tol:
                    end_count += 1
                    if end_count == 5:
                        break
                else:
                    end_count = 0
                loss_old = loss

            np.random.shuffle(idx)
            for i in idx:
                yi_proba = self._predict_proba(X[i], W)
                grad = self._gradient(X[i], y[i], yi_proba)
                W -= self.eta * grad

    def _preprocess_data_X(self, X):
        m, n = X.shape
        X_ = np.empty((m, n+1))
        X_[:, 0] = 1
        X_[:, 1:] = X

        return X_

    def train(self, X_train, y_train):
        X_train = self._preprocess_data_X(X_train)

        k = np.unique(y_train).size
        _, n = X_train.shape
        self.W = np.random.random((k, n)) * 0.05

        self._stochastic_gradient_descent(self.W, X_train, y_train)

    def predict(self, X):
        X = self._preprocess_data_X(X)
        Z = self._z(X, self.W)
        return np.argmax(Z, axis=1)
