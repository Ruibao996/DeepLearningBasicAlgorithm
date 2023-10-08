import numpy as np


class LogisticRegression:
    def __init__(self, n_iter=200, eta=0.12, tol=None):
        self.n_iter = n_iter
        self.eta = eta
        self.tol = tol
        self.w = None
        self.loss = []
        self.iterTime = []

    def _z(self, X, w):
        return np.dot(X, w)

    def _sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def _predict_proba(self, X, w):
        # calculate y_proba in train function
        return self._sigmoid(self._z(X, w))

    def _loss(self, y, y_proba):
        m = y.size
        p = y_proba * (2*y - 1) + (1 - y)
        return -np.sum(np.log(p)) / m

    def _gradient(self, X, y, y_proba):
        return np.matmul((y_proba - y), X) / y.size

    def _gradient_descent(self, w, X, y):
        if self.tol is not None:
            loss_old = np.inf

        for step_i in range(self.n_iter):
            y_proba = self._predict_proba(X, w)

            loss = self._loss(y, y_proba)
            print('%4i Loss: %s' % (step_i, loss))

            if self.tol is not None:
                if loss_old - loss < self.tol:
                    break
                loss_old = loss

            grad = self._gradient(X, y, y_proba)
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
        X = self._preprocess_data_X(X)
        y_pred = self._predict_proba(X, self.w)
        return np.where(y_pred >= 0.5, 1, 0)
