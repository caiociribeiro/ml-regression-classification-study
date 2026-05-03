import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X_train, y_train):
        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train, dtype=float)

        X_bias = np.c_[np.ones(X_train.shape[0]), X_train]

        self.weights = np.linalg.pinv(X_bias) @ y_train

    def predict(self, X_test):
        X_test = np.array(X_test, dtype=float)
        X_bias = np.c_[np.ones(X_test.shape[0]), X_test]

        return X_bias @ self.weights