import numpy as np
# regressao linear multipla usando pseudo-inversa
class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X_train, y_train):
        # converte inputs para float
        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train, dtype=float)

        # adiciona coluna de 1s
        X_bias = np.c_[np.ones(X_train.shape[0]), X_train]

        # calcula pesos usando pseudo-inversa
        self.weights = np.linalg.pinv(X_bias) @ y_train

    def predict(self, X_test):
        X_test = np.array(X_test, dtype=float)
        X_bias = np.c_[np.ones(X_test.shape[0]), X_test]

        return X_bias @ self.weights