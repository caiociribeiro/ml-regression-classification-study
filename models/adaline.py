import numpy as np


# Adaline One-vs-Rest (OvR) para classificação multiclasse
# predicao escolhe a classe cujo classificador retorna o maior score
class Adaline:
    def __init__(self, lr=0.01, n_iter=50, fit_intercept=True):
        self.lr = lr
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.classifiers_ = []
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        self.classifiers_ = []

        for c in self.classes_:
            y_bin = np.where(y == c, 1.0, -1.0)
            clf = Adaline(
                lr=self.lr,
                n_iter=self.n_iter,
                fit_intercept=self.fit_intercept,
            )
            clf.fit(X, y_bin)
            self.classifiers_.append(clf)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # matriz (n_samples, n_classes) com os scores de cada classificador
        scores = np.column_stack(
            [clf.decision_function(X) for clf in self.classifiers_]
        )
        # vence a classe com maior score
        return self.classes_[np.argmax(scores, axis=1)]
