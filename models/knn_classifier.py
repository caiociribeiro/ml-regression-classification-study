import numpy as np

class KNNClassifier:
    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train, dtype=float)
        self.y_train = np.array(y_train)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def _distance(self, x1, x2):
        if self.metric == "euclidean":
            return self._euclidean_distance(x1, x2)
        return self._manhattan_distance(x1, x2)

    def predict(self, X_test):
        X_test = np.array(X_test, dtype=float)
        return np.array([self._predict_single(x) for x in X_test])

    def _predict_single(self, x):
        distances = [self._distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]

        labels = [self.y_train[i] for i in k_indices]
        unique, counts = np.unique(labels, return_counts=True)

        return unique[np.argmax(counts)]