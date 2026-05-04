import numpy as np
from .kd_tree import KDTree


class KNNClassifier:
    def __init__(self, k=5, metric="euclidean", use_kdtree=True):
        self.k = k
        self.metric = metric
        self.use_kdtree = use_kdtree
        self.tree = None

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train, dtype=float)
        self.y_train = np.array(y_train)

        if self.use_kdtree:
            self.tree = KDTree(self.X_train)
            print("Usando KD-tree")

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

        predictions = []
        total = len(X_test)

        if self.use_kdtree and self.tree is not None:
            p = 2 if self.metric == "euclidean" else 1
            k_eff = min(self.k, len(self.X_train))
            dists, idxs = self.tree.query(X_test, k=k_eff, p=p)
            if k_eff == 1:
                idxs = idxs.reshape(-1, 1)

            for i, neighbors in enumerate(idxs):
                if i % 500 == 0:
                    print(f"Predizendo amostra {i+1}/{total}")

                labels = self.y_train[neighbors]
                unique, counts = np.unique(labels, return_counts=True)
                predictions.append(unique[np.argmax(counts)])
        else:
            for i, x in enumerate(X_test):
                if i % 500 == 0:
                    print(f"Predizendo amostra {i+1}/{total}")

                distances = [self._distance(x, x_train) for x_train in self.X_train]
                k_indices = np.argsort(distances)[:self.k]

                labels = [self.y_train[i] for i in k_indices]
                unique, counts = np.unique(labels, return_counts=True)
                predictions.append(unique[np.argmax(counts)])

        print(f"Predição completa ({total} amostras)")

        return np.array(predictions)