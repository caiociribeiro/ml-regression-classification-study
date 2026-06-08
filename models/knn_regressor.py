import numpy as np


# knn regressor com opcao de usar kd-tree
class KNNRegressor:
    def __init__(self, k=5, metric="euclidean"):
        # numero de vizinhos, metrica de distancia e flag para uso de kd-tree
        self.k = k
        self.metric = metric

    def fit(self, X_train, y_train):
        # armazena treinamento como ndarray de float
        self.X_train = np.array(X_train, dtype=float)
        self.y_train = np.array(y_train, dtype=float)

    def _euclidean_distance(self, x1, x2):
        # dist euclidiana
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1, x2):
        # dist manhattan
        return np.sum(np.abs(x1 - x2))

    def get_distance(self, x1, x2):
        # retorna baseado na metrica escolhida
        if self.metric == "euclidean":
            return self._euclidean_distance(x1, x2)
        return self._manhattan_distance(x1, x2)

    def _predict_single(self, x):
        # distancias ate todos os pontos de treino e media dos k mais proximos
        distances = [self.get_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[: self.k]
        k_values = self.y_train[k_indices]
        return np.mean(k_values)

    def predict(self, X_test):
        # prediz para cada amostra de X_test
        X_test = np.array(X_test, dtype=float)

        predictions = []
        total = len(X_test)

        # busca por força bruta (todos os pontos)
        for i, x in enumerate(X_test):
            if i % 500 == 0:
                print(f"Predizendo amostra {i + 1}/{total}")

            predictions.append(self._predict_single(x))

        print(f"Predição completa ({total} amostras)")

        return np.array(predictions)
