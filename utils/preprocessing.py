import numpy as np


class Preprocessor:
    def __init__(self):
        self.means = None
        self.stds = None
        self.category_maps = {} # Mapeamento para colunas categóricas
        self.categorical_cols = []

    def fit(self, X):
        X = np.array(X, dtype=object)

        n_samples, n_features = X.shape

        # Detecta colunas categóricas
        for col in range(n_features):
            if isinstance(X[0, col], str):
                self.categorical_cols.append(col)

                unique_values = list(set(X[:, col]))
                mapping = {val: idx for idx, val in enumerate(unique_values)}
                self.category_maps[col] = mapping

                # Aplica conversão
                X[:, col] = [mapping[val] for val in X[:, col]]

        # Converte tudo para float
        X = X.astype(float)

        # Preenche NaN com media
        self.means = np.nanmean(X, axis=0)
        self.means = np.where(np.isnan(self.means), 0.0, self.means)

        X_filled = np.where(np.isnan(X), self.means, X)

        # Desvio padrão
        self.stds = np.std(X_filled, axis=0)
        self.stds = np.where(self.stds == 0, 1.0, self.stds)

        return self

    def transform(self, X):
        X = np.array(X, dtype=object)

        for col in self.categorical_cols:
            mapping = self.category_maps[col]
            X[:, col] = [mapping.get(val, 0) for val in X[:, col]]

        X = X.astype(float)

        # Preenche NaN com media
        X_filled = np.where(np.isnan(X), self.means, X)

        # Normalização (Z-score) 
        X_scaled = (X_filled - self.means) / self.stds

        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)