import numpy as np


class Preprocessor:
    def __init__(self):
        self.means = None
        self.stds = None
        self.category_maps = {}  # Mapeamento para colunas categóricas
        self.categorical_cols = []

    def _replace_empty_strings_with_nan(self, arr):
        if arr.dtype != object:
            return arr

        for col in range(arr.shape[1]):
            for i in range(arr.shape[0]):
                v = arr[i, col]
                if isinstance(v, str) and v.strip() == "":
                    arr[i, col] = np.nan
        return arr

    def _is_categorical_column(self, col_values):
        for v in col_values:
            if isinstance(v, str) and v.strip() != "":
                return True
        return False

    def fit(self, X):
        X = np.array(X, dtype=object)

        # Limpeza leve: strings vazias -> NaN
        X = self._replace_empty_strings_with_nan(X)

        n_samples, n_features = X.shape

        # Detecta colunas categóricas com base em qualquer valor string nao-vazio
        for col in range(n_features):
            col_vals = X[:, col]
            if self._is_categorical_column(col_vals):
                self.categorical_cols.append(col)

                # Preserve order of first appearance for deterministic mapping
                seen = set()
                unique_values = []
                for v in col_vals:
                    if isinstance(v, float) and np.isnan(v):
                        continue
                    if v not in seen:
                        seen.add(v)
                        unique_values.append(v)

                mapping = {val: idx for idx, val in enumerate(unique_values)}
                self.category_maps[col] = mapping

                # Apply mapping; unknowns -> -1, NaN stays NaN
                X[:, col] = [mapping.get(v, -1) if not (isinstance(v, float) and np.isnan(v)) else np.nan for v in col_vals]

        # Converte tudo para float (vai falhar com strings não mapeadas)
        try:
            X_float = X.astype(float)
        except Exception as e:
            raise ValueError(f"Falha ao converter X para float durante fit: {e}")

        # Preenche NaN com media
        self.means = np.nanmean(X_float, axis=0)
        self.means = np.where(np.isnan(self.means), 0.0, self.means)

        X_filled = np.where(np.isnan(X_float), self.means, X_float)

        # Desvio padrão
        self.stds = np.std(X_filled, axis=0)
        self.stds = np.where(self.stds == 0, 1.0, self.stds)

        return self

    def transform(self, X):
        X = np.array(X, dtype=object)

        # Limpeza leve: strings vazias -> NaN
        X = self._replace_empty_strings_with_nan(X)

        # Aplica mapeamentos de categorias; categorias desconhecidas -> -1
        for col in self.categorical_cols:
            mapping = self.category_maps.get(col, {})
            col_vals = X[:, col]
            X[:, col] = [mapping.get(v, -1) if not (isinstance(v, float) and np.isnan(v)) else np.nan for v in col_vals]

        try:
            X_float = X.astype(float)
        except Exception as e:
            raise ValueError(f"Falha ao converter X para float durante transform: {e}")

        if self.means is None or self.stds is None:
            raise RuntimeError("Preprocessor precisa ser 'fit' antes de 'transform'")

        # Preenche NaN com media calculada no fit
        X_filled = np.where(np.isnan(X_float), self.means, X_float)

        # Normalização (Z-score)
        X_scaled = (X_filled - self.means) / self.stds

        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)