import numpy as np

class Preprocessor:
    def __init__(self):
        # medias e desvios calculados no fit
        self.means = None
        self.stds = None
        # mapas para colunas categoricas {col_index: {valor: código}}
        self.category_maps = {}
        # lista de indices de colunas categoricas
        self.categorical_cols = []

    def _replace_empty_strings_with_nan(self, arr):
        # se nao for array de objetos, nao altera
        if arr.dtype != object:
            return arr

        # substitui strings vazias por NaN para facilitar tratamento
        for col in range(arr.shape[1]):
            for i in range(arr.shape[0]):
                v = arr[i, col]
                if isinstance(v, str) and v.strip() == "":
                    arr[i, col] = np.nan
        return arr

    def _is_categorical_column(self, col_values):
        # verifica se existe ao menos uma string nao vazia
        for v in col_values:
            if isinstance(v, str) and v.strip() != "":
                return True
        return False

    def fit(self, X):
        # converte para array de objetos para lidar com strings/NaN
        X = np.array(X, dtype=object)

        # normaliza strings vazias para NaN
        X = self._replace_empty_strings_with_nan(X)

        n_samples, n_features = X.shape

        # detecta colunas categoricas e cria mapeamento para inteiros
        for col in range(n_features):
            col_vals = X[:, col]
            if self._is_categorical_column(col_vals):
                self.categorical_cols.append(col)

                # coleta valores unicos
                # ignora NaN
                seen = set()
                unique_values = []
                for v in col_vals:
                    if isinstance(v, float) and np.isnan(v):
                        continue
                    if v not in seen:
                        seen.add(v)
                        unique_values.append(v)

                # mapeia cada valor para um inteiro (0..k-1)
                mapping = {val: idx for idx, val in enumerate(unique_values)}
                self.category_maps[col] = mapping

                # substitui os valores pela codificacao
                # novo valor vira -1
                X[:, col] = [mapping.get(v, -1) if not (isinstance(v, float) and np.isnan(v)) else np.nan for v in col_vals]

        # tenta converter tudo para float para calcular medias e desvios
        try:
            X_float = X.astype(float)
        except Exception as e:
            raise ValueError(f"Falha ao converter X para float durante fit: {e}")

        # calcula medias ignorando NaN
        self.means = np.nanmean(X_float, axis=0)
        # se coluna inteira for NaN, usa 0.0 como fallback
        self.means = np.where(np.isnan(self.means), 0.0, self.means)

        # preenche NaN com as medias para calcular desvio
        X_filled = np.where(np.isnan(X_float), self.means, X_float)

        # calcula desvios
        # se zero, usa 1 para evitar divisao por zero
        self.stds = np.std(X_filled, axis=0)
        self.stds = np.where(self.stds == 0, 1.0, self.stds)

        return self

    def transform(self, X):
        # prepara X como array de objetos e normaliza strings vazias
        X = np.array(X, dtype=object)

        X = self._replace_empty_strings_with_nan(X)

        # aplica mapeamentos de categorias definidos no fit
        for col in self.categorical_cols:
            mapping = self.category_maps.get(col, {})
            col_vals = X[:, col]
            # valores desconhecidos viram -1
            # NaN nao altera para evitar perder a informacao de valor ausente
            X[:, col] = [mapping.get(v, -1) if not (isinstance(v, float) and np.isnan(v)) else np.nan for v in col_vals]

        # converte para float antes de normalizar
        try:
            X_float = X.astype(float)
        except Exception as e:
            raise ValueError()

        # requer que fit tenha sido executado antes
        if self.means is None or self.stds is None:
            raise RuntimeError()

        # preenche valores ausentes com as medias calculadas no fit
        X_filled = np.where(np.isnan(X_float), self.means, X_float)

        # aplica padronização por Z-score
        X_scaled = (X_filled - self.means) / self.stds

        return X_scaled

    def fit_transform(self, X):
        # executa fit e depois transform
        self.fit(X)
        return self.transform(X)