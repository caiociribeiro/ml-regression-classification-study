import openml
import numpy as np


def load_dataset(dataset_id, target_name):
    # pega o dataset pelo id
    dataset = openml.datasets.get_dataset(dataset_id)

    # extrai X, y e nomes de atributos
    X, y, _, attribute_names = dataset.get_data(target=target_name)

    # converte X para ndarray
    X_arr = np.asarray(X)

    # garante que X seja 2D
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    # converte y para ndarra
    y_arr = np.asarray(y)

    # tenta obter nomes de features a partir dos atributos
    feature_names = None
    if attribute_names:
        try:
            attr_list = list(attribute_names)
            # remove o target da lista de atributos
            if target_name in attr_list:
                attr_list = [n for n in attr_list if n != target_name]

            # usa os nomes se o comprimento bater com X
            if len(attr_list) == X_arr.shape[1]:
                feature_names = attr_list
        except Exception:
            feature_names = None

    # se features sem nomes, cria genericos
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X_arr.shape[1])]

    print("Dataset carregado:", getattr(dataset, "name", dataset_id))
    print("Shape de X:", X_arr.shape)
    print("Shape de y:", y_arr.shape)
    print("Target:", target_name)

    return X_arr, y_arr, feature_names