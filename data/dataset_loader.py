import openml
import numpy as np


def load_dataset(dataset_id, target_name):
    dataset = openml.datasets.get_dataset(dataset_id)

    X, y, _, attribute_names = dataset.get_data(
        target=target_name
    )

    X = np.array(X)
    y = np.array(y)

    print("Dataset carregado:", dataset.name)
    print("Shape de X:", X.shape)
    print("Shape de y:", y.shape)
    print("Target:", target_name)

    return X, y, attribute_names