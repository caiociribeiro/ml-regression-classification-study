import numpy as np

def mse(y_true, y_pred):
    # converte para float e calcula mean squared error
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    # raiz do MSE
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    # mean absolute error
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)


def r2_adjusted(y_true, y_pred, n_features):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    n_samples = len(y_true)
    r2 = r2_score(y_true, y_pred)

    return 1 - ((1 - r2) * (n_samples - 1)) / (n_samples - n_features - 1)


def regression_metrics(y_true, y_pred, n_features):
    # retorna dict com metricas de regressao
    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "r2_adjusted": r2_adjusted(y_true, y_pred, n_features),
    }


def classification_metrics(y_true, y_pred):
    # calcula accuracy, precision, recall e f1
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)

    classes = np.unique(y_true)
    precisions = []
    recalls = []

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        # evita divisao por zero adicionando valor muito pequeno ao denominador
        precisions.append(tp / (tp + fp + 1e-9))
        recalls.append(tp / (tp + fn + 1e-9))

    precision = np.mean(precisions)
    recall = np.mean(recalls)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }