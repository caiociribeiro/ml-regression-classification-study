import numpy as np
import time

def k_fold_split(X, y, n_folds=5, random_state=42):
    X = np.array(X)
    y = np.array(y)

    indices = np.arange(len(X))
    np.random.seed(random_state)
    np.random.shuffle(indices)

    fold_sizes = np.full(n_folds, len(X) // n_folds)
    fold_sizes[:len(X) % n_folds] += 1

    folds = []
    start = 0

    for fold_size in fold_sizes:
        end = start + fold_size

        test_idx = indices[start:end]
        train_idx = np.concatenate((indices[:start], indices[end:]))

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        folds.append((X_train, X_test, y_train, y_test))
        start = end

    return folds


def cross_validation(model, X, y, metrics_fn, n_folds=5, random_state=42, **kwargs):
    folds = k_fold_split(X, y, n_folds, random_state)

    metrics_list = []
    train_times = []
    test_times = []

    for i, (train_idx, test_idx) in enumerate(folds):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        start = time.time()
        model.fit(X_train, y_train)
        train_times.append(time.time() - start)

        start = time.time()
        y_pred = model.predict(X_test)
        test_times.append(time.time() - start)

        metrics = metrics_fn(y_test, y_pred, **kwargs)
        metrics_list.append(metrics)

    return metrics_list, train_times, test_times