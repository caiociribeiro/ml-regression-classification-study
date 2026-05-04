import time
import numpy as np

from config import (
    DATASET_CLASSIFICATION_ID,
    CLASSIFICATION_TARGET_NAME,
    N_FOLDS,
    RANDOM_SEED,
    K_VALUES
)

from data.dataset_loader import load_dataset
from utils.preprocessing import Preprocessor
from utils.validation import k_fold_split
from utils.metrics import classification_metrics

from models.knn_classifier import KNNClassifier
from models.naive_bayes import NaiveBayes
from models.bayesian_multivariate import BayesianMultivariate


def summarize(values):
    values = np.array(values, dtype=float)
    return np.mean(values), np.std(values)


def run_model(model_name, model, X, y):
    print(f"\n===== Executando: {model_name} =====")

    folds = k_fold_split(X, y, n_folds=N_FOLDS, random_state=RANDOM_SEED)

    metric_results = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    train_times = []
    test_times = []

    for i, (X_train, X_test, y_train, y_test) in enumerate(folds):
        print(f"\n[{model_name}] Fold {i+1}/{len(folds)}")

        pre = Preprocessor()
        X_train = pre.fit_transform(X_train)
        X_test = pre.transform(X_test)

        # TREINO
        start_train = time.time()
        model.fit(X_train, y_train)
        end_train = time.time()

        # TESTE
        start_test = time.time()
        y_pred = model.predict(X_test)
        end_test = time.time()

        metrics = classification_metrics(y_test, y_pred)

        print("Métricas do fold:", metrics)

        for key in metric_results:
            metric_results[key].append(metrics[key])

        train_times.append(end_train - start_train)
        test_times.append(end_test - start_test)

    return {
        "model": model_name,
        "acc_mean": summarize(metric_results["accuracy"])[0],
        "acc_std": summarize(metric_results["accuracy"])[1],
        "f1_mean": summarize(metric_results["f1"])[0],
        "f1_std": summarize(metric_results["f1"])[1],
        "train_time_mean": summarize(train_times)[0],
        "train_time_std": summarize(train_times)[1],
        "test_time_mean": summarize(test_times)[0],
        "test_time_std": summarize(test_times)[1],
    }

def print_summary(results):
    print("\n===== RESULTADOS =====\n")

    col_model = 25
    col_metric = 20
    col_time = 22

    print(
        f"{'Modelo':<{col_model}} "
        f"{'Accuracy':<{col_metric}} "
        f"{'F1':<{col_metric}} "
        f"{'Tempo Treino (s)':<{col_time}} "
        f"{'Tempo Teste (s)':<{col_time}}"
    )

    print("-" * (col_model + col_metric*2 + col_time*2 + 4))

    for r in results:
        print(
            f"{r['model']:<{col_model}} "
            f"{f'{r['acc_mean']:.4f} ± {r['acc_std']:.4f}':<{col_metric}} "
            f"{f'{r['f1_mean']:.4f} ± {r['f1_std']:.4f}':<{col_metric}} "
            f"{f'{r['train_time_mean']:.4f} ± {r['train_time_std']:.4f}':<{col_time}} "
            f"{f'{r['test_time_mean']:.4f} ± {r['test_time_std']:.4f}':<{col_time}}"
        )

def run_classification(use_kdtree=False):
    print("\n===== CLASSIFICAÇÃO =====")

    X, y, _ = load_dataset(
        DATASET_CLASSIFICATION_ID,
        target_name=CLASSIFICATION_TARGET_NAME
    )

    results = []

    results.append(
        run_model(
            "KNN (Euclidiana)",
            KNNClassifier(k=K_VALUES[0], metric="euclidean", use_kdtree=use_kdtree),
            X,
            y
        )
    )

    results.append(
        run_model(
            "KNN (Manhattan)",
            KNNClassifier(k=K_VALUES[0], metric="manhattan", use_kdtree=use_kdtree),
            X,
            y
        )
    )

    results.append(run_model("Naive Bayes", NaiveBayes(), X, y))
    results.append(run_model("Bayes Multivariado", BayesianMultivariate(), X, y))

    print_summary(results)
