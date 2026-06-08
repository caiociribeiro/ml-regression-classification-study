import time
import numpy as np
import json

from config import (
    DATASET_CLASSIFICATION_ID,
    CLASSIFICATION_TARGET_NAME,
    N_FOLDS,
    RANDOM_SEED,
    K_VALUES,
    ADALINE_CONFIGS,
)

from data.dataset_loader import load_dataset
from utils.preprocessing import Preprocessor
from utils.validation import k_fold_split
from utils.metrics import classification_metrics

from models.knn_classifier import KNNClassifier
from models.naive_bayes import NaiveBayes
from models.bayesian_multivariate import BayesianMultivariate
from models.adaline import Adaline


def summarize(values):
    values = np.array(values, dtype=float)
    return np.mean(values), np.std(values)


def run_model(model_name, model_factory, X, y):
    print(f"\nExecutando: {model_name}")

    folds = k_fold_split(X, y, n_folds=N_FOLDS, random_state=RANDOM_SEED)

    metric_results = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "f1": [],
    }

    classes = np.unique(y)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    confusion_sum = np.zeros((len(classes), len(classes)), dtype=int)

    train_times = []
    test_times = []

    for i, (X_train, X_test, y_train, y_test) in enumerate(folds):
        print(f"\n[{model_name}] Fold {i + 1}/{len(folds)}")

        pre = Preprocessor()
        X_train_s = pre.fit_transform(X_train)
        X_test_s = pre.transform(X_test)

        model = model_factory()

        start = time.time()
        model.fit(X_train_s, y_train)
        train_times.append(time.time() - start)

        start = time.time()
        y_pred = model.predict(X_test_s)
        test_times.append(time.time() - start)

        metrics = classification_metrics(y_test, y_pred)
        print("Métricas do fold:", metrics)

        for t, p in zip(y_test, y_pred):
            confusion_sum[class_to_idx[t], class_to_idx[p]] += 1

        for key in metric_results:
            metric_results[key].append(metrics[key])

    return {
        "model": model_name,
        "acc_mean": summarize(metric_results["accuracy"])[0],
        "acc_std": summarize(metric_results["accuracy"])[1],
        "precision_mean": summarize(metric_results["precision"])[0],
        "precision_std": summarize(metric_results["precision"])[1],
        "recall_mean": summarize(metric_results["recall"])[0],
        "recall_std": summarize(metric_results["recall"])[1],
        "specificity_mean": summarize(metric_results["specificity"])[0],
        "specificity_std": summarize(metric_results["specificity"])[1],
        "f1_mean": summarize(metric_results["f1"])[0],
        "f1_std": summarize(metric_results["f1"])[1],
        "train_time_mean": summarize(train_times)[0],
        "train_time_std": summarize(train_times)[1],
        "test_time_mean": summarize(test_times)[0],
        "test_time_std": summarize(test_times)[1],
        "confusion_matrix": confusion_sum.tolist(),
        "classes": classes.tolist(),
    }


def print_summary(results):
    print("\nRESULTADOS\n")

    col_model = 42
    col_metric = 20
    col_time = 22

    print(
        f"{'Modelo':<{col_model}} "
        f"{'Accuracy':<{col_metric}} "
        f"{'F1':<{col_metric}} "
        f"{'Tempo Treino (s)':<{col_time}} "
        f"{'Tempo Teste (s)':<{col_time}}"
    )
    print("-" * (col_model + col_metric * 2 + col_time * 2 + 4))

    for r in results:
        acc_str = f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f}"
        f1_str = f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
        train_str = f"{r['train_time_mean']:.4f} ± {r['train_time_std']:.4f}"
        test_str = f"{r['test_time_mean']:.4f} ± {r['test_time_std']:.4f}"
        print(
            f"{r['model']:<{col_model}} "
            f"{acc_str:<{col_metric}} "
            f"{f1_str:<{col_metric}} "
            f"{train_str:<{col_time}} "
            f"{test_str:<{col_time}}"
        )


def run_classification():
    print("\n===== CLASSIFICAÇÃO =====")

    X, y, _ = load_dataset(
        DATASET_CLASSIFICATION_ID, target_name=CLASSIFICATION_TARGET_NAME
    )

    results = []

    # KNN
    results.append(
        run_model(
            "KNN (Euclidiana)",
            lambda: KNNClassifier(k=K_VALUES[0], metric="euclidean"),
            X,
            y,
        )
    )
    results.append(
        run_model(
            "KNN (Manhattan)",
            lambda: KNNClassifier(k=K_VALUES[0], metric="manhattan"),
            X,
            y,
        )
    )

    # Bayesianos
    results.append(run_model("Naive Bayes", lambda: NaiveBayes(), X, y))
    results.append(
        run_model("Bayes Multivariado", lambda: BayesianMultivariate(), X, y)
    )

    # Adaline OvR (multiclasse)
    for config in ADALINE_CONFIGS:
        name = f"Adaline lr={config['lr']} epochs={config['epochs']}"
        results.append(
            run_model(
                name,
                lambda c=config: Adaline(lr=c["lr"], n_iter=c["epochs"]),
                X,
                y,
            )
        )

    print_summary(results)

    # Salva CSV de métricas
    import os
    import csv

    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "classification_metrics.csv")

    fieldnames = [
        "model",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "specificity_mean",
        "specificity_std",
        "f1_mean",
        "f1_std",
        "confusion_matrix",
        "train_time_mean",
        "train_time_std",
        "test_time_mean",
        "test_time_std",
    ]

    def _round_val(v):
        if v == "":
            return v
        try:
            if isinstance(v, (int, float)) or (
                hasattr(v, "astype") and getattr(v, "ndim", 0) == 0
            ):
                return round(float(v), 4)
        except Exception:
            pass
        return v

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "model": r["model"],
                    "accuracy_mean": _round_val(r["acc_mean"]),
                    "accuracy_std": _round_val(r["acc_std"]),
                    "precision_mean": _round_val(r.get("precision_mean", "")),
                    "precision_std": _round_val(r.get("precision_std", "")),
                    "recall_mean": _round_val(r.get("recall_mean", "")),
                    "recall_std": _round_val(r.get("recall_std", "")),
                    "specificity_mean": _round_val(r.get("specificity_mean", "")),
                    "specificity_std": _round_val(r.get("specificity_std", "")),
                    "f1_mean": _round_val(r["f1_mean"]),
                    "f1_std": _round_val(r["f1_std"]),
                    "confusion_matrix": json.dumps(r.get("confusion_matrix", [])),
                    "train_time_mean": _round_val(r["train_time_mean"]),
                    "train_time_std": _round_val(r["train_time_std"]),
                    "test_time_mean": _round_val(r["test_time_mean"]),
                    "test_time_std": _round_val(r["test_time_std"]),
                }
            )

    print(f"CSV salvo em: {csv_path}")

    # Salva CSV de matrizes de confusão
    if results:
        classes = results[0].get("classes", [])
        if classes:
            cm_csv_path = os.path.join("results", "confusion_matrices.csv")
            cm_fieldnames = ["model"] + [
                f"cm_{t}_{p}" for t in classes for p in classes
            ]
            with open(cm_csv_path, "w", newline="", encoding="utf-8") as cf:
                cm_writer = csv.DictWriter(cf, fieldnames=cm_fieldnames)
                cm_writer.writeheader()
                for r in results:
                    row = {"model": r["model"]}
                    cm = np.array(r.get("confusion_matrix", []), dtype=int)
                    for idx, val in enumerate(cm.flatten()):
                        row[cm_fieldnames[1 + idx]] = int(val)
                    cm_writer.writerow(row)
            print(f"CSV de matrizes de confusão salvo em: {cm_csv_path}")
