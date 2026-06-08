import time
import numpy as np

from config import (
    DATASET_REGRESSION_ID,
    REGRESSION_TARGET_NAME,
    N_FOLDS,
    RANDOM_SEED,
    K_VALUES,
    MLP_CONFIGS,
)

from data.dataset_loader import load_dataset
from utils.preprocessing import Preprocessor
from utils.validation import k_fold_split
from utils.metrics import regression_metrics

from models.knn_regressor import KNNRegressor
from models.linear_regression import LinearRegression
from models.mlp import MLPRegressor


def summarize(values):
    values = np.array(values, dtype=float)
    return np.mean(values), np.std(values)


def run_model(model_name, model_factory, X, y, normalize_y=False):
    print(f"\nExecutando: {model_name}")

    folds = k_fold_split(X, y, n_folds=N_FOLDS, random_state=RANDOM_SEED)

    metric_results = {"mse": [], "rmse": [], "mae": [], "r2": [], "r2_adjusted": []}
    train_times = []
    test_times = []

    for i, (X_train, X_test, y_train, y_test) in enumerate(folds):
        print(f"\n[{model_name}] Fold {i + 1}/{len(folds)}")

        preprocessor = Preprocessor()
        X_train_s = preprocessor.fit_transform(X_train)
        X_test_s = preprocessor.transform(X_test)

        # FIX: normalize_y parametrizado em vez de if "MLP" in model_name
        if normalize_y:
            y_mean = np.mean(y_train)
            y_std = np.std(y_train) or 1.0
            y_train_fit = (y_train - y_mean) / y_std
        else:
            y_train_fit = y_train

        model = model_factory()

        start = time.time()
        model.fit(X_train_s, y_train_fit)
        train_times.append(time.time() - start)

        start = time.time()
        y_pred = model.predict(X_test_s)
        test_times.append(time.time() - start)

        if normalize_y:
            y_pred = y_pred * y_std + y_mean

        metrics = regression_metrics(y_test, y_pred, n_features=X_train_s.shape[1])
        print(f"  R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}")

        for key in metric_results:
            metric_results[key].append(metrics[key])

    return {
        "model": model_name,
        "mse_mean": summarize(metric_results["mse"])[0],
        "mse_std": summarize(metric_results["mse"])[1],
        "rmse_mean": summarize(metric_results["rmse"])[0],
        "rmse_std": summarize(metric_results["rmse"])[1],
        "mae_mean": summarize(metric_results["mae"])[0],
        "mae_std": summarize(metric_results["mae"])[1],
        "r2_mean": summarize(metric_results["r2"])[0],
        "r2_std": summarize(metric_results["r2"])[1],
        "r2_adjusted_mean": summarize(metric_results["r2_adjusted"])[0],
        "r2_adjusted_std": summarize(metric_results["r2_adjusted"])[1],
        "train_time_mean": summarize(train_times)[0],
        "train_time_std": summarize(train_times)[1],
        "test_time_mean": summarize(test_times)[0],
        "test_time_std": summarize(test_times)[1],
    }


def print_summary(results):
    print("\nRESULTADOS\n")

    col_model = 42
    col_metric = 20
    col_time = 22

    print(
        f"{'Modelo':<{col_model}} "
        f"{'R2':<{col_metric}} "
        f"{'R2 Ajustado':<{col_metric}} "
        f"{'Tempo Treino (s)':<{col_time}} "
        f"{'Tempo Teste (s)':<{col_time}}"
    )
    print("-" * (col_model + col_metric * 2 + col_time * 2 + 4))

    for r in results:
        r2_str = f"{r['r2_mean']:.4f} ± {r['r2_std']:.4f}"
        r2a_str = f"{r['r2_adjusted_mean']:.4f} ± {r['r2_adjusted_std']:.4f}"
        train_str = f"{r['train_time_mean']:.4f} ± {r['train_time_std']:.4f}"
        test_str = f"{r['test_time_mean']:.4f} ± {r['test_time_std']:.4f}"
        print(
            f"{r['model']:<{col_model}} "
            f"{r2_str:<{col_metric}} "
            f"{r2a_str:<{col_metric}} "
            f"{train_str:<{col_time}} "
            f"{test_str:<{col_time}}"
        )


def run_regression():
    print("\n===== REGRESSÃO =====")

    X, y, feature_names = load_dataset(
        DATASET_REGRESSION_ID, target_name=REGRESSION_TARGET_NAME
    )

    # remove colunas que vazam informação do target
    columns_to_remove = [
        REGRESSION_TARGET_NAME,
        "running_total_cases",
        "running_total_cases_prev_day",
    ]
    indices_to_remove = [
        i for i, name in enumerate(feature_names) if name in columns_to_remove
    ]
    X = np.delete(X, indices_to_remove, axis=1)

    results = []

    # KNN
    results.append(
        run_model(
            "KNN (Euclidiana)",
            lambda: KNNRegressor(k=K_VALUES[0], metric="euclidean"),
            X,
            y,
        )
    )
    results.append(
        run_model(
            "KNN (Manhattan)",
            lambda: KNNRegressor(k=K_VALUES[0], metric="manhattan"),
            X,
            y,
        )
    )

    # Regressão Linear
    results.append(run_model("Regressão Linear", lambda: LinearRegression(), X, y))

    # MLP — normalize_y=True: treina em y padronizado, reverte na predição
    for config in MLP_CONFIGS:
        name = (
            f"MLP {config['hidden_layer_sizes']} "
            f"{config['activation']} "
            f"lr={config['lr']} "
            f"epochs={config['epochs']}"
        )
        results.append(
            run_model(
                name,
                lambda c=config: MLPRegressor(
                    hidden_layer_sizes=c["hidden_layer_sizes"],
                    lr=c["lr"],
                    epochs=c["epochs"],
                    activation=c["activation"],
                ),
                X,
                y,
                normalize_y=True,
            )
        )

    # ── Salva CSV de métricas ─────────────────────────────────────────────────
    import os
    import csv

    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "regression_metrics.csv")

    fieldnames = [
        "model",
        "mse_mean",
        "mse_std",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "r2_mean",
        "r2_std",
        "r2_adjusted_mean",
        "r2_adjusted_std",
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
                    "mse_mean": _round_val(r.get("mse_mean", "")),
                    "mse_std": _round_val(r.get("mse_std", "")),
                    "rmse_mean": _round_val(r.get("rmse_mean", "")),
                    "rmse_std": _round_val(r.get("rmse_std", "")),
                    "mae_mean": _round_val(r.get("mae_mean", "")),
                    "mae_std": _round_val(r.get("mae_std", "")),
                    "r2_mean": _round_val(r["r2_mean"]),
                    "r2_std": _round_val(r["r2_std"]),
                    "r2_adjusted_mean": _round_val(r["r2_adjusted_mean"]),
                    "r2_adjusted_std": _round_val(r["r2_adjusted_std"]),
                    "train_time_mean": _round_val(r["train_time_mean"]),
                    "train_time_std": _round_val(r["train_time_std"]),
                    "test_time_mean": _round_val(r["test_time_mean"]),
                    "test_time_std": _round_val(r["test_time_std"]),
                }
            )

    print(f"CSV salvo em: {csv_path}")

    print_summary(results)
