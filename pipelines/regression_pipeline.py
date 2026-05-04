import time
import numpy as np

from config import (
    DATASET_REGRESSION_ID,
    REGRESSION_TARGET_NAME,
    N_FOLDS,
    RANDOM_SEED,
    K_VALUES
)

from data.dataset_loader import load_dataset
from utils.preprocessing import Preprocessor
from utils.validation import k_fold_split
from utils.metrics import regression_metrics

from models.knn_regressor import KNNRegressor
from models.linear_regression import LinearRegression


# media e desvio
def summarize(values):
    values = np.array(values, dtype=float)
    return np.mean(values), np.std(values)


# executa um modelo em todos os folds e retorna resumo
# model_factory: função que cria instancia do modelo
def run_model(model_name, model_factory, X, y):
    print(f"\nExecutando: {model_name}")

    # obtem folds com k-fold
    folds = k_fold_split(X, y, n_folds=N_FOLDS, random_state=RANDOM_SEED)

    # resultados por metrica
    metric_results = {
        "mse": [],
        "rmse": [],
        "mae": [],
        "r2": [],
        "r2_adjusted": [],
    }

    train_times = []
    test_times = []

    # para cada fold: preprocessa, treina, prediz e calcula metricas
    for i, (X_train, X_test, y_train, y_test) in enumerate(folds):
        print(f"\n[{model_name}] Fold {i+1}/{len(folds)}")

        preprocessor = Preprocessor()
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        model = model_factory()

        start_train = time.time()
        model.fit(X_train, y_train)
        end_train = time.time()

        start_test = time.time()
        y_pred = model.predict(X_test)
        end_test = time.time()

        metrics = regression_metrics(
            y_test,
            y_pred,
            n_features=X_train.shape[1]
        )

        # armazena metricas
        for key in metric_results:
            metric_results[key].append(metrics[key])

        train_times.append(end_train - start_train)
        test_times.append(end_test - start_test)

    # cria dict resumo com medias e desvios das metricas/tempos
    summary = {
        "model": model_name,
        "r2_mean": summarize(metric_results["r2"])[0],
        "r2_std": summarize(metric_results["r2"])[1],
        "r2_adjusted_mean": summarize(metric_results["r2_adjusted"])[0],
        "r2_adjusted_std": summarize(metric_results["r2_adjusted"])[1],
        "train_time_mean": summarize(train_times)[0],
        "train_time_std": summarize(train_times)[1],
        "test_time_mean": summarize(test_times)[0],
        "test_time_std": summarize(test_times)[1],
    }

    return summary


# imprime tabela resultados
def print_summary(results):
    print("\nRESULTADOS\n")

    col_model = 25
    col_metric = 20
    col_time = 22

    print(
        f"{'Modelo':<{col_model}} "
        f"{'R2':<{col_metric}} "
        f"{'R2 Ajustado':<{col_metric}} "
        f"{'Tempo Treino (s)':<{col_time}} "
        f"{'Tempo Teste (s)':<{col_time}}"
    )

    print("-" * (col_model + col_metric*2 + col_time*2 + 4))

    for r in results:
        print(
            f"{r['model']:<{col_model}} "
            f"{f'{r['r2_mean']:.4f} ± {r['r2_std']:.4f}':<{col_metric}} "
            f"{f'{r['r2_adjusted_mean']:.4f} ± {r['r2_adjusted_std']:.4f}':<{col_metric}} "
            f"{f'{r['train_time_mean']:.4f} ± {r['train_time_std']:.4f}':<{col_time}} "
            f"{f'{r['test_time_mean']:.4f} ± {r['test_time_std']:.4f}':<{col_time}}"
        )


# prepara dados e executa modelos de regressao
# retorna resumo dos resultados
def run_regression(use_kdtree=False):
    print("\n===== REGRESSÃO =====")

    # carrega dataset
    X, y, feature_names = load_dataset(
        DATASET_REGRESSION_ID,
        target_name=REGRESSION_TARGET_NAME
    )

    # colunas a remover quando presentes
    columns_to_remove = [
        REGRESSION_TARGET_NAME,
        "running_total_cases",
        "running_total_cases_prev_day"
    ]

    # identifica indices das colunas a remover
    indices_to_remove = [
        i for i, name in enumerate(feature_names)
        if name in columns_to_remove
    ]

    # remove colunas indesejadas
    X = np.delete(X, indices_to_remove, axis=1)

    results = []

    # executa KNN euclidiana
    results.append(
        run_model(
            "KNN (Euclidiana)",
            lambda: KNNRegressor(k=K_VALUES[0], metric="euclidean", use_kdtree=use_kdtree),
            X,
            y
        )
    )

    # executa KNN manhattan
    results.append(
        run_model(
            "KNN (Manhattan)",
            lambda: KNNRegressor(k=K_VALUES[0], metric="manhattan", use_kdtree=use_kdtree),
            X,
            y
        )
    )

    # executa regressao multipla
    results.append(
        run_model(
            "Regressão Linear",
            lambda: LinearRegression(),
            X,
            y
        )
    )

    # imprime resumo final
    print_summary(results)