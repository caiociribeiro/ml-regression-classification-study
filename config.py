DATASET_REGRESSION_ID = 43371
DATASET_CLASSIFICATION_ID = 46255

REGRESSION_TARGET_NAME = "daily_new_cases"
CLASSIFICATION_TARGET_NAME = "GradeClass"

N_FOLDS = 5
RANDOM_SEED = 42

K_VALUES = [5]
DISTANCE_METRICS = ["euclidean", "manhattan"]

ADALINE_CONFIGS = [
    {"lr": 0.001, "epochs": 100},   # era epochs=50  — muito pouco para OvR com 5 classes
    {"lr": 0.01,  "epochs": 200},   # era epochs=100
    {"lr": 0.1,   "epochs": 300},   # era epochs=200
]

MLP_CONFIGS = [
    {
        "hidden_layer_sizes": (32,),
        "lr": 0.01,
        "epochs": 100,
        "activation": "sigmoid",
    },
    {
        "hidden_layer_sizes": (64,),
        "lr": 0.005,
        "epochs": 200,
        "activation": "tanh",
    },
    {
        "hidden_layer_sizes": (64, 32),
        "lr": 0.001,    # era 0.0001 — inviável para convergir em 300 épocas com clipping
        "epochs": 300,
        "activation": "relu",
    },
]
