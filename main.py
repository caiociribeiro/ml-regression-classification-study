import sys

from pipelines.regression_pipeline import run_regression
from pipelines.classification_pipeline import run_classification


def main():
    if len(sys.argv) < 2:
        print("Use:")
        print("  python main.py r")
        print("  python main.py c")
        return

    arg = sys.argv[1]

    if arg in ["r", "regression"]:
        run_regression()
    elif arg in ["c", "classification"]:
        run_classification()
    else:
        print("Argumento invalido")


if __name__ == "__main__":
    main()
