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

    use_kdtree = False
    if "use-kdtree" in sys.argv[2:]:
        use_kdtree = True

    if arg in ["r", "regression"]:
        run_regression(use_kdtree=use_kdtree)
    elif arg in ["c", "classification"]:
        run_classification(use_kdtree=use_kdtree)
    else:
        print("Argumento inválido")


if __name__ == "__main__":
    main()