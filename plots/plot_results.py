import os
import matplotlib.pyplot as plt


def plot_r2(results, save_path):
    models = [r["model"] for r in results]
    r2 = [r["r2_mean"] for r in results]
    r2_std = [r["r2_std"] for r in results]

    plt.figure()
    plt.bar(models, r2, yerr=r2_std)
    plt.title("Comparação de R2 entre modelos")
    plt.xlabel("Modelo")
    plt.ylabel("R2")
    plt.xticks(rotation=20)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "r2_comparison.png"))
    plt.close()


def plot_times(results, save_path):
    models = [r["model"] for r in results]
    train_times = [r["train_time_mean"] for r in results]
    test_times = [r["test_time_mean"] for r in results]

    x = range(len(models))

    plt.figure()
    plt.bar(x, train_times, label="Treino")
    plt.bar(x, test_times, bottom=train_times, label="Teste")

    plt.xticks(x, models, rotation=20)
    plt.title("Tempo de execução")
    plt.xlabel("Modelo")
    plt.ylabel("Tempo (s)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "time_comparison.png"))
    plt.close()


def generate_plots(results, output_dir="output/figures"):
    os.makedirs(output_dir, exist_ok=True)

    plot_r2(results, output_dir)
    plot_times(results, output_dir)

    print("Plots salvos em:", output_dir)