import matplotlib.pyplot as plt


def plot_actual_vs_pred(dates, actual, predicted, threshold=0.1, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual, label="Actual Liquidity")
    plt.plot(dates, predicted, label="Predicted Liquidity", linestyle="--")
    plt.axhline(threshold, color="red", linestyle=":", label=f"Threshold={threshold}")
    plt.xlabel("Index")
    plt.ylabel("Liquidity Ratio")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    import numpy as np
    plot_actual_vs_pred(range(10), np.random.rand(10), np.random.rand(10), save_path=None)
