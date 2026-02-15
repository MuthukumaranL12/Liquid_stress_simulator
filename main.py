import os
import argparse
import json
from dataset_generator import generate_and_save
import pandas as pd
from model import train_and_predict, load_model
from visualizer import plot_actual_vs_pred
from sklearn.metrics import precision_score, recall_score, f1_score


def run_all(data_path: str = "data/simulated_data.csv",
            days: int = 365,
            lookback: int = 5,
            seed: int | None = None,
            threshold: float = 0.1,
            save_model_path: str = "models/rf_model.joblib",
            tune: bool = False):
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = generate_and_save(save_path=data_path, days=days, seed=seed)

    model, X_test, y_test, preds, metrics = train_and_predict(df, lookback=lookback, save_model_path=save_model_path, tune=tune)

    # Build index/labels for test range
    test_start = lookback
    indices = list(range(test_start, test_start + len(preds)))

    # Print regression metrics
    print("Regression metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # Detection (binary) metrics for threshold-based crisis alerts
    y_true_bin = (y_test < threshold).astype(int)
    y_pred_bin = (preds < threshold).astype(int)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    metrics.update({
        "detection_precision": float(precision),
        "detection_recall": float(recall),
        "detection_f1": float(f1)
    })

    # Alerts based on predictions
    alerts = [(idx, float(p)) for idx, p in zip(indices, preds) if p < threshold]

    if alerts:
        print(f"Liquidity Crisis Warnings (predicted_liquidity < {threshold}):")
        for a in alerts[:20]:
            print(f" Day index: {a[0]}  predicted_liquidity: {a[1]:.4f}")
        if len(alerts) > 20:
            print(f" ...and {len(alerts)-20} more alerts")
    else:
        print(f"No imminent liquidity crisis predicted (threshold {threshold})")

    # Save predictions and metrics
    out_df = pd.DataFrame({"index": indices, "predicted_liquidity": preds, "actual_liquidity": y_test})
    out_df.to_csv("data/predictions.csv", index=False)

    with open("data/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Visualize
    plot_actual_vs_pred(indices, y_test, preds, threshold=threshold, save_path="data/liquidity_plot.png")

    print("Saved: data/simulated_data.csv, data/predictions.csv, data/liquidity_plot.png, data/metrics.json, models/*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run liquidity stress simulation + model")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--model-path", type=str, default="models/rf_model.joblib")
    parser.add_argument("--tune", action="store_true", help="Run a small grid-search for hyperparameters")
    args = parser.parse_args()

    run_all(days=args.days, seed=args.seed, lookback=args.lookback, threshold=args.threshold, save_model_path=args.model_path, tune=args.tune)
