"""Train and persist model + simulated dataset for deployment.

Usage:
    python train_model.py --days 365 --seed 42 --lookback 5 --save-model models/rf_model.joblib

This script is intended to be run offline (CI or locally). It writes:
 - data/simulated_data.csv
 - models/rf_model.joblib
 - data/metrics.json
"""
import argparse
import json
import os

import pandas as pd
from dataset_generator import generate_and_save
from model import train_and_predict


def main(days: int, seed: int, lookback: int, save_model: str, tune: bool):
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = generate_and_save(save_path="data/simulated_data.csv", days=days, seed=seed)

    model, X_test, y_test, preds, metrics = train_and_predict(df, lookback=lookback, save_model_path=save_model, tune=tune)

    # save metrics
    with open("data/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:")
    print(" - data/simulated_data.csv")
    print(f" - {save_model}")
    print(" - data/metrics.json")
    print("Metrics:", metrics)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--save-model", type=str, default="models/rf_model.joblib")
    p.add_argument("--tune", action="store_true")
    args = p.parse_args()
    main(args.days, args.seed, args.lookback, args.save_model, args.tune)
