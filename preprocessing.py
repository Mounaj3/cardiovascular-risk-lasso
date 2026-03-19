# Preprocess the Framingham dataset for systolic blood pressure prediction.
#
# Usage:
#   python preprocessing.py
#   python preprocessing.py --test-size 0.25

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


DATA_PATH = "framingham.csv"
PROCESSED_DIR = "data/processed"

TARGET = "sysBP"
DROP_COLS = ["TenYearCHD", "diaBP"]


def load_data():
    data = pd.read_csv(DATA_PATH)
    print(f"Loaded {data.shape[0]} rows, {data.shape[1]} columns")
    return data


def handle_missing_values(data):
    print(f"Missing values:\n{data.isnull().sum()[data.isnull().sum() > 0]}")
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())
    print(f"Missing values after imputation: {data.isnull().sum().sum()}")
    return data


def engineer_features(data):
    data["overweight"]   = (data["BMI"] >= 25).astype(int)
    data["heavy_smoker"] = (data["cigsPerDay"] >= 20).astype(int)
    return data


def split_and_save(data, test_size):
    data = data.drop(columns=DROP_COLS)

    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    print(f"\nTarget sysBP — mean: {y.mean():.1f} | std: {y.std():.1f} | min: {y.min()} | max: {y.max()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"Saved to {PROCESSED_DIR}/")


def main(test_size):
    data = load_data()
    data = handle_missing_values(data)
    data = engineer_features(data)
    split_and_save(data, test_size)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set proportion (default: 0.2)")
    args = parser.parse_args()
    main(args.test_size)