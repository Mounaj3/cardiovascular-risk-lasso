# Train and evaluate a Lasso regression model to predict systolic blood pressure.
#
# Usage:
#   python model.py
#   python model.py --alpha 0.5 --n-folds 5

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm


PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "outputs"
MODEL_DIR = "outputs/models"


def load_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Target mean: {y_train.mean():.1f} | std: {y_train.std():.1f}")
    return X_train, X_test, y_train, y_test


def build_model(alpha):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Lasso(alpha=alpha, max_iter=5000))
    ])


def cross_validate(model, X_train, y_train, n_folds):
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(cv.split(X_train),
                                                      total=n_folds,
                                                      desc="Cross-validation")):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val   = X_train.iloc[val_idx]
        y_fold_val   = y_train.iloc[val_idx]

        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)
        rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
        scores.append(rmse)

    scores = np.array(scores)
    print(f"\n{n_folds}-Fold CV RMSE: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"Folds: {[round(s, 4) for s in scores]}")
    return scores


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n--- Test Set Results ---")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R2   : {r2:.4f}")

    return y_pred, rmse, mae, r2


def plot_predictions(y_test, y_pred, alpha):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.4, color="steelblue", s=15)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linewidth=1)
    ax.set_xlabel("Actual sysBP")
    ax.set_ylabel("Predicted sysBP")
    ax.set_title(f"Actual vs Predicted - Lasso (alpha={alpha})")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "predictions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_residuals(y_test, y_pred, alpha):
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.4, color="steelblue", s=15)
    axes[0].axhline(y=0, color="red", linewidth=1)
    axes[0].set_xlabel("Predicted sysBP")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=40, color="steelblue", edgecolor="white")
    axes[1].axvline(x=0, color="red", linewidth=1)
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residuals Distribution")

    plt.suptitle(f"Residual Analysis - Lasso (alpha={alpha})", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "residuals.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "lasso.pkl")
    joblib.dump(model, path)
    print(f"Model saved: {path}")


def main(alpha, n_folds):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("--- Loading data ---")
    X_train, X_test, y_train, y_test = load_data()

    print(f"\n--- Building Lasso (alpha={alpha}) ---")
    model = build_model(alpha)

    print(f"\n--- Cross Validation ---")
    cross_validate(model, X_train, y_train, n_folds)

    print(f"\n--- Training on full train set ---")
    model.fit(X_train, y_train)

    print(f"\n--- Evaluation ---")
    y_pred, rmse, mae, r2 = evaluate(model, X_test, y_test)

    print(f"\n--- Saving outputs ---")
    plot_predictions(y_test, y_pred, alpha)
    plot_residuals(y_test, y_pred, alpha)
    save_model(model)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Lasso regularization strength (default: 1.0)")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    args = parser.parse_args()
    main(args.alpha, args.n_folds)