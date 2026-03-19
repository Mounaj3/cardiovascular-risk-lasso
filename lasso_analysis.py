# Analyze Lasso coefficients, regularization path and optimal alpha.
#
# Usage:
#   python lasso_analysis.py
#   python lasso_analysis.py --alpha 0.5

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "outputs"


def load_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()
    return X_train, X_test, y_train, y_test


def fit_lasso(X_train, y_train, alpha):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Lasso(alpha=alpha, max_iter=5000))
    ])
    model.fit(X_train, y_train)
    return model


def print_and_plot_coefficients(coefs, alpha):
    kept   = coefs[coefs != 0].sort_values()
    zeroed = coefs[coefs == 0]

    print(f"\n--- Lasso Coefficients (alpha={alpha}) ---")
    print(f"Features kept   : {len(kept)}")
    print(f"Features zeroed : {len(zeroed)}")

    print(f"\nKept features:")
    for feat, val in kept.sort_values(ascending=False).items():
        print(f"  {feat:<20} {val:+.4f}")

    print(f"\nZeroed out:")
    for feat in zeroed.index:
        print(f"  {feat}")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["steelblue" if c > 0 else "tomato" for c in kept]
    kept.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Lasso Coefficients (alpha={alpha}) — "
                 f"{len(kept)} kept, {len(zeroed)} zeroed out")
    ax.set_xlabel("Coefficient value (standardized features)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"lasso_coefficients.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")


def plot_regularization_path(X_train, y_train, feature_names):
    alphas = np.logspace(-3, 2, 100)
    coefs_path = []

    for alpha in alphas:
        model = fit_lasso(X_train, y_train, alpha)
        coefs_path.append(model.named_steps["regressor"].coef_)

    coefs_path = np.array(coefs_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, feat in enumerate(feature_names):
        ax.plot(alphas, coefs_path[:, i], label=feat)

    ax.set_xscale("log")
    ax.axhline(y=0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Alpha (regularization strength)")
    ax.set_ylabel("Coefficient value")
    ax.set_title("Lasso Regularization Path — features zeroed out as alpha increases")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lasso_regularization_path.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_alpha_vs_r2(X_train, y_train, X_test, y_test):
    alphas = np.logspace(-3, 2, 50)
    r2_train_scores = []
    r2_test_scores  = []

    for alpha in alphas:
        model = fit_lasso(X_train, y_train, alpha)
        r2_train_scores.append(r2_score(y_train, model.predict(X_train)))
        r2_test_scores.append(r2_score(y_test, model.predict(X_test)))

    best_alpha = alphas[np.argmax(r2_test_scores)]
    best_r2    = max(r2_test_scores)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, r2_train_scores, label="Train R²", color="steelblue")
    ax.plot(alphas, r2_test_scores,  label="Test R²",  color="tomato")
    ax.axvline(x=best_alpha, color="green", linewidth=1,
               linestyle="--", label=f"Best alpha={best_alpha:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("Alpha (regularization strength)")
    ax.set_ylabel("R²")
    ax.set_title("R² vs Alpha — finding the optimal regularization")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lasso_alpha_vs_r2.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    print(f"Best alpha: {best_alpha:.4f} (Test R²={best_r2:.4f})")

    return best_alpha


def main(alpha):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    feature_names = X_train.columns.tolist()

    # 1. Coefficients for given alpha
    model = fit_lasso(X_train, y_train, alpha)
    coefs = pd.Series(model.named_steps["regressor"].coef_, index=feature_names)
    print_and_plot_coefficients(coefs, alpha)

    # 2. Regularization path
    print("\nComputing regularization path...")
    plot_regularization_path(X_train, y_train, feature_names)

    # 3. R² vs alpha
    print("\nComputing R² vs alpha...")
    best_alpha = plot_alpha_vs_r2(X_train, y_train, X_test, y_test)

    # 4. Refit with best alpha
    print(f"\nRefitting with best alpha={best_alpha:.4f}...")
    best_model = fit_lasso(X_train, y_train, best_alpha)
    best_coefs = pd.Series(
        best_model.named_steps["regressor"].coef_,
        index=feature_names
    )
    print_and_plot_coefficients(best_coefs, round(best_alpha, 4))

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Lasso regularization strength (default: 1.0)")
    args = parser.parse_args()
    main(args.alpha)
