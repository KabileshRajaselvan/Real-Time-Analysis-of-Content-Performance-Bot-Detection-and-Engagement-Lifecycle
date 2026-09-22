"""
Train and evaluate a bot-vs-human classifier on the synthetic account
dataset (see synthetic_accounts.py for the honest-synthetic-data
rationale).

Run:
    python -m src.bot_classifier

Outputs:
    outputs/bot_detection_metrics.json  - measured precision/recall/F1/AUC
    outputs/bot_detection_report.txt    - full classification report
    outputs/feature_importances.csv
    outputs/roc_curve.png
"""
from __future__ import annotations

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split

from src.synthetic_accounts import generate_accounts
from src.features import build_text_features

FEATURE_COLS = [
    "account_age_days",
    "follower_following_ratio",
    "posts_per_day",
    "posting_interval_cv",
    "burstiness",
    "avg_hashtags_per_post",
    "default_profile_ratio",
    "duplicate_content_rate",
]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def build_dataset(n_accounts: int = 2000, seed: int = 42) -> pd.DataFrame:
    df, posts_by_account = generate_accounts(n_accounts=n_accounts, seed=seed)
    text_feats = build_text_features(posts_by_account)
    df["duplicate_content_rate"] = df["account_id"].map(text_feats)
    return df


def train_and_evaluate(df: pd.DataFrame, seed: int = 42) -> dict:
    X = df[FEATURE_COLS]
    y = df["is_bot"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    clf = GradientBoostingClassifier(random_state=seed)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "bot_fraction_test": float(y_test.mean()),
    }

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(OUT_DIR, "bot_detection_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(y_test, y_pred, target_names=["human", "bot"])
    with open(os.path.join(OUT_DIR, "bot_detection_report.txt"), "w") as f:
        f.write(report)

    importances = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(os.path.join(OUT_DIR, "feature_importances.csv"), index=False)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Bot Detection ROC Curve (synthetic held-out test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=120)
    plt.close()

    joblib.dump(clf, os.path.join(OUT_DIR, "bot_classifier.joblib"))

    return metrics


def main():
    df = build_dataset()
    df.to_csv(os.path.join(OUT_DIR if os.path.isdir(OUT_DIR) else ".", "synthetic_accounts.csv"), index=False)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "synthetic_accounts.csv"), index=False)
    metrics = train_and_evaluate(df)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
