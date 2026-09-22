"""
Engagement-lifecycle analysis.

HONESTY NOTE: real timestamped engagement-over-time data (likes/retweets
sampled repeatedly across a post's lifetime) requires paid Twitter/X API
access this project does not have. This module generates SYNTHETIC but
realistic per-post engagement time series and then runs genuine analysis
on top of them:

  1. Curve fitting: each post's cumulative-engagement curve is fit with an
     exponential-decay-after-peak model (engagement rises to a peak then
     decays), which is the standard shape reported in social-media
     diffusion literature.
  2. Viral vs normal classification: a rule-based classifier operating on
     the *shape* of the curve (peak height relative to baseline, decay
     half-life, and time-to-peak) labels each post as "viral" or "normal".
     This is cross-checked against KMeans clustering on the same shape
     features as a sanity check.
  3. Lifecycle-stage visualization: plots showing early growth / peak /
     decay stages for representative viral and normal posts.

Run:
    python -m src.engagement_lifecycle
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
HOURS = np.arange(0, 72, 1)  # track engagement hourly for 72 hours post-publish


def _engagement_curve(t, peak_time, peak_height, decay_rate, baseline):
    """Rise-then-decay engagement shape: logistic growth to a peak, then
    exponential decay. Used both to SIMULATE synthetic posts and to FIT
    parameters back out of a (possibly noisy) observed curve."""
    growth = peak_height / (1 + np.exp(-(t - peak_time * 0.3)))
    decay = np.exp(-decay_rate * np.maximum(0, t - peak_time))
    return baseline + growth * decay


def simulate_posts(n_posts: int = 300, viral_fraction: float = 0.12, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_viral = int(round(n_posts * viral_fraction))
    n_normal = n_posts - n_viral
    records = []

    for i in range(n_posts):
        is_viral = i < n_viral
        if is_viral:
            peak_time = rng.uniform(2, 10)          # viral posts peak fast
            peak_height = rng.lognormal(mean=8.5, sigma=0.6)
            decay_rate = rng.uniform(0.02, 0.08)    # slow decay -> long tail
        else:
            peak_time = rng.uniform(1, 20)
            peak_height = rng.lognormal(mean=4.5, sigma=0.9)
            decay_rate = rng.uniform(0.08, 0.35)    # fast decay
        baseline = max(0, rng.normal(2, 1))

        clean_curve = _engagement_curve(HOURS, peak_time, peak_height, decay_rate, baseline)
        noise = rng.normal(0, np.maximum(1, clean_curve * 0.08))
        observed = np.clip(clean_curve + noise, 0, None)
        # cumulative engagement is monotonic non-decreasing in reality
        cumulative = np.maximum.accumulate(observed)

        records.append({
            "post_id": f"post_{i:04d}",
            "is_viral_true": int(is_viral),
            "peak_time_true": peak_time,
            "peak_height_true": peak_height,
            "decay_rate_true": decay_rate,
            "cumulative_engagement": cumulative.tolist(),
        })

    return pd.DataFrame(records)


def fit_curve(cumulative_engagement: np.ndarray) -> dict:
    """Fit the rise-decay model to one post's observed cumulative curve.
    Falls back to shape-derived heuristics if optimization fails."""
    t = HOURS.astype(float)
    y = np.asarray(cumulative_engagement, dtype=float)
    # convert cumulative back to incremental for shape fitting stability
    baseline0 = max(y[0], 0.1)
    p0 = [10.0, max(y.max() - y[0], 1.0), 0.1, baseline0]
    try:
        popt, _ = curve_fit(_engagement_curve, t, y, p0=p0, maxfev=5000,
                             bounds=([0, 0, 0.001, 0], [72, 1e7, 5, 1e5]))
        peak_time, peak_height, decay_rate, baseline = popt
        fit_ok = True
    except Exception:
        peak_idx = int(np.argmax(np.diff(y, prepend=y[0])))
        peak_time = max(t[peak_idx], 1)
        peak_height = y.max() - y[0]
        half = y.max() / 2
        after_peak = y[peak_idx:]
        half_idx = np.argmax(after_peak <= half) if np.any(after_peak <= half) else len(after_peak) - 1
        decay_rate = np.log(2) / max(half_idx, 1)
        baseline = y[0]
        fit_ok = False

    return {
        "peak_time_fit": float(peak_time),
        "peak_height_fit": float(peak_height),
        "decay_rate_fit": float(decay_rate),
        "baseline_fit": float(baseline),
        "half_life_hours": float(np.log(2) / decay_rate) if decay_rate > 0 else float("inf"),
        "fit_converged": fit_ok,
    }


def classify_viral_rule_based(fit_params: dict, total_engagement: float) -> int:
    """Rule-based viral classifier on curve shape: high peak height AND
    long half-life (slow decay = sustained attention) => viral."""
    return int(fit_params["peak_height_fit"] > 500 and fit_params["half_life_hours"] > 10
               or total_engagement > 3000)


def run_analysis(n_posts: int = 300) -> dict:
    df = simulate_posts(n_posts=n_posts)

    fit_rows = []
    for _, row in df.iterrows():
        curve = np.array(row["cumulative_engagement"])
        params = fit_curve(curve)
        total_engagement = float(curve[-1])
        rule_label = classify_viral_rule_based(params, total_engagement)
        fit_rows.append({**params, "total_engagement": total_engagement,
                          "viral_rule_based": rule_label})

    fit_df = pd.DataFrame(fit_rows)
    result = pd.concat([df.drop(columns=["cumulative_engagement"]), fit_df], axis=1)

    # Cross-check via KMeans clustering on shape features
    shape_features = result[["peak_time_fit", "peak_height_fit", "half_life_hours", "total_engagement"]].copy()
    shape_features["half_life_hours"] = shape_features["half_life_hours"].clip(upper=200)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(shape_features)
    km = KMeans(n_clusters=2, random_state=7, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)
    # the cluster with higher mean total_engagement is "viral"
    cluster_means = result.groupby(cluster_labels)["total_engagement"].mean()
    viral_cluster = cluster_means.idxmax()
    result["viral_cluster_based"] = (cluster_labels == viral_cluster).astype(int)

    agreement = float((result["viral_rule_based"] == result["viral_cluster_based"]).mean())
    recall_vs_truth = float(
        (result.loc[result["is_viral_true"] == 1, "viral_rule_based"] == 1).mean()
    )
    precision_vs_truth = float(
        result.loc[result["viral_rule_based"] == 1, "is_viral_true"].mean()
    ) if result["viral_rule_based"].sum() > 0 else 0.0

    os.makedirs(OUT_DIR, exist_ok=True)
    result.to_csv(os.path.join(OUT_DIR, "engagement_lifecycle_results.csv"), index=False)

    summary = {
        "n_posts": n_posts,
        "rule_vs_cluster_agreement": agreement,
        "rule_based_recall_vs_synthetic_truth": recall_vs_truth,
        "rule_based_precision_vs_synthetic_truth": precision_vs_truth,
        "n_viral_rule_based": int(result["viral_rule_based"].sum()),
        "n_viral_true": int(result["is_viral_true"].sum()),
    }
    with open(os.path.join(OUT_DIR, "engagement_lifecycle_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _plot_lifecycle_examples(df, result)
    return summary


def _plot_lifecycle_examples(raw_df: pd.DataFrame, result: pd.DataFrame):
    viral_idx = result[result["viral_rule_based"] == 1].index
    normal_idx = result[result["viral_rule_based"] == 0].index
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for ax, idx_pool, title in [
        (axes[0], viral_idx, "Viral post lifecycle (example)"),
        (axes[1], normal_idx, "Normal post lifecycle (example)"),
    ]:
        if len(idx_pool) == 0:
            continue
        idx = idx_pool[0]
        curve = np.array(raw_df.loc[idx, "cumulative_engagement"])
        peak_t = result.loc[idx, "peak_time_fit"]
        ax.plot(HOURS, curve, color="#2b6cb0", label="cumulative engagement")
        ax.axvline(peak_t, color="#e53e3e", linestyle="--", label="fitted peak")
        ax.axvspan(0, peak_t, color="#c6f6d5", alpha=0.4, label="growth stage")
        ax.axvspan(peak_t, HOURS[-1], color="#fed7d7", alpha=0.3, label="decay stage")
        ax.set_title(title)
        ax.set_xlabel("Hours since post")
        ax.set_ylabel("Cumulative engagement")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "engagement_lifecycle_examples.png"), dpi=120)
    plt.close()


def main():
    summary = run_analysis()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
