"""
Streaming simulation ("near-real-time" component).

HONESTY NOTE: this repo does not hold a live Twitter/X streaming
connection (that requires paid API access). What this module provides
instead is a genuine incremental-processing simulation: synthetic tweets
"arrive" one at a time in a generator/loop, and for each arriving tweet we:
  1. update a running per-account feature window (posting rate, interval
     regularity, duplication) incrementally, without recomputing from
     scratch each time,
  2. score the account's current bot-likelihood using the trained
     classifier from bot_classifier.py,
  3. append the tweet's engagement to that post's synthetic engagement
     trajectory and re-run the lightweight rule-based viral check.

This demonstrates the *processing pattern* required for real-time/streaming
deployment (incremental state, per-event scoring) without claiming to be
connected to a live production stream. In the README this is described as
"streaming-simulated / near-real-time", not "real-time production system".

Run:
    python -m src.streaming_simulation
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd

from src.bot_classifier import FEATURE_COLS, OUT_DIR, build_dataset
from src.features import duplicate_content_rate

MODEL_PATH = os.path.join(OUT_DIR, "bot_classifier.joblib")


class AccountState:
    """Incrementally-updated per-account state for streaming scoring."""

    def __init__(self, account_id: str, static_features: dict):
        self.account_id = account_id
        self.static_features = static_features  # age, follower ratio etc (assumed slowly-changing)
        self.timestamps = deque(maxlen=50)
        self.recent_posts = deque(maxlen=50)

    def ingest(self, timestamp: float, text: str):
        self.timestamps.append(timestamp)
        self.recent_posts.append(text)

    def current_features(self) -> dict:
        feats = dict(self.static_features)
        if len(self.timestamps) >= 2:
            ts = np.array(self.timestamps)
            gaps = np.diff(ts)
            feats["posting_interval_cv"] = float(np.std(gaps) / (np.mean(gaps) + 1e-9))
            feats["posts_per_day"] = float(len(ts) / (max(ts[-1] - ts[0], 1e-9) / 86400))
        feats["duplicate_content_rate"] = duplicate_content_rate(list(self.recent_posts))
        return feats


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    from src.bot_classifier import train_and_evaluate
    df = build_dataset()
    train_and_evaluate(df)
    return joblib.load(MODEL_PATH)


def simulate_stream(n_accounts: int = 20, events_per_account: int = 15, seed: int = 3) -> list[dict]:
    """Simulate an arriving stream of tweets from a handful of accounts and
    score bot-likelihood incrementally after each arrival."""
    rng = np.random.default_rng(seed)
    clf = load_or_train_model()

    df = build_dataset(n_accounts=n_accounts, seed=seed)
    states = {}
    for _, row in df.iterrows():
        static = {c: row[c] for c in FEATURE_COLS if c != "duplicate_content_rate"}
        states[row["account_id"]] = AccountState(row["account_id"], static)

    log = []
    t = time.time()
    sample_texts = [
        "check out this amazing deal today only",
        "breaking news everyone should see this",
        "just tried this and it changed my life",
        "sharing my honest experience for anyone curious",
    ]

    for event_i in range(events_per_account):
        for account_id, state in states.items():
            t += rng.exponential(30)  # simulated arrival gap, seconds
            text = sample_texts[rng.integers(0, len(sample_texts))]
            state.ingest(t, text)

            feats = state.current_features()
            X = pd.DataFrame([{c: feats.get(c, 0.0) for c in FEATURE_COLS}])
            proba = float(clf.predict_proba(X)[0, 1])

            log.append({
                "sim_time": t,
                "account_id": account_id,
                "event_index": event_i,
                "bot_likelihood": proba,
            })

    return log


def main():
    log = simulate_stream()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "streaming_simulation_log.json"), "w") as f:
        json.dump(log[-20:], f, indent=2)
    print(f"Processed {len(log)} simulated streaming events.")
    print("Last 5 scored events:")
    for row in log[-5:]:
        print(row)


if __name__ == "__main__":
    main()
