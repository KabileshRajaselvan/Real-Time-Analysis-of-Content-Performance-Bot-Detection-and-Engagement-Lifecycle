"""
Synthetic account + post generator for bot-detection modeling.

HONESTY NOTE
------------
The Twitter/X API now requires a paid tier for meaningful search/streaming
access, so this repository does not ship real labeled bot/human account
data. Instead this module generates a SYNTHETIC dataset with documented,
programmatic ground truth (`is_bot`), following the same "honest synthetic
data" pattern used elsewhere in this author's portfolio (e.g. the fraud /
churn projects).

The generator intentionally overlaps the two populations so the problem is
NOT trivially separable:
  * Some bots post at low frequency and low regularity ("sleeper" bots).
  * Some humans are extremely active / repetitive (power users, marketers).
  * Noise is added to every feature.

Features generated per account (account-level, aggregated over a synthetic
posting history):
  - account_age_days
  - follower_following_ratio
  - posts_per_day
  - posting_interval_cv        (coefficient of variation of inter-post
                                 gaps -> LOW cv = suspiciously regular timing)
  - burstiness                 (Fano-factor style burst score)
  - duplicate_content_rate     (fraction of an account's own posts that are
                                 near-duplicates of each other, via TF-IDF
                                 cosine similarity - computed downstream in
                                 features.py from the raw post texts this
                                 module also generates)
  - avg_hashtags_per_post
  - default_profile_ratio      (proxy: 1 if using default-looking profile)

Ground truth label: is_bot (1 = bot-like synthetic account, 0 = human-like)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 42

# A small pool of template sentences used to synthesize post text so that
# downstream TF-IDF / duplicate-content features have something real to
# operate on. Bots draw from a much smaller template pool with more copying.
_TEMPLATES = [
    "check out this amazing deal on {p} today only",
    "breaking news about {p} everyone should see this",
    "just tried {p} and honestly it changed my life",
    "why is nobody talking about {p} right now",
    "{p} is trending and here is why it matters",
    "my honest thoughts on {p} after using it for a week",
    "cant believe what happened with {p} today",
    "sharing my experience with {p} for anyone curious",
    "this thread about {p} is a must read",
    "here is a quick guide to {p} for beginners",
]

_TOPICS = [
    "crypto", "the market", "this new phone", "climate policy", "the election",
    "sports", "AI tools", "the game last night", "this recipe", "streaming",
]


def _make_post_texts(n_posts: int, duplication_rate: float, rng: np.random.Generator) -> list[str]:
    """Generate n_posts of text for one account with a target duplication rate."""
    if n_posts == 0:
        return []
    texts = []
    # Pool size controls duplication: small pool + reuse => high duplication.
    pool_size = max(1, int(round(n_posts * (1 - duplication_rate))))
    pool = []
    for _ in range(pool_size):
        tmpl = _TEMPLATES[rng.integers(0, len(_TEMPLATES))]
        topic = _TOPICS[rng.integers(0, len(_TOPICS))]
        pool.append(tmpl.format(p=topic))
    for _ in range(n_posts):
        base = pool[rng.integers(0, len(pool))]
        # humans/mixed accounts add small perturbation even to "reused" posts
        if rng.random() < 0.3:
            extra = _TOPICS[rng.integers(0, len(_TOPICS))]
            base = f"{base} also thinking about {extra}"
        texts.append(base)
    return texts


def generate_accounts(n_accounts: int = 2000, bot_fraction: float = 0.35,
                       seed: int = RNG_SEED) -> tuple[pd.DataFrame, dict]:
    """
    Generate a synthetic labeled dataset of accounts with engineered
    behavioral features and raw post text history.

    Returns
    -------
    df : DataFrame, one row per account, with features + is_bot label.
    posts_by_account : dict[account_id] -> list[str] raw post texts
    """
    rng = np.random.default_rng(seed)
    n_bots = int(round(n_accounts * bot_fraction))
    n_humans = n_accounts - n_bots
    labels = np.array([1] * n_bots + [0] * n_humans)
    rng.shuffle(labels)

    rows = []
    posts_by_account = {}

    for i in range(n_accounts):
        is_bot = labels[i]
        acct_id = f"acct_{i:05d}"

        if is_bot:
            # Bots: mostly young accounts, skewed follower ratio, high
            # posting frequency, regular timing, high duplication -- but
            # WITH substantial noise/overlap so it is not trivial.
            account_age_days = max(1, rng.normal(280, 260))
            follower_following_ratio = np.clip(rng.lognormal(mean=-0.2, sigma=1.3), 0.001, 60)
            posts_per_day = max(0.05, rng.normal(10, 9))
            # low CV => very regular intervals (bot-like), but noisy
            posting_interval_cv = np.clip(rng.normal(0.55, 0.40), 0.02, 3.0)
            burstiness = np.clip(rng.normal(0.4, 0.45), -1, 1)
            duplication_rate = np.clip(rng.normal(0.42, 0.30), 0.0, 0.95)
            avg_hashtags_per_post = max(0, rng.normal(2.4, 1.9))
            default_profile_ratio = rng.random() < 0.45
        else:
            # Humans: older accounts on average, more organic ratios/timing,
            # lower duplication -- but with real overlap into "bot-like"
            # ranges (e.g. active marketers, fan accounts).
            account_age_days = max(1, rng.normal(550, 430))
            follower_following_ratio = np.clip(rng.lognormal(mean=0.1, sigma=1.3), 0.001, 80)
            posts_per_day = max(0.01, rng.normal(4.5, 5.5))
            posting_interval_cv = np.clip(rng.normal(0.75, 0.45), 0.02, 3.0)
            burstiness = np.clip(rng.normal(0.2, 0.4), -1, 1)
            duplication_rate = np.clip(rng.normal(0.20, 0.20), 0.0, 0.9)
            avg_hashtags_per_post = max(0, rng.normal(1.4, 1.5))
            default_profile_ratio = rng.random() < 0.22

        n_posts = int(np.clip(rng.poisson(max(1, posts_per_day * 5)), 3, 300))
        texts = _make_post_texts(n_posts, duplication_rate, rng)
        posts_by_account[acct_id] = texts

        rows.append({
            "account_id": acct_id,
            "account_age_days": account_age_days,
            "follower_following_ratio": follower_following_ratio,
            "posts_per_day": posts_per_day,
            "posting_interval_cv": posting_interval_cv,
            "burstiness": burstiness,
            "avg_hashtags_per_post": avg_hashtags_per_post,
            "default_profile_ratio": float(default_profile_ratio),
            "n_posts": n_posts,
            "is_bot": int(is_bot),
        })

    df = pd.DataFrame(rows)
    meta = {"n_accounts": n_accounts, "bot_fraction": bot_fraction, "seed": seed}
    return df, posts_by_account


if __name__ == "__main__":
    df, posts = generate_accounts()
    print(df.head())
    print(df["is_bot"].value_counts())
