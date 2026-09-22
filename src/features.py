"""
Text-based feature extraction for bot detection.

Computes a per-account "duplicate content rate" from raw post text using
TF-IDF cosine similarity across an account's own posts. This is meant to
consume text that has already been through the cleaning pipeline built in
`twitter_final_Preprocess.ipynb` (lowercased, stripped of mentions/
hashtags/URLs/punctuation/emojis, stopwords removed). For the synthetic
demo we apply a light version of the same cleaning inline so the module
is runnable standalone.

We deliberately compute this via TF-IDF similarity rather than exact
string match, since bots that append small perturbations (see
synthetic_accounts.py) should still count as high-duplication.
"""
from __future__ import annotations

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def duplicate_content_rate(posts: list[str], similarity_threshold: float = 0.8) -> float:
    """
    Fraction of post pairs (within one account's history) whose TF-IDF
    cosine similarity exceeds `similarity_threshold`. Returns 0.0 for
    accounts with fewer than 2 posts.
    """
    cleaned = [clean_text(p) for p in posts if p and p.strip()]
    if len(cleaned) < 2:
        return 0.0

    vectorizer = TfidfVectorizer(min_df=1)
    try:
        tfidf = vectorizer.fit_transform(cleaned)
    except ValueError:
        return 0.0

    sims = cosine_similarity(tfidf)
    n = sims.shape[0]
    iu = np.triu_indices(n, k=1)
    pair_sims = sims[iu]
    if pair_sims.size == 0:
        return 0.0
    return float(np.mean(pair_sims >= similarity_threshold))


def build_text_features(posts_by_account: dict[str, list[str]]) -> dict[str, float]:
    """Return {account_id: duplicate_content_rate} for every account."""
    return {acct: duplicate_content_rate(posts) for acct, posts in posts_by_account.items()}
