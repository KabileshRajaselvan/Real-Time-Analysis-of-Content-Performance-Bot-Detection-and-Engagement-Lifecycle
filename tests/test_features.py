import numpy as np

from src.features import clean_text, duplicate_content_rate
from src.synthetic_accounts import generate_accounts


def test_clean_text_strips_noise():
    raw = "Check THIS out @someone #cool http://x.co !!! 😀"
    cleaned = clean_text(raw)
    assert "@" not in cleaned
    assert "#" not in cleaned
    assert "http" not in cleaned
    assert cleaned == cleaned.lower()


def test_duplicate_content_rate_identical_posts_is_high():
    posts = ["hello world this is a post"] * 5
    rate = duplicate_content_rate(posts, similarity_threshold=0.8)
    assert rate == 1.0


def test_duplicate_content_rate_distinct_posts_is_low():
    posts = [
        "the weather today is sunny and warm",
        "quantum computers use qubits for parallelism",
        "my dog learned a new trick yesterday",
        "the stock market fell sharply this morning",
    ]
    rate = duplicate_content_rate(posts, similarity_threshold=0.8)
    assert rate < 0.3


def test_duplicate_content_rate_handles_few_posts():
    assert duplicate_content_rate([]) == 0.0
    assert duplicate_content_rate(["only one post"]) == 0.0


def test_generate_accounts_shape_and_labels():
    df, posts = generate_accounts(n_accounts=200, bot_fraction=0.3, seed=1)
    assert len(df) == 200
    assert set(df["is_bot"].unique()) <= {0, 1}
    assert abs(df["is_bot"].mean() - 0.3) < 0.05
    assert len(posts) == 200
    # bots should on average post more regularly (lower interval CV)
    bot_cv = df.loc[df.is_bot == 1, "posting_interval_cv"].mean()
    human_cv = df.loc[df.is_bot == 0, "posting_interval_cv"].mean()
    assert bot_cv < human_cv
