import numpy as np

from src.engagement_lifecycle import simulate_posts, fit_curve, classify_viral_rule_based


def test_simulate_posts_shape():
    df = simulate_posts(n_posts=50)
    assert len(df) == 50
    assert set(df["is_viral_true"].unique()) <= {0, 1}
    assert all(len(c) == 72 for c in df["cumulative_engagement"])


def test_fit_curve_returns_expected_keys():
    df = simulate_posts(n_posts=5)
    curve = np.array(df.iloc[0]["cumulative_engagement"])
    params = fit_curve(curve)
    for key in ["peak_time_fit", "peak_height_fit", "decay_rate_fit", "half_life_hours"]:
        assert key in params


def test_classify_viral_rule_based_extreme_cases():
    big = {"peak_height_fit": 10000, "half_life_hours": 50}
    small = {"peak_height_fit": 5, "half_life_hours": 2}
    assert classify_viral_rule_based(big, total_engagement=5000) == 1
    assert classify_viral_rule_based(small, total_engagement=10) == 0
