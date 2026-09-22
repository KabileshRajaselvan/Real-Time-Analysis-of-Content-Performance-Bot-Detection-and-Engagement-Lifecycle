from src.bot_classifier import build_dataset, train_and_evaluate, FEATURE_COLS


def test_build_dataset_has_all_feature_columns():
    df = build_dataset(n_accounts=150, seed=1)
    for col in FEATURE_COLS:
        assert col in df.columns
    assert "is_bot" in df.columns
    assert len(df) == 150


def test_train_and_evaluate_produces_reasonable_metrics():
    df = build_dataset(n_accounts=600, seed=1)
    metrics = train_and_evaluate(df, seed=1)
    assert 0.5 < metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    # the problem is designed to be non-trivial: a classifier that is much
    # better than random but not literally perfect
    assert metrics["roc_auc"] > 0.7
