import numpy as np
import pandas as pd
import pytest

from src.batch import capacity_summary, score_batch, validate_batch
from src.final_model_v2 import V2_PATH, holdout_data, load_v2
from src.score import score_claim

needs_model = pytest.mark.skipif(not V2_PATH.exists(), reason="run: python -m src.final_model_v2")


@pytest.fixture(scope="module")
def setup():
    b = load_v2()
    X, y = holdout_data(b)
    df = X.head(20).copy().reset_index(drop=True)
    df["FraudFound_P"] = y[:20]
    return b, df


@needs_model
def test_matches_single_scoring(setup):
    b, df = setup
    valid, issues = validate_batch(b, df)
    assert len(issues) == 0
    s = score_batch(b, valid, 10.0, 1.0)
    for i in range(5):
        r = int(s.loc[i, "source_row"])
        single = score_claim(b, dict(df.loc[r]), L=10.0, c=1.0)
        assert s.loc[i, "raw_score"] == pytest.approx(single["raw_score"])
        assert s.loc[i, "prob"] == pytest.approx(single["prob"])
        assert s.loc[i, "zone"] == single["zone"]


@needs_model
def test_sorted_by_risk(setup):
    b, df = setup
    valid, _ = validate_batch(b, df)
    s = score_batch(b, valid, 10.0, 1.0)
    assert s["raw_score"].is_monotonic_decreasing
    assert list(s["rank"]) == list(range(1, len(s) + 1))


@needs_model
def test_unknown_category_goes_to_issues(setup):
    b, df = setup
    bad = df.copy()
    col = b["cat_cols"][0]
    bad.loc[3, col] = "NOT_A_REAL_VALUE"
    valid, issues = validate_batch(b, bad)
    assert len(valid) == len(df) - 1
    assert len(issues) == 1 and issues.loc[0, "csv_line"] == 5


@needs_model
def test_missing_column_raises(setup):
    b, df = setup
    with pytest.raises(ValueError):
        validate_batch(b, df.drop(columns=[b["feature_list"][0]]))


def test_capacity_summary_toy_case():
    scored = pd.DataFrame({"prob": [0.9, 0.5, 0.1, 0.1], "FraudFound_P": [1, 0, 1, 0]})
    r = capacity_summary(scored, K=2, L=10.0, c=1.0)
    assert r["expected_frauds_in_top_K"] == pytest.approx(1.4)
    assert r["recall_top_K"] == pytest.approx(0.5)
    assert r["precision_top_K"] == pytest.approx(0.5)
    assert r["cost_top_K"] == pytest.approx((2 + 10 * 1) / 4)
    assert r["cost_investigate_none"] == pytest.approx(20 / 4)
    assert r["cost_investigate_all"] == pytest.approx(1.0)