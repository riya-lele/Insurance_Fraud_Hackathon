import pytest

from src.final_model_v2 import V2_PATH, holdout_data, load_v2

pytestmark = pytest.mark.skipif(
    not V2_PATH.exists(), reason="run: python -m src.final_model_v2"
)


def test_index_sets_do_not_overlap():
    b = load_v2()
    fit, cal, hold = set(b["fit_index"]), set(b["cal_index"]), set(b["holdout_index"])
    assert not (fit & cal)
    assert not (fit & hold)
    assert not (cal & hold)


def test_scores_a_holdout_row():
    from src.score import score_claim
    b = load_v2()
    X, y = holdout_data(b)
    r = score_claim(b, dict(X.iloc[0]))
    assert 0.0 <= r["prob"] <= 1.0
    assert r["zone"] in {"approve", "review", "investigate"}