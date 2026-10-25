"""Batch scoring, validation and capacity analysis for the hackathon add-ons."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.final_model import calibrated_probs, decision_zone, raw_scores, threshold_for


def validate_batch(bundle: dict, df: pd.DataFrame):
    """Return (valid_rows, issues). Assumes df has a default 0..n-1 index (fresh from read_csv)."""
    feats = bundle["feature_list"]
    missing_cols = [f for f in feats if f not in df.columns]
    if missing_cols:
        raise ValueError("Missing required columns: " + ", ".join(missing_cols))

    work = df.copy()
    problems = pd.Series([""] * len(work), index=work.index, dtype=object)

    for col in bundle["num_cols"]:
        num = pd.to_numeric(work[col], errors="coerce")
        bad = num.isna()
        problems[bad] = problems[bad] + f"{col}: missing or not a number; "
        work[col] = num

    for col in bundle["cat_cols"]:
        allowed = {v for v in bundle["category_maps"][col] if v is not None}
        s = work[col]
        bad = s.isna() | ~s.isin(allowed)
        problems[bad] = problems[bad] + f"{col}: missing or unknown value; "

    ok = (problems == "").values
    issues = pd.DataFrame({
        "csv_line": np.asarray(work.index[~ok]) + 2,   # +2: header row, 1-based lines
        "problem": problems[~ok].values,
    })
    return work.loc[ok].copy(), issues


def score_batch(bundle: dict, df: pd.DataFrame, L: float, c: float) -> pd.DataFrame:
    """Score validated rows. Ranks by raw score; zones use the calibrated probability."""
    feats = bundle["feature_list"]
    X = df[feats]
    raw = raw_scores(bundle, X)
    prob = calibrated_probs(bundle, X)
    t = threshold_for(L=L, c=c)

    out = df.copy()
    out["raw_score"] = raw
    out["prob"] = prob
    out["zone"] = [decision_zone(p, t) for p in prob]
    out = out.sort_values("raw_score", ascending=False)
    out.index.name = "source_row"
    out = out.reset_index()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def capacity_summary(scored: pd.DataFrame, K: int, L: float, c: float) -> dict:
    """Review the top K claims by raw score. Costs are per claim and hypothetical."""
    n = len(scored)
    K = int(max(0, min(K, n)))
    top = scored.head(K)
    exp_top = float(top["prob"].sum())
    exp_all = float(scored["prob"].sum())
    out = {
        "n_claims": n,
        "K": K,
        "expected_frauds_in_top_K": exp_top,
        "expected_frauds_total": exp_all,
        "expected_recall_top_K": exp_top / exp_all if exp_all > 0 else float("nan"),
    }
    if "FraudFound_P" in scored.columns:
        y = scored["FraudFound_P"].astype(int)
        total = int(y.sum())
        caught = int(top["FraudFound_P"].astype(int).sum())
        out.update({
            "actual_frauds_total": total,
            "actual_frauds_in_top_K": caught,
            "recall_top_K": caught / total if total else float("nan"),
            "precision_top_K": caught / K if K else float("nan"),
            "cost_top_K": (c * K + L * (total - caught)) / n,
            "cost_investigate_none": (L * total) / n,
            "cost_investigate_all": (c * n) / n,
        })
    return out


def make_sample(bundle: dict, n: int = 300, seed: int = 1, path=None) -> Path:
    """Write a sample of holdout claims (with the true label) for the demo."""
    from src.final_model_v2 import holdout_data
    X, y = holdout_data(bundle)
    df = X.copy()
    df["FraudFound_P"] = y
    s = df.sample(n=min(n, len(df)), random_state=seed)
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "data" / "sample_claims.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    from src.final_model_v2 import load_v2
    print("wrote", make_sample(load_v2()))