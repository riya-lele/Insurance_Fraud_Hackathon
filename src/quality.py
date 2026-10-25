"""Calibration, gains and decision-curve helpers. No Streamlit imports, so they are testable."""
from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_table(y, p, n_bins: int = 8) -> pd.DataFrame:
    """Equal-size bins by predicted probability. Columns: mean_pred, frac_pos, n."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ranks = pd.Series(p).rank(method="first")
    bins = pd.qcut(ranks, n_bins, labels=False)
    df = pd.DataFrame({"bin": bins, "p": p, "y": y})
    g = df.groupby("bin").agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size"))
    return g.reset_index(drop=True)


def brier(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def ece(y, p, n_bins: int = 8) -> float:
    """Expected calibration error: bin-size-weighted mean gap between predicted and observed rate."""
    t = reliability_table(y, p, n_bins)
    return float(np.sum(t["n"] * (t["mean_pred"] - t["frac_pos"]).abs()) / t["n"].sum())


def gains_curve(y, score):
    """Return (share_of_claims_reviewed, share_of_frauds_captured), both starting at 0."""
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="stable")
    cum = np.concatenate([[0.0], np.cumsum(y[order])])
    n = len(y)
    x = np.arange(n + 1) / n
    return x, cum / y.sum()


def net_benefit(y, p, thresholds):
    """Decision curve. NB(t) = TP/n - FP/n * t/(1-t). Returns (model, treat_all)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    prev = y.mean()
    model, treat_all = [], []
    for t in thresholds:
        pred = p >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        w = t / (1 - t)
        model.append(tp / n - fp / n * w)
        treat_all.append(prev - (1 - prev) * w)
    return np.array(model), np.array(treat_all)