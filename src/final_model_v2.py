"""Holdout-aware final model for the hackathon add-ons.
Does not touch models/final.joblib."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from src.costs import INVESTIGATION_COST
from src.data import load_dataset
from src.final_model import category_maps, feature_list, threshold_for
from src.models import _fit_g0_raw, transform_ordinal

DEFAULT_L = 10.0
DEFAULT_C = INVESTIGATION_COST
DEFAULT_SEED = 42
V2_PATH = Path(__file__).resolve().parents[1] / "models" / "final_v2.joblib"


def train_v2(seed: int = DEFAULT_SEED, L: float = DEFAULT_L, c: float = DEFAULT_C,
             holdout_size: float = 0.20, cal_size: float = 0.25) -> dict:
    """20% holdout; the rest is split 75/25 into fit and calibration parts."""
    X, y, num_cols, cat_cols = load_dataset(verbose=False)

    X_rest, X_hold, y_rest, y_hold = train_test_split(
        X, y, test_size=holdout_size, stratify=y, random_state=seed
    )
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_rest, y_rest, test_size=cal_size, stratify=y_rest, random_state=seed
    )

    enc, model, _ = _fit_g0_raw(
        X_fit, np.asarray(y_fit).astype(int), num_cols, cat_cols, seed
    )
    X_cal_enc = transform_ordinal(enc, X_cal, num_cols, cat_cols)
    raw_cal = model.predict_proba(X_cal_enc)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_cal, np.asarray(y_cal).astype(int))

    return {
        "model": model,
        "calibrator": calibrator,
        "encoder": enc,
        "category_maps": category_maps(enc, cat_cols),
        "num_cols": list(num_cols),
        "cat_cols": list(cat_cols),
        "feature_list": feature_list(num_cols, cat_cols),
        "L": float(L),
        "c": float(c),
        "threshold": float(threshold_for(L=L, c=c)),
        "seed": int(seed),
        "fit_index": X_fit.index.to_numpy(),
        "cal_index": X_cal.index.to_numpy(),
        "holdout_index": X_hold.index.to_numpy(),
        "version": "v2",
    }


def save_v2(bundle: dict, path: Path | str = V2_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_v2(path: Path | str = V2_PATH) -> dict:
    return joblib.load(path)


def holdout_data(bundle: dict):
    """Return (X_holdout, y_holdout) for the claims the model and calibrator never saw."""
    X, y, _, _ = load_dataset(verbose=False)
    idx = bundle["holdout_index"]
    return X.loc[idx], np.asarray(y.loc[idx]).astype(int)


if __name__ == "__main__":
    b = train_v2()
    save_v2(b)
    X, y, _, _ = load_dataset(verbose=False)
    for name in ["fit_index", "cal_index", "holdout_index"]:
        idx = b[name]
        print(f"{name}: {len(idx)} claims, {int(np.asarray(y.loc[idx]).sum())} frauds")
    print("saved to", V2_PATH)