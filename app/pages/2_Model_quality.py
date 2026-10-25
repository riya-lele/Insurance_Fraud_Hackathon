import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score

from src.final_model import calibrated_probs, raw_scores
from src.final_model_v2 import V2_PATH, holdout_data, load_v2
from src.quality import brier, ece, gains_curve, net_benefit, reliability_table

st.set_page_config(page_title="Model quality", page_icon="📈", layout="wide")

if not V2_PATH.exists():
    st.error("models/final_v2.joblib not found. Run: python -m src.final_model_v2")
    st.stop()


@st.cache_resource
def get_bundle():
    return load_v2()


@st.cache_data
def get_holdout_scores():
    b = get_bundle()
    X, y = holdout_data(b)
    Xf = X[b["feature_list"]]
    return y, raw_scores(b, Xf), calibrated_probs(b, Xf)


bundle = get_bundle()
y, raw, prob = get_holdout_scores()

st.title("📈 Model quality on unseen claims")
st.write("All figures use the holdout set: claims that neither the model nor the calibrator saw. "
         "Costs are hypothetical.")

st.sidebar.header("Cost setting")
L = st.sidebar.slider("How much worse is a missed fraud than one investigation?",
                      2, 100, int(bundle["L"]), key="L")
c = float(bundle["c"])
t = c / (L + c)
st.sidebar.write(f"Decision threshold: **{t:.1%}**")

base = float(y.mean())
n_bins = st.sidebar.slider("Calibration bins", 4, 12, 8)

# ---- headline numbers ----
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Holdout claims", len(y))
m2.metric("Frauds", int(y.sum()), f"{base:.1%} base rate", delta_color="off")
m3.metric("PR-AUC", f"{average_precision_score(y, raw):.3f}", f"chance = {base:.3f}", delta_color="off")
m4.metric("ROC-AUC", f"{roc_auc_score(y, raw):.3f}")
m5.metric("Brier score", f"{brier(y, prob):.4f}", f"ECE {ece(y, prob, n_bins):.3f}", delta_color="off")
st.caption("PR-AUC and ROC-AUC use the raw score (ranking). Brier and ECE use the calibrated probability. "
           "Lower Brier and ECE are better. With about 185 frauds, every number here is noisy.")

c1, c2, c3 = st.columns(3)

# ---- reliability ----
with c1:
    st.subheader("Calibration")
    rt = reliability_table(y, prob, n_bins)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    top = max(rt["mean_pred"].max(), rt["frac_pos"].max()) * 1.1
    ax.plot([0, top], [0, top], "--", color="grey", label="perfect")
    ax.plot(rt["mean_pred"], rt["frac_pos"], "o-", color="#4C78A8", label="model")
    ax.set_xlabel("Predicted fraud probability")
    ax.set_ylabel("Observed fraud rate")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Points near the dashed line mean a predicted 10% really behaves like about 10%.")

# ---- gains ----
with c2:
    st.subheader("Gains (lift)")
    gx, gy = gains_curve(y, raw)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot(gx, gy, color="#4C78A8", label="model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="random")
    ax.set_xlabel("Share of claims reviewed (riskiest first)")
    ax.set_ylabel("Share of all frauds found")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    for k in (0.05, 0.10, 0.20):
        i = int(round(k * len(y)))
        st.write(f"Review the top **{k:.0%}** of claims: find **{gy[i]:.0%}** of frauds "
                 f"({gy[i] / k:.1f}x better than random).")

# ---- net benefit ----
with c3:
    st.subheader("Net benefit")
    ts = np.linspace(0.01, 0.5, 50)
    nb, nb_all = net_benefit(y, prob, ts)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot(ts, nb, color="#4C78A8", label="model")
    ax.plot(ts, nb_all, color="#E45756", label="investigate all")
    ax.axhline(0, color="grey", label="investigate none")
    ax.axvline(t, color="black", ls=":", label=f"current threshold {t:.1%}")
    ax.set_ylim(min(-0.02, nb.min() * 1.2), max(nb.max(), 0) * 1.3 + 0.005)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Net benefit (frauds per claim)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("A higher line is better. It counts true frauds found minus false alarms weighted by "
               "the threshold. It is the same cost trade-off as the threshold c/(L+c), "
               "shown as a curve, not new information.")

st.subheader("Calibration table")
st.dataframe(rt.rename(columns={"mean_pred": "mean predicted", "frac_pos": "observed fraud rate", "n": "claims"})
             .style.format({"mean predicted": "{:.1%}", "observed fraud rate": "{:.1%}"}),
             use_container_width=True)
st.caption("ECE depends on the number of bins. Small bins with few frauds are noisy, "
           "and these figures come from a single holdout split.")
           