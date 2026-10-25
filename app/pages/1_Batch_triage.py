import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.batch import capacity_summary, score_batch, validate_batch
from src.final_model_v2 import V2_PATH, load_v2

st.set_page_config(page_title="Batch triage", page_icon="📋", layout="wide")

if not V2_PATH.exists():
    st.error("models/final_v2.joblib not found. Run: python -m src.final_model_v2")
    st.stop()


@st.cache_resource
def get_bundle():
    return load_v2()


bundle = get_bundle()

st.title("📋 Batch triage: rank a list of claims")
st.write("Upload a CSV of claims. Every claim is scored, ranked by risk and given a decision. "
         "Costs are hypothetical.")

# ---- sidebar ----
st.sidebar.header("Cost setting")
L = st.sidebar.slider("How much worse is a missed fraud than one investigation?",
                      2, 100, int(bundle["L"]), key="L")
c = float(bundle["c"])
t = c / (L + c)
st.sidebar.write(f"Claims at **{t:.1%}** risk or more go to review, **{2*t:.1%}** or more to investigation.")

# ---- input ----
source = st.radio("Claims source", ["Upload a CSV", "Use sample claims"], horizontal=True)
df = None
if source == "Upload a CSV":
    up = st.file_uploader("CSV with one row per claim", type="csv")
    if up is not None:
        df = pd.read_csv(up)
else:
    sample = ROOT / "data" / "sample_claims.csv"
    if sample.exists():
        df = pd.read_csv(sample)
    else:
        st.warning("No sample file yet. Run: python -m src.batch")

if df is None:
    st.info("Choose a source above. The CSV needs these columns: " + ", ".join(bundle["feature_list"])
            + ". A FraudFound_P column (0/1) is optional and adds evaluation figures.")
    st.stop()

# ---- validate and score ----
try:
    valid, issues = validate_batch(bundle, df)
except ValueError as err:
    st.error(str(err))
    st.stop()

if len(issues):
    st.warning(f"{len(issues)} row(s) skipped because of data problems.")
    st.dataframe(issues, use_container_width=True)
if valid.empty:
    st.error("No valid rows to score.")
    st.stop()

scored = score_batch(bundle, valid, L, c)

# ---- KPIs ----
zc = scored["zone"].value_counts()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Claims scored", len(scored))
k2.metric("Investigate", int(zc.get("investigate", 0)))
k3.metric("Review", int(zc.get("review", 0)))
k4.metric("Approve", int(zc.get("approve", 0)))
flagged = int(zc.get("investigate", 0) + zc.get("review", 0))
st.caption(f"{flagged / len(scored):.1%} of claims flagged. "
           f"Expected frauds in this batch (sum of probabilities): {scored['prob'].sum():.1f}.")

# ---- ranked worklist ----
st.subheader("Ranked worklist (highest risk first)")
lead = ["rank", "source_row", "zone", "prob", "raw_score"]
if "FraudFound_P" in scored.columns:
    lead.append("FraudFound_P")
rest = [col for col in scored.columns if col not in lead]
view = scored[lead + rest].head(500)
bg = {"approve": "rgba(46,160,67,0.25)", "review": "rgba(210,153,34,0.35)",
      "investigate": "rgba(248,81,73,0.40)"}
styled = (view.style
          .apply(lambda r: [f"background-color: {bg[r['zone']]}"] * len(r), axis=1)
          .format({"prob": "{:.1%}", "raw_score": "{:.3f}"}))
st.dataframe(styled, height=420, use_container_width=True)
if len(scored) > 500:
    st.caption("Showing the first 500 rows. The download contains all rows.")
st.download_button("Download ranked CSV", scored.to_csv(index=False).encode("utf-8"),
                   file_name="ranked_claims.csv", mime="text/csv")

# ---- capacity mode ----
st.subheader("Capacity mode: how many claims can your team review?")
K = st.number_input("Claims reviewable today (K)", min_value=1, max_value=len(scored),
                    value=min(50, len(scored)), step=1)
cap = capacity_summary(scored, int(K), L, c)
m1, m2, m3 = st.columns(3)
m1.metric("Expected frauds in top K", f"{cap['expected_frauds_in_top_K']:.1f}")
m2.metric("Expected share of all frauds", f"{cap['expected_recall_top_K']:.0%}")
m3.metric("Reviewing", f"{cap['K']} of {cap['n_claims']}")

if "actual_frauds_total" in cap:
    a1, a2, a3 = st.columns(3)
    a1.metric("Actual frauds in top K", f"{cap['actual_frauds_in_top_K']} of {cap['actual_frauds_total']}")
    a2.metric("Recall in top K", f"{cap['recall_top_K']:.0%}")
    a3.metric("Precision in top K", f"{cap['precision_top_K']:.0%}")
    st.write("Cost per claim under the three policies (lower is better):")
    st.bar_chart(pd.DataFrame({"cost per claim": {
        f"Review top {cap['K']}": cap["cost_top_K"],
        "Investigate nothing": cap["cost_investigate_none"],
        "Investigate everything": cap["cost_investigate_all"]}}))
    st.caption("Uses the true labels in your file. Costs are hypothetical: an investigation costs 1, "
               f"a missed fraud costs {L}. Small batches give noisy results.")
else:
    st.caption("Add a FraudFound_P column to see actual recall, precision and cost.")