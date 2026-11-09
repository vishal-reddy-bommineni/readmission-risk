"""Step 3: small scoring app for demos and interviews.

Usage:  streamlit run src/app.py
Pick a held-out test encounter, see its calibrated risk and top drivers, then change a few
inputs to see how the score moves.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from src.config import DASHBOARD_DIR, MODELS_DIR, REPORTS_DIR

st.set_page_config(page_title="30-Day Readmission Risk", layout="wide")


@st.cache_resource
def load():
    bundle = joblib.load(MODELS_DIR / "readmission_model.joblib")
    # categorical codes like admission_type "1" must stay strings, as they were in training
    cat_dtypes = {c: str for c in bundle["features"]["categorical"]}
    scored = pd.read_csv(DASHBOARD_DIR / "scored_test_encounters.csv", dtype=cat_dtypes)
    return bundle, scored


bundle, scored = load()
feats = bundle["features"]["numeric"] + bundle["features"]["categorical"]


def score(row: pd.DataFrame) -> tuple[float, pd.Series]:
    Xt = bundle["xgb_prep"].transform(row[feats])
    names = list(bundle["xgb_prep"].get_feature_names_out())
    raw = bundle["xgb_model"].predict_proba(Xt)[:, 1]
    contrib = bundle["xgb_model"].get_booster().predict(
        xgb.DMatrix(Xt, feature_names=names), pred_contribs=True)[0, :-1]
    if bundle["final_model"].startswith("XGB"):
        p = float(bundle["calibrator"].predict(raw)[0])
    else:
        p = float(bundle["logreg"].predict_proba(row[feats])[:, 1][0])
    return p, pd.Series(contrib, index=names)


def tier(p: float) -> str:
    if p >= bundle["threshold_top10"]:
        return "High (top 10%)"
    return "Medium" if p >= bundle["medium_cutoff"] else "Low"


st.title("30-Day Hospital Readmission Risk")
st.caption("Model trained on the public UCI Diabetes 130-US Hospitals dataset (1999-2008). "
           "Portfolio demo only, not for clinical use.")

left, right = st.columns([1, 2])
with left:
    tier_filter = st.selectbox("Show encounters in tier", ["High (top 10%)", "Medium", "Low"])
    pool = scored[scored["risk_tier"] == tier_filter]
    enc = st.selectbox("Test-set encounter", pool["encounter_id"].head(500))
    row = scored[scored["encounter_id"] == enc].copy()

    st.subheader("What-if")
    row["number_inpatient"] = st.slider("Inpatient visits in prior year", 0, 20, int(row["number_inpatient"].iloc[0]))
    row["number_emergency"] = st.slider("ER visits in prior year", 0, 20, int(row["number_emergency"].iloc[0]))
    row["length_of_stay"] = st.slider("Length of stay (days)", 1, 14, int(row["length_of_stay"].iloc[0]))
    row["num_medications"] = st.slider("Number of medications", 1, 80, int(row["num_medications"].iloc[0]))
    row["total_prior_visits"] = row["number_outpatient"] + row["number_emergency"] + row["number_inpatient"]
    row["any_prior_inpatient"] = (row["number_inpatient"] > 0).astype(int)

with right:
    p, contrib = score(row)
    base_rate = scored["label_readmit_30d"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted 30-day risk", f"{p:.1%}", f"{(p - base_rate) * 100:+.1f} pts vs base rate")
    c2.metric("Risk tier", tier(p))
    c3.metric("Actually readmitted <30d?", "Yes" if row["label_readmit_30d"].iloc[0] == 1 else "No")

    st.subheader("Top factors for this encounter (SHAP, log-odds)")
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
    st.bar_chart(top.rename("contribution").to_frame(), horizontal=True)
    st.caption("Positive bars push risk up, negative bars push it down.")

    results = REPORTS_DIR / "RESULTS.md"
    if results.exists():
        with st.expander("Model evaluation (from the last training run)"):
            st.markdown(results.read_text())
