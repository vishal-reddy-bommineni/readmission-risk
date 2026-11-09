"""Step 3: scoring app + population dashboard.

Usage:  streamlit run src/app.py
Tab 1 "Care-team dashboard": the held-out test set as a care team would see it.
Tab 2 "Patient risk": pick one encounter, see its calibrated risk and top drivers, try what-ifs.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import altair as alt
import joblib
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
TIERS = ["High (top 10%)", "Medium", "Low"]


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


DISCHARGE = {"1": "home", "2": "short-term hospital", "3": "skilled nursing facility",
             "5": "other inpatient care", "6": "home health service", "22": "rehab facility",
             "28": "psychiatric hospital", "7": "left AMA"}


def pretty(feature: str) -> str:
    """Turn model column names like discharge_disposition_22 into readable labels."""
    if feature.startswith("discharge_disposition_"):
        code = feature.rsplit("_", 1)[1]
        return f"Discharged to {DISCHARGE.get(code, 'code ' + code)}"
    return feature.replace("_", " ").replace("cm ", "comorbidity: ").capitalize()


def rate_chart(df, by, title, sort=None):
    g = (df.groupby(by, observed=True)
           .agg(n=("label_readmit_30d", "size"), observed=("label_readmit_30d", "mean"),
                predicted=("predicted_risk", "mean"))
           .reset_index())
    g = g[g["n"] >= 100]
    long = g.melt(id_vars=[by, "n"], value_vars=["observed", "predicted"],
                  var_name="rate", value_name="value")
    return (alt.Chart(long, title=title).mark_bar()
            .encode(x=alt.X(f"{by}:N", sort=sort, title=None,
                            axis=alt.Axis(labelAngle=0 if len(g) <= 4 else -35, labelLimit=140)),
                    xOffset="rate:N",
                    y=alt.Y("value:Q", axis=alt.Axis(format="%"), title="30-day readmission"),
                    color=alt.Color("rate:N", scale=alt.Scale(range=["#4C78A8", "#F58518"]),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=[by, "rate", alt.Tooltip("value:Q", format=".1%"), "n"])
            .properties(height=340))


st.title("30-Day Hospital Readmission Risk")
st.caption("Held-out test set, UCI Diabetes 130-US Hospitals (1999-2008). "
           "Portfolio demo only, not for clinical use.")

tab_dash, tab_patient = st.tabs(["Care-team dashboard", "Patient risk"])

# ------------------------------------------------------------------ dashboard
with tab_dash:
    n = len(scored)
    high = scored["risk_tier"] == "High (top 10%)"
    base_rate = scored["label_readmit_30d"].mean()
    captured = scored.loc[high, "label_readmit_30d"].sum() / scored["label_readmit_30d"].sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Discharges scored", f"{n:,}")
    k2.metric("Observed 30-day readmission", f"{base_rate:.1%}")
    k3.metric("Flagged high risk", f"{high.mean():.1%}")
    k4.metric("Readmissions caught by high-risk list", f"{captured:.1%}",
              f"{scored.loc[high, 'label_readmit_30d'].mean() / base_rate:.1f}x base rate",
              delta_color="off")

    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(rate_chart(scored, "risk_tier", "Readmission rate by risk tier", sort=TIERS),
                        width="stretch")
    with c2:
        st.altair_chart(rate_chart(scored, "age_band", "Observed vs predicted by age band"),
                        width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        st.altair_chart(rate_chart(scored, "primary_dx_group", "By primary diagnosis group"),
                        width="stretch")
    with c4:
        drivers = (scored.loc[high, "driver_1"].map(pretty).value_counts().head(8)
                   .rename_axis("driver").reset_index(name="patients"))
        st.altair_chart(alt.Chart(drivers, title="Top driver among high-risk patients")
                        .mark_bar(color="#4C78A8")
                        .encode(y=alt.Y("driver:N", sort="-x", title=None,
                                        axis=alt.Axis(labelLimit=260)), x="patients:Q")
                        .properties(height=340), width="stretch")

    st.subheader("Outreach list: highest-risk discharges")
    show = (scored[high].sort_values("predicted_risk", ascending=False)
            [["encounter_id", "age_band", "length_of_stay", "number_inpatient", "discharge_disposition",
              "predicted_risk", "driver_1", "driver_2", "driver_3", "label_readmit_30d"]]
            .rename(columns={"label_readmit_30d": "actually_readmitted"}))
    st.dataframe(show.head(200), hide_index=True, width="stretch",
                 column_config={"predicted_risk": st.column_config.ProgressColumn(
                     "predicted risk", format="%.2f", min_value=0, max_value=1)})

# ------------------------------------------------------------------ single patient
with tab_patient:
    left, right = st.columns([1, 2])
    with left:
        tier_filter = st.selectbox("Show encounters in tier", TIERS)
        pool = scored[scored["risk_tier"] == tier_filter]
        enc = st.selectbox("Test-set encounter", pool["encounter_id"].head(500))
        row = scored[scored["encounter_id"] == enc].copy()

        st.subheader("What-if")
        row["number_inpatient"] = st.slider("Inpatient visits in prior year", 0, 20,
                                            int(row["number_inpatient"].iloc[0]))
        row["number_emergency"] = st.slider("ER visits in prior year", 0, 20,
                                            int(row["number_emergency"].iloc[0]))
        row["length_of_stay"] = st.slider("Length of stay (days)", 1, 14,
                                          int(row["length_of_stay"].iloc[0]))
        row["num_medications"] = st.slider("Number of medications", 1, 80,
                                           int(row["num_medications"].iloc[0]))
        row["total_prior_visits"] = row["number_outpatient"] + row["number_emergency"] + row["number_inpatient"]
        row["any_prior_inpatient"] = (row["number_inpatient"] > 0).astype(int)

    with right:
        p, contrib = score(row)
        base_rate = scored["label_readmit_30d"].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted 30-day risk", f"{p:.1%}", f"{(p - base_rate) * 100:+.1f} pts vs base rate",
                  delta_color="inverse")
        c2.metric("Risk tier", tier(p))
        c3.metric("Actually readmitted <30d?", "Yes" if row["label_readmit_30d"].iloc[0] == 1 else "No")

        st.subheader("Top factors for this encounter (SHAP, log-odds)")
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
        bars = pd.DataFrame({"feature": [pretty(f) for f in top.index], "contribution": top.values})
        st.altair_chart(alt.Chart(bars).mark_bar()
                        .encode(y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
                                x="contribution:Q",
                                color=alt.condition("datum.contribution > 0", alt.value("#E45756"),
                                                    alt.value("#4C78A8")))
                        .properties(height=280), width="stretch")
        st.caption("Red bars push risk up, blue bars push it down.")

        results = REPORTS_DIR / "RESULTS.md"
        if results.exists():
            with st.expander("Model evaluation (from the last training run)"):
                st.markdown(results.read_text())
