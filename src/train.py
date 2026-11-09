"""Step 2: split, train, calibrate, evaluate, explain, audit.

Usage:  python -m src.train            (add --n-boot 1000 for tighter confidence intervals)

Everything this script reports is written to reports/ so the README numbers always come
from a run, never from memory.
"""
import argparse
import json
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from src.calibration import IsotonicCalibrator, PlattCalibrator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (AUDIT_COLS, BASELINE_FEATURES, CATEGORICAL_FEATURES, DASHBOARD_DIR,
                        FEATURES_PARQUET, FIG_DIR, GROUP_COL, MODELS_DIR, NUMERIC_FEATURES,
                        REPORTS_DIR, SEED, TARGET, TOP_K_FRACTION)

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ----------------------------------------------------------------------------- splitting
def grouped_split(df: pd.DataFrame):
    """70/15/15 train/val/test with every patient in exactly one split.

    Patients appear many times in this dataset. A random encounter-level split lets the model
    'recognise' a patient it saw in training, which inflates test scores (see leakage_check).
    """
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    trainval_idx, test_idx = next(outer.split(df, groups=df[GROUP_COL]))
    trainval = df.iloc[trainval_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=SEED)
    tr_idx, va_idx = next(inner.split(trainval, groups=trainval[GROUP_COL]))
    train, val, test = trainval.iloc[tr_idx], trainval.iloc[va_idx], df.iloc[test_idx]

    pats = [set(s[GROUP_COL]) for s in (train, val, test)]
    assert not (pats[0] & pats[1]) and not (pats[0] & pats[2]) and not (pats[1] & pats[2]), \
        "patient overlap between splits"
    return train.copy(), val.copy(), test.copy()


# ----------------------------------------------------------------------------- models
def make_preprocessor(numeric, categorical, scale: bool):
    num = StandardScaler() if scale else "passthrough"
    transformers = [("num", num, numeric)]
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                                                  sparse_output=False), categorical))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def train_logistic(train, numeric, categorical):
    pipe = Pipeline([
        ("prep", make_preprocessor(numeric, categorical, scale=True)),
        ("clf", LogisticRegression(max_iter=3000, C=1.0)),
    ])
    pipe.fit(train[numeric + categorical], train[TARGET])
    return pipe


def train_xgboost(train, val, numeric=None, categorical=None):
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical
    prep = make_preprocessor(numeric, categorical, scale=False)
    X_tr = prep.fit_transform(train[numeric + categorical])
    X_va = prep.transform(val[numeric + categorical])
    model = xgb.XGBClassifier(
        n_estimators=3000, learning_rate=0.03, max_depth=4, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="aucpr", early_stopping_rounds=150, random_state=SEED, n_jobs=-1,
    )
    model.fit(X_tr, train[TARGET], eval_set=[(X_va, val[TARGET])], verbose=False)
    return prep, model


# ----------------------------------------------------------------------------- metrics
def top_k_stats(y, p, frac=TOP_K_FRACTION, threshold=None):
    """Precision / recall / lift when the care team acts on the riskiest `frac` of discharges."""
    if threshold is None:
        # flag EXACTLY the top `frac` by rank. A quantile cut-off would flag more than `frac`
        # for models with many tied scores (e.g. the 3-count baseline), an unfair comparison.
        k = int(round(frac * len(p)))
        order = np.argsort(-p, kind="stable")
        flagged = np.zeros(len(p), dtype=bool)
        flagged[order[:k]] = True
        threshold = float(p[order[k - 1]])
    else:
        flagged = p >= threshold
    prevalence = y.mean()
    precision = y[flagged].mean() if flagged.any() else np.nan
    recall = y[flagged].sum() / max(y.sum(), 1)
    return {"threshold": float(threshold), "flag_rate": float(flagged.mean()),
            "precision": float(precision), "recall": float(recall),
            "lift": float(precision / prevalence) if prevalence > 0 else np.nan}


def point_metrics(y, p):
    return {"auroc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p)}


def cluster_bootstrap_ci(y, p, groups, n_boot, seed=SEED):
    """95% CI that resamples PATIENTS (not encounters), since one patient's visits are correlated."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    n_pat = len(uniq)
    draws = {"auroc": [], "pr_auc": [], "brier": []}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat)[codes]
        if y[w > 0].min() == y[w > 0].max():
            continue
        draws["auroc"].append(roc_auc_score(y, p, sample_weight=w))
        draws["pr_auc"].append(average_precision_score(y, p, sample_weight=w))
        draws["brier"].append(brier_score_loss(y, p, sample_weight=w))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in draws.items()}


# ----------------------------------------------------------------------------- leakage demo
def leakage_check(df):
    """Train with the WRONG (random encounter-level) split, then compare, inside that same test
    set, encounters whose patient was also in training vs patients the model never saw.
    The gap is how much a naive split flatters the model on returning patients."""
    rng = np.random.default_rng(SEED)
    is_test = rng.random(len(df)) < 0.15
    tr, te = df[~is_test], df[is_test]
    is_es = rng.random(len(tr)) < 0.15
    prep, model = train_xgboost(tr[~is_es], tr[is_es])
    p = model.predict_proba(prep.transform(te[ALL_FEATURES]))[:, 1]
    seen = te[GROUP_COL].isin(set(tr[GROUP_COL])).to_numpy()
    y = te[TARGET].to_numpy()
    out = {"share_of_test_encounters_with_patient_in_train": float(seen.mean()),
           "naive_split_auroc_all": float(roc_auc_score(y, p)),
           "naive_split_pr_auc_all": float(average_precision_score(y, p))}
    for label, m in (("seen_patients", seen), ("unseen_patients", ~seen)):
        out[f"n_{label}"] = int(m.sum())
        out[f"readmit_rate_{label}"] = float(y[m].mean())
        out[f"auroc_{label}"] = float(roc_auc_score(y[m], p[m]))
        out[f"pr_auc_{label}"] = float(average_precision_score(y[m], p[m]))
    return out


def choose_calibrator(p_val, y_val, groups_val):
    """Pick Platt vs isotonic using 2-fold patient-grouped cross-fitting INSIDE the validation
    set (fit on one half, score log loss on the other), then refit the winner on all of it."""
    from sklearn.metrics import log_loss
    from sklearn.model_selection import GroupKFold
    scores = {}
    for cls in (PlattCalibrator, IsotonicCalibrator):
        losses, briers = [], []
        for a, b in GroupKFold(n_splits=2).split(p_val, groups=groups_val):
            c = cls().fit(p_val[a], y_val[a])
            q = np.clip(c.predict(p_val[b]), 1e-6, 1 - 1e-6)
            losses.append(log_loss(y_val[b], q))
            briers.append(brier_score_loss(y_val[b], q))
        scores[cls.name] = {"log_loss": float(np.mean(losses)), "brier": float(np.mean(briers))}
    best = min(scores, key=lambda k: scores[k]["log_loss"])
    winner = {"Platt": PlattCalibrator, "Isotonic": IsotonicCalibrator}[best]().fit(p_val, y_val)
    return winner, scores


# ----------------------------------------------------------------------------- plots
def plot_roc_pr(y, preds: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUROC {roc_auc_score(y, p):.3f})")
        axes[1].plot(rec, prec, label=f"{name} (PR-AUC {average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].axhline(y.mean(), color="k", ls="--", lw=0.8, label=f"prevalence {y.mean():.3f}")
    axes[0].set(title="ROC (test set)", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision-recall (test set)", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_pr.png", dpi=150)
    plt.close(fig)


def plot_calibration(y, preds: dict):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in preds.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", label=f"{name} (Brier {brier_score_loss(y, p):.4f})")
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="perfect calibration")
    ax.set(title="Calibration (test set, deciles)", xlabel="Mean predicted risk",
           ylabel="Observed readmission rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration.png", dpi=150)
    plt.close(fig)


def shap_values(prep, model, X):
    """Per-feature contributions (log-odds) using XGBoost's built-in TreeSHAP."""
    Xt = prep.transform(X)
    names = list(prep.get_feature_names_out())
    contribs = model.get_booster().predict(xgb.DMatrix(Xt, feature_names=names), pred_contribs=True)
    return contribs[:, :-1], Xt, names


def plot_shap(sv, Xt, names, n=4000):
    import shap
    idx = np.random.default_rng(SEED).choice(len(sv), size=min(n, len(sv)), replace=False)
    shap.summary_plot(sv[idx], pd.DataFrame(Xt[idx], columns=names), max_display=15, show=False)
    plt.title("Top drivers of predicted 30-day readmission risk (XGBoost, test set)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
    return mean_abs


# ----------------------------------------------------------------------------- subgroup audit
def subgroup_audit(test, p, threshold):
    rows = []
    y = test[TARGET].to_numpy()
    for col in AUDIT_COLS:
        for level, mask in test.groupby(col).groups.items():
            m = test.index.isin(mask)
            if m.sum() < 100:
                continue
            yy, pp = y[m], p[m]
            rows.append({
                "attribute": col, "group": level, "n": int(m.sum()),
                "observed_rate": yy.mean(), "mean_predicted": pp.mean(),
                "auroc": roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan,
                "flagged_rate": (pp >= threshold).mean(),
                "recall_at_global_threshold": ((pp >= threshold) & (yy == 1)).sum() / max(yy.sum(), 1),
            })
    out = pd.DataFrame(rows).round(4)
    out.to_csv(REPORTS_DIR / "subgroup_audit.csv", index=False)
    return out


# ----------------------------------------------------------------------------- main
def main(n_boot: int, skip_leakage: bool):
    for d in (MODELS_DIR, FIG_DIR, DASHBOARD_DIR):
        d.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEATURES_PARQUET).reset_index(drop=True)
    train, val, test = grouped_split(df)
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,} encounters "
          f"(prevalence {train[TARGET].mean():.3f} / {val[TARGET].mean():.3f} / {test[TARGET].mean():.3f})")

    # --- models
    base = train_logistic(train, BASELINE_FEATURES, [])
    lr = train_logistic(train, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    xprep, xmodel = train_xgboost(train, val)
    print(f"XGBoost stopped at {xmodel.best_iteration + 1} trees")

    def xgb_proba(frame):
        return xmodel.predict_proba(xprep.transform(frame[ALL_FEATURES]))[:, 1]

    # --- calibrate XGBoost on the validation set (never on test); method chosen inside validation
    iso, calib_scores = choose_calibrator(xgb_proba(val), val[TARGET].to_numpy(),
                                          val[GROUP_COL].to_numpy())
    calib_name = f"XGBoost + {iso.name} calibration"
    print(f"calibration (val cross-fit log loss): {calib_scores} -> {iso.name}")

    y = test[TARGET].to_numpy()
    preds = {
        "Prevalence only": np.full(len(test), train[TARGET].mean()),
        "LogReg, 3 utilization features": base.predict_proba(test[BASELINE_FEATURES])[:, 1],
        "LogReg, all features": lr.predict_proba(test[ALL_FEATURES])[:, 1],
        "XGBoost": xgb_proba(test),
    }
    preds[calib_name] = iso.predict(preds["XGBoost"])

    # --- pick the final model on VALIDATION performance, not test
    val_auc_lr = roc_auc_score(val[TARGET], lr.predict_proba(val[ALL_FEATURES])[:, 1])
    val_auc_xgb = roc_auc_score(val[TARGET], xgb_proba(val))
    final_name = calib_name if val_auc_xgb >= val_auc_lr else "LogReg, all features"
    print(f"validation AUROC: LogReg {val_auc_lr:.4f} | XGBoost {val_auc_xgb:.4f} -> final = {final_name}")

    # threshold for "top 10%" is fixed on validation, then applied to test
    val_final = iso.predict(xgb_proba(val)) if final_name.startswith("XGB") else \
        lr.predict_proba(val[ALL_FEATURES])[:, 1]
    threshold = float(np.quantile(val_final, 1 - TOP_K_FRACTION))

    # --- metrics
    groups = test[GROUP_COL].to_numpy()
    results = {}
    for name, p in preds.items():
        r = {k: float(v) for k, v in point_metrics(y, p).items()}
        if name != "Prevalence only":
            r["ci95"] = cluster_bootstrap_ci(y, p, groups, n_boot)
            r["top_10pct"] = top_k_stats(y, p)
        results[name] = r
        print(f"{name:34s} AUROC {r['auroc']:.4f}  PR-AUC {r['pr_auc']:.4f}  Brier {r['brier']:.4f}")
    results["final_model"] = final_name
    results["calibration_choice"] = {"chosen": iso.name, "validation_crossfit": calib_scores}
    results["validation_auroc"] = {"LogReg, all features": float(val_auc_lr), "XGBoost": float(val_auc_xgb)}
    results["final_at_validation_threshold"] = top_k_stats(y, preds[final_name], threshold=threshold)

    # --- figures
    plot_roc_pr(y, {k: v for k, v in preds.items() if k not in ("Prevalence only", calib_name)})
    plot_calibration(y, {k: preds[k] for k in ("LogReg, all features", "XGBoost", calib_name)})
    sv, Xt, names = shap_values(xprep, xmodel, test[ALL_FEATURES])
    importance = plot_shap(sv, Xt, names)
    importance.head(20).to_csv(REPORTS_DIR / "shap_importance_top20.csv", header=["mean_abs_shap"])

    # --- subgroup audit
    audit = subgroup_audit(test.reset_index(drop=True), preds[final_name], threshold)

    # --- leakage demonstration
    if not skip_leakage:
        results["leakage_check"] = leakage_check(df)
        print("naive encounter-level split:", results["leakage_check"])

    # --- dashboard extract: one row per test encounter with risk + top 3 drivers
    top3 = np.argsort(-sv, axis=1)[:, :3]
    dash = test[["encounter_id", "patient_nbr", TARGET] + AUDIT_COLS + ALL_FEATURES].copy()
    dash["predicted_risk"] = preds[final_name]
    dash["risk_tier"] = pd.cut(dash["predicted_risk"],
                               [-1, np.quantile(val_final, 0.5), threshold, 2],
                               labels=["Low", "Medium", "High (top 10%)"])
    for i in range(3):
        dash[f"driver_{i + 1}"] = [names[j] for j in top3[:, i]]
    dash.to_csv(DASHBOARD_DIR / "scored_test_encounters.csv", index=False)

    # --- save artifacts
    joblib.dump({"xgb_prep": xprep, "xgb_model": xmodel, "calibrator": iso, "logreg": lr,
                 "final_model": final_name, "threshold_top10": threshold,
                 "medium_cutoff": float(np.quantile(val_final, 0.5)),
                 "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES}},
                MODELS_DIR / "readmission_model.joblib")
    results["run"] = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                      "n_train": len(train), "n_val": len(val), "n_test": len(test),
                      "n_boot": n_boot, "xgb_trees": int(xmodel.best_iteration + 1)}
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    write_results_md(results, audit, importance)
    print(f"\nDone. See {REPORTS_DIR / 'RESULTS.md'}")


def write_results_md(res, audit, importance):
    lines = ["# Results (auto-generated by `python -m src.train` - do not edit by hand)", "",
             f"Run: {res['run']['timestamp']} | test n = {res['run']['n_test']:,} encounters | "
             f"95% CIs from {res['run']['n_boot']} patient-level bootstrap resamples", "",
             "| Model | AUROC (95% CI) | PR-AUC (95% CI) | Brier | Top-10% precision | Top-10% recall | Lift |",
             "|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        if not isinstance(r, dict) or "auroc" not in r:
            continue
        if "ci95" in r:
            ci = r["ci95"]
            t = r["top_10pct"]
            lines.append(f"| {name} | {r['auroc']:.3f} ({ci['auroc'][0]:.3f}-{ci['auroc'][1]:.3f}) | "
                         f"{r['pr_auc']:.3f} ({ci['pr_auc'][0]:.3f}-{ci['pr_auc'][1]:.3f}) | "
                         f"{r['brier']:.4f} | {t['precision']:.3f} | {t['recall']:.3f} | {t['lift']:.2f}x |")
        else:
            lines.append(f"| {name} | {r['auroc']:.3f} | {r['pr_auc']:.3f} | {r['brier']:.4f} | - | - | - |")
    f = res["final_at_validation_threshold"]
    lines += ["", f"**Final model:** {res['final_model']} (chosen on validation AUROC).",
              f"At the threshold fixed on validation it flags {f['flag_rate']:.1%} of test discharges, "
              f"catching {f['recall']:.1%} of 30-day readmissions with precision {f['precision']:.1%} "
              f"({f['lift']:.2f}x the base rate).", ""]
    cc = res["calibration_choice"]
    lines += ["## Calibration choice", "",
              "Chosen by 2-fold patient-grouped cross-fitting inside the validation set (lower log loss wins):", "",
              "| Method | Log loss | Brier |", "|---|---|---|"]
    lines += [f"| {k} | {v['log_loss']:.5f} | {v['brier']:.5f} |" for k, v in cc["validation_crossfit"].items()]
    lines += ["", f"Chosen: **{cc['chosen']}**.", ""]
    if "leakage_check" in res:
        lc = res["leakage_check"]
        lines += ["## Leakage check (what a naive encounter-level split would do)", "",
                  f"With a random encounter-level split, {lc['share_of_test_encounters_with_patient_in_train']:.1%} "
                  "of test encounters belong to a patient who is also in training. Same model, same test set:", "",
                  "| Test encounters | n | 30-day rate | AUROC | PR-AUC |", "|---|---|---|---|---|",
                  f"| Patient also in training | {lc['n_seen_patients']:,} | {lc['readmit_rate_seen_patients']:.3f} | "
                  f"{lc['auroc_seen_patients']:.3f} | {lc['pr_auc_seen_patients']:.3f} |",
                  f"| Patient never seen | {lc['n_unseen_patients']:,} | {lc['readmit_rate_unseen_patients']:.3f} | "
                  f"{lc['auroc_unseen_patients']:.3f} | {lc['pr_auc_unseen_patients']:.3f} |",
                  f"| All (naive headline) | - | - | {lc['naive_split_auroc_all']:.3f} | {lc['naive_split_pr_auc_all']:.3f} |",
                  ""]
    lines += ["## Top 10 drivers (mean |SHAP|, log-odds)", "",
              importance.head(10).round(4).to_frame("mean_abs_shap").to_markdown(), "",
              "## Subgroup audit (final model, test set)", "", audit.to_markdown(index=False), ""]
    (REPORTS_DIR / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--skip-leakage", action="store_true")
    a = ap.parse_args()
    main(a.n_boot, a.skip_leakage)
