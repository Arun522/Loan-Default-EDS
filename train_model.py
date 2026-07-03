"""EWS prototype pipeline: baseline vs structured-only vs full (structured+text) models.

Outputs:
  metrics.json     - benchmark table (baseline scorecard vs EWS models)
  dashboard_data.json - scored portfolio for the dashboard (heatmap, watchlist, drill-down)
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
# Platt scaling used for calibration
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss

df = pd.read_csv("portfolio.csv")
STRUCT = ["bureau_score", "score_migration_3m", "dpd_current", "dpd_trend_6m",
          "emi_bounce_6m", "util_ratio", "util_spike_3m", "cashflow_coverage",
          "salary_regularity", "balance_volatility", "new_enquiries_3m",
          "leverage_buildup", "ltv_drift", "gst_filing_gap_m", "interest_coverage"]
TEXT = ["news_adverse_severity", "legal_event_flag", "call_sentiment_drift",
        "hardship_mentions_6m", "audit_qualified_flag", "web_footprint_decline"]
FEATURE_LABELS = {
    "bureau_score": "Bureau score level", "score_migration_3m": "Bureau score migration",
    "dpd_current": "Days past due", "dpd_trend_6m": "DPD trend (6m)",
    "emi_bounce_6m": "EMI bounce frequency", "util_ratio": "Limit utilisation",
    "util_spike_3m": "Utilisation spike", "cashflow_coverage": "Cash-flow coverage",
    "salary_regularity": "Salary-credit regularity", "balance_volatility": "Balance volatility",
    "new_enquiries_3m": "New credit enquiries", "leverage_buildup": "Leverage build-up",
    "ltv_drift": "LTV drift", "gst_filing_gap_m": "GST filing gaps",
    "interest_coverage": "Interest coverage", "news_adverse_severity": "Adverse news event",
    "legal_event_flag": "Legal / insolvency filing", "call_sentiment_drift": "Call-centre sentiment",
    "hardship_mentions_6m": "Hardship mentions in calls", "audit_qualified_flag": "Qualified audit opinion",
    "web_footprint_decline": "Web footprint decline",
}
tr, te = df[df.vintage == "train"], df[df.vintage == "oot"]
y_tr, y_te = tr.default_12m.values, te.default_12m.values

def flag_threshold(scores, budget=0.05):
    """Red-flag the top `budget` share of the book (operating point)."""
    return np.quantile(scores, 1 - budget)

def evaluate(name, s_tr, s_te):
    thr = flag_threshold(s_tr)
    flags = s_te >= thr
    return {
        "model": name,
        "auc": round(float(roc_auc_score(y_te, s_te)), 3),
        "precision_red": round(float(precision_score(y_te, flags, zero_division=0)), 3),
        "capture_red": round(float(recall_score(y_te, flags)), 3),
        "flag_rate": round(float(flags.mean()), 3),
    }

results = []

# 1) Baseline "legacy scorecard": bureau score + current DPD only (logistic)
base_cols = ["bureau_score", "dpd_current"]
lr = LogisticRegression(max_iter=1000).fit(tr[base_cols], y_tr)
s_tr_b, s_te_b = lr.predict_proba(tr[base_cols])[:, 1], lr.predict_proba(te[base_cols])[:, 1]
results.append(evaluate("Legacy scorecard (bureau + DPD)", s_tr_b, s_te_b))

def fit_xgb(cols):
    m = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=(1 - y_tr.mean()) / y_tr.mean(),
        eval_metric="aucpr", random_state=7)
    m.fit(tr[cols], y_tr)
    return m, m.predict_proba(tr[cols])[:, 1], m.predict_proba(te[cols])[:, 1]

# 2) EWS structured-only
m_s, s_tr_s, s_te_s = fit_xgb(STRUCT)
results.append(evaluate("EWS Phase 1 - structured only (XGBoost)", s_tr_s, s_te_s))

# 3) EWS full: structured + unstructured signals
FULL = STRUCT + TEXT
m_f, s_tr_f, s_te_f = fit_xgb(FULL)
results.append(evaluate("EWS Phase 2 - structured + text signals", s_tr_f, s_te_f))

# calibration (Platt scaling on train scores -> smooth calibrated PDs)
logit_tr = np.log(np.clip(s_tr_f, 1e-6, 1 - 1e-6) / (1 - np.clip(s_tr_f, 1e-6, 1 - 1e-6)))
logit_te = np.log(np.clip(s_te_f, 1e-6, 1 - 1e-6) / (1 - np.clip(s_te_f, 1e-6, 1 - 1e-6)))
platt = LogisticRegression(max_iter=1000).fit(logit_tr.reshape(-1, 1), y_tr)
pd_cal = np.clip(platt.predict_proba(logit_te.reshape(-1, 1))[:, 1], 0.0005, 0.97)
brier = round(float(brier_score_loss(y_te, pd_cal)), 4)

# SHAP-style contributions via XGBoost pred_contribs (exact TreeSHAP)
booster = m_f.get_booster()
dm = xgb.DMatrix(te[FULL])
contribs = booster.predict(dm, pred_contribs=True)[:, :-1]  # drop bias term

# Unified Risk Grade (1-10) from calibrated PD
GRADE_BANDS = [0.005, 0.01, 0.02, 0.035, 0.06, 0.10, 0.15, 0.25, 0.40]
def grade(p):
    return int(np.searchsorted(GRADE_BANDS, p) + 1)

thr_red = flag_threshold(s_tr_f, 0.05)
thr_amber = flag_threshold(s_tr_f, 0.16)

scored = te.copy().reset_index(drop=True)
scored["pd_12m"] = np.round(pd_cal, 4)
scored["score"] = s_te_f
scored["grade"] = scored.pd_12m.apply(grade)
# Red purely by model score (keeps precision honest); severe events force at least Amber
hard_event = (scored.legal_event_flag == 1) | (scored.news_adverse_severity > 0.6)
scored["rag"] = np.where(scored.score >= thr_red, "Red",
                 np.where((scored.score >= thr_amber) | hard_event, "Amber", "Green"))

# reason codes: top-4 positive contributors per loan
def reasons(i):
    row = contribs[i]
    idx = np.argsort(row)[::-1][:4]
    return [{"code": FEATURE_LABELS[FULL[j]], "value": round(float(row[j]), 3)}
            for j in idx if row[j] > 0][:4]

# 12-month hazard curve (prototype): shape by DPD trend / event recency
def pd_curve(p, trend, event):
    w = np.linspace(1, 12, 12)
    shape = np.exp((0.12 * trend + 0.5 * event) * (w - 6.5) / 6.5)
    h = shape / shape.sum() * p
    return np.round(np.cumsum(h), 4).tolist()

# ---- dashboard payload ----
heat = (scored.groupby(["segment", "grade"]).agg(
    n=("loan_id", "count"), exp=("exposure_inr_lakh", "sum")).reset_index())
heatmap = [{"segment": r.segment, "grade": int(r.grade), "n": int(r.n),
            "exposure": round(float(r.exp), 1)} for r in heat.itertuples()]

watch = scored[scored.rag != "Green"].sort_values("pd_12m", ascending=False).head(120)
watchlist = []
for i, r in watch.iterrows():
    watchlist.append({
        "loan_id": r.loan_id, "borrower": r.borrower, "segment": r.segment,
        "exposure": float(r.exposure_inr_lakh), "pd": float(r.pd_12m),
        "grade": int(r.grade), "rag": r.rag,
        "reasons": reasons(scored.index.get_loc(i) if not isinstance(i, int) else i),
        "curve": pd_curve(r.pd_12m, r.dpd_trend_6m,
                          max(r.news_adverse_severity, r.legal_event_flag)),
        "events": [e for e in [
            "Adverse news event detected (LLM pipeline)" if r.news_adverse_severity > 0.3 else None,
            "Legal / insolvency filing found" if r.legal_event_flag else None,
            f"{int(r.hardship_mentions_6m)} hardship mention(s) in service calls" if r.hardship_mentions_6m else None,
            "Qualified audit opinion (annual report NLP)" if r.audit_qualified_flag else None,
            f"EMI bounces in last 6m: {int(r.emi_bounce_6m)}" if r.emi_bounce_6m else None,
            f"Bureau score moved {int(r.score_migration_3m):+d} in 3m" if abs(r.score_migration_3m) > 20 else None,
        ] if e],
    })

flagged = scored.rag != "Green"
summary = {
    "loans": int(len(scored)), "exposure": round(float(scored.exposure_inr_lakh.sum()), 0),
    "red": int((scored.rag == "Red").sum()), "amber": int((scored.rag == "Amber").sum()),
    "green": int((scored.rag == "Green").sum()),
    "avg_pd": round(float(scored.pd_12m.mean()), 4), "brier": brier,
    "precision_red": round(float(scored[scored.rag == "Red"].default_12m.mean()), 3),
    "capture_red_amber": round(float(scored[flagged].default_12m.sum() / scored.default_12m.sum()), 3),
}
json.dump({"summary": summary, "metrics": results, "heatmap": heatmap,
           "watchlist": watchlist, "grade_bands": GRADE_BANDS},
          open("dashboard_data.json", "w"))
json.dump(results, open("metrics.json", "w"), indent=2)

print(json.dumps(results, indent=2))
print("summary:", summary)
