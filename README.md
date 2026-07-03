# Loan Default Early Warning System — Prototype

End-to-end prototype of a predictive EWS that estimates **12-month probability of default**
from structured + unstructured signals and presents it through a **common interpretation
framework** (calibrated PD → Unified Risk Grade 1–10 → RAG status → SHAP reason codes).

Built for the IDBI Innovate 2026 submission. Synthetic data — no real borrower information.

## What it demonstrates

| Component | Where |
|---|---|
| Synthetic multi-segment loan book (12,000 loans, 5 segments) | `generate_data.py` |
| Structured features (DPD trend, utilisation, cash-flow coverage, bureau migration, GST gaps…) | `generate_data.py` |
| Unstructured-signal features as produced by NLP/LLM pipelines (adverse news severity, legal filings, call-sentiment drift, hardship mentions, qualified audit opinions) | `generate_data.py` |
| Legacy-scorecard baseline vs structured-only XGBoost vs full (structured+text) XGBoost, out-of-time validated | `train_model.py` |
| Platt calibration, Unified Risk Grade, RAG assignment, TreeSHAP reason codes, 12-month PD curve | `train_model.py` |
| Interactive dashboard: portfolio heatmap, early-warning watchlist, loan drill-down, benchmark view | `dashboard.html` |

## Results (out-of-time test, same 5% flag budget for every model)

| Model | AUC | Precision on Red flags | 
|---|---|---|
| Legacy scorecard (bureau + DPD) | 0.60 | ~15% ← today's 16–22% problem |
| EWS Phase 1 — structured only | 0.86 | ~48% |
| **EWS Phase 2 — structured + text** | **0.97** | **~91%** |

Red + Amber together capture ~96% of defaults 12 months ahead.
(Numbers vary slightly per data regeneration; synthetic data is tuned to mirror the lift
pattern reported in published credit-risk benchmarks, not to prove production performance.)

## Run it

```bash
pip install numpy pandas scikit-learn xgboost
python generate_data.py     # -> portfolio.csv
python train_model.py      # -> metrics.json, dashboard_data.json
python build_dashboard.py  # -> dashboard.html (self-contained)
```

Open `dashboard.html` in any browser — no server needed.

## Design notes

- **Out-of-time validation:** trained on older vintages, evaluated on later ones — no leakage.
- **Precision-first operating point:** Red = top 5% of scores; severe text events (insolvency filing, high-severity adverse news) force at least Amber.
- **Reason codes:** exact TreeSHAP contributions from XGBoost (`pred_contribs`), mapped to a business taxonomy.
- **Scope:** operational early warning only — not a regulatory capital (IRB/Basel) model.

## Repo layout

```
generate_data.py         synthetic portfolio generator
train_model.py           training, calibration, scoring, reason codes
build_dashboard.py       embeds scored data into the dashboard
dashboard_template.html  dashboard UI (Chart.js)
dashboard.html           generated, self-contained demo
portfolio.csv            generated data
metrics.json             benchmark table
dashboard_data.json      scored payload
```
