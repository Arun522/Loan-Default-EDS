"""Synthetic loan portfolio generator for the EWS prototype.

Creates a realistic multi-segment loan book with structured features,
unstructured-signal features (as produced by NLP/LLM pipelines), and a
12-month forward default label whose drivers mirror real credit dynamics.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 12000

SEGMENTS = {
    "Retail Unsecured": 0.34,
    "Retail Secured":   0.26,
    "SME":              0.22,
    "Mid Corporate":    0.12,
    "Microfinance":     0.06,
}
BASE_HAZARD = {  # segment-level 12m default base rates
    "Retail Unsecured": 0.045, "Retail Secured": 0.020, "SME": 0.055,
    "Mid Corporate": 0.030, "Microfinance": 0.060,
}

def make_portfolio(n=N):
    seg = RNG.choice(list(SEGMENTS), p=list(SEGMENTS.values()), size=n)
    df = pd.DataFrame({"segment": seg})
    df["loan_id"] = [f"LN{100000+i}" for i in range(n)]
    df["borrower"] = [f"Borrower-{i:05d}" for i in range(n)]
    df["exposure_inr_lakh"] = np.round(np.exp(RNG.normal(3.2, 1.1, n)), 1)

    # --- structured features ---
    df["bureau_score"] = np.clip(RNG.normal(715, 55, n), 350, 900).round()
    df["score_migration_3m"] = RNG.normal(0, 18, n).round()          # velocity, not level
    df["dpd_current"] = RNG.choice([0, 0, 0, 0, 5, 15, 35], size=n)
    df["dpd_trend_6m"] = np.clip(RNG.normal(0, 1.0, n), -3, 4).round(2)
    df["emi_bounce_6m"] = RNG.poisson(0.25, n)
    df["util_ratio"] = np.clip(RNG.beta(2.2, 3.2, n) + RNG.normal(0, .05, n), 0, 1).round(3)
    df["util_spike_3m"] = np.clip(RNG.normal(0, 0.12, n), -0.4, 0.6).round(3)
    df["cashflow_coverage"] = np.clip(RNG.normal(1.6, 0.55, n), 0.1, 4).round(2)
    df["salary_regularity"] = np.clip(RNG.beta(6, 1.4, n), 0, 1).round(3)
    df["balance_volatility"] = np.clip(RNG.gamma(2.0, 0.16, n), 0, 2).round(3)
    df["new_enquiries_3m"] = RNG.poisson(0.8, n)
    df["leverage_buildup"] = np.clip(RNG.normal(0, 0.25, n), -0.6, 1.2).round(3)
    df["ltv_drift"] = np.where(df.segment == "Retail Secured",
                               np.clip(RNG.normal(0, 0.08, n), -.2, .35), 0).round(3)
    df["gst_filing_gap_m"] = np.where(df.segment.isin(["SME", "Mid Corporate"]),
                                      RNG.poisson(0.5, n), 0)
    df["interest_coverage"] = np.where(df.segment.isin(["SME", "Mid Corporate"]),
                                       np.clip(RNG.normal(3.0, 1.6, n), 0.2, 9), np.nan).round(2)

    # --- unstructured-signal features (outputs of NLP/LLM pipelines) ---
    df["news_adverse_severity"] = np.where(RNG.random(n) < 0.07,
                                           RNG.beta(2, 2, n), 0).round(3)   # 0-1
    df["legal_event_flag"] = (RNG.random(n) < 0.035).astype(int)            # suits, cheque bounce, insolvency
    df["call_sentiment_drift"] = np.clip(RNG.normal(0, 0.30, n), -1, 1).round(3)  # negative = deteriorating
    df["hardship_mentions_6m"] = RNG.poisson(0.15, n)
    df["audit_qualified_flag"] = np.where(df.segment.isin(["SME", "Mid Corporate"]),
                                          (RNG.random(n) < 0.05).astype(int), 0)
    df["web_footprint_decline"] = np.where(df.segment.isin(["SME", "Mid Corporate", "Microfinance"]),
                                           np.clip(RNG.normal(0, .2, n), 0, 1), 0).round(3)

    # --- latent 12m default hazard ---
    z = (
        -(df.bureau_score - 715) / 5500
        - df.score_migration_3m / 300
        + df.dpd_current / 260
        + df.dpd_trend_6m / 11
        + df.emi_bounce_6m / 9
        + (df.util_ratio - 0.4) / 6.5 + df.util_spike_3m / 4.5
        - (df.cashflow_coverage - 1.6) / 9
        - (df.salary_regularity - 0.8) / 6
        + df.balance_volatility / 8
        + df.new_enquiries_3m / 32
        + df.leverage_buildup / 5.5
        + df.ltv_drift / 3.2
        + df.gst_filing_gap_m / 22
        + df.news_adverse_severity / 2.6
        + df.legal_event_flag / 3.4
        - df.call_sentiment_drift / 6.0
        + df.hardship_mentions_6m / 10
        + df.audit_qualified_flag / 4.5
        + df.web_footprint_decline / 5.0
        + RNG.normal(0, 0.02, n)                      # idiosyncratic noise
    )
    base = df.segment.map(BASE_HAZARD).astype(float)
    logit0 = np.log(base / (1 - base))
    z_std = (z - z.mean()) / z.std()
    pd12 = 1 / (1 + np.exp(-(logit0 - 5.5 + z_std * 5.0)))
    df["default_12m"] = (RNG.random(n) < pd12).astype(int)
    df["true_pd"] = pd12.round(4)

    # vintage split: first 70% train (older), last 30% out-of-time test
    df["vintage"] = np.where(np.arange(n) < int(n * 0.7), "train", "oot")
    return df

if __name__ == "__main__":
    df = make_portfolio()
    df.to_csv("portfolio.csv", index=False)
    print(f"portfolio.csv: {len(df)} loans, default rate {df.default_12m.mean():.2%}")
    print(df.groupby('segment').default_12m.mean().round(3))
