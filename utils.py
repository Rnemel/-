import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "date",
    "company_name",
    "users",
    "active_users",
    "retention_rate",
    "churn_rate",
    "revenue",
    "marketing_spend",
]

def standardize_columns(df: pd.DataFrame):
    lower_map = {c.lower(): c for c in df.columns}
    rename_map = {}
    for req in REQUIRED_COLUMNS:
        if req in lower_map:
            rename_map[lower_map[req]] = req
    df = df.rename(columns=rename_map)
    return df

def validate_and_clean(df: pd.DataFrame):
    warnings = []
    errors = []
    df = standardize_columns(df)
    original_rows = len(df)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
        return None, warnings, errors, None
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    except Exception:
        errors.append("date column is not parseable")
    if df["date"].isna().any():
        warnings.append("Some dates could not be parsed; rows with invalid dates are dropped")
        df = df.dropna(subset=["date"]).copy()
    num_cols = ["users", "active_users", "retention_rate", "churn_rate", "revenue", "marketing_spend"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[num_cols].isna().any().any():
        warnings.append("Some numeric values are invalid; rows with NaNs in required numeric fields are dropped")
        df = df.dropna(subset=num_cols).copy()
    for c in ["users", "active_users", "revenue", "marketing_spend"]:
        neg_mask = df[c] < 0
        if neg_mask.any():
            warnings.append(f"Negative {c} values clamped to 0")
            df.loc[neg_mask, c] = 0
    exceed_mask = df["active_users"] > df["users"]
    if exceed_mask.any():
        warnings.append("active_users exceeded users; clamped to users")
        df.loc[exceed_mask, "active_users"] = df.loc[exceed_mask, "users"]
    max_ret = float(df["retention_rate"].max()) if len(df) else 0.0
    max_churn = float(df["churn_rate"].max()) if len(df) else 0.0
    normalized_flags = []
    if max_ret > 1.5:
        df["retention_rate"] = df["retention_rate"] / 100.0
        normalized_flags.append("retention_rate")
    if max_churn > 1.5:
        df["churn_rate"] = df["churn_rate"] / 100.0
        normalized_flags.append("churn_rate")
    if normalized_flags:
        warnings.append(f"Normalized rates to 0–1: {', '.join(normalized_flags)}")
    df = df.sort_values("date").reset_index(drop=True)
    missing_counts = {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS}
    companies = df["company_name"].nunique()
    rows = len(df)
    min_date = df["date"].min() if rows else None
    max_date = df["date"].max() if rows else None
    months_coverage = None
    if rows:
        months_coverage = max(1, int(((max_date - min_date).days) / 30.0) + 1)
        if months_coverage < 6:
            warnings.append("Short time coverage (<6 months); confidence will be reduced")
    dropped_rows_total = max(0, original_rows - rows)
    summary = {
        "rows": rows,
        "original_rows": original_rows,
        "dropped_rows_total": dropped_rows_total,
        "companies": companies,
        "min_date": min_date,
        "max_date": max_date,
        "missing_counts": missing_counts,
        "warnings": warnings,
        "months_coverage": months_coverage,
    }
    return df, warnings, errors, summary

def filter_company_range(df: pd.DataFrame, company: str, start_date: pd.Timestamp, end_date: pd.Timestamp):
    subset = df[(df["company_name"] == company) & (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))].copy()
    subset = subset.sort_values("date").reset_index(drop=True)
    return subset

def monthly_growth_rates(series: pd.Series):
    vals = series.astype(float).values
    rates = []
    for i in range(1, len(vals)):
        prev = max(vals[i-1], 1.0)
        rate = (vals[i] - vals[i-1]) / prev
        rates.append(rate)
    return np.array(rates)

def volatility_std(rates: np.ndarray):
    if len(rates) == 0:
        return np.nan
    return float(np.nanstd(rates))

def share_profitable_months(revenue: pd.Series, spend: pd.Series):
    rev = revenue.astype(float).values
    sp = spend.astype(float).values
    if len(rev) == 0:
        return 0.0
    profitable = (rev >= sp).sum()
    return float(profitable) / float(len(rev))

def detect_spikes(rates: np.ndarray, threshold: float = 0.40):
    if len(rates) == 0:
        return 0
    return int((rates >= threshold).sum())

def is_spend_near_zero(spend: pd.Series):
    if len(spend) == 0:
        return True
    return bool((spend <= 1.0).all())

def arrow_from_trend(trend: float, up_thresh: float, down_thresh: float):
    if trend >= up_thresh:
        return "up"
    if trend <= -down_thresh:
        return "down"
    return "flat"
