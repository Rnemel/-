import pandas as pd
import numpy as np

def get_sample_data():
    dates = pd.date_range(start="2025-01-01", periods=12, freq="MS")
    alpha_users = np.linspace(1000, 2200, 12).round().astype(int)
    alpha_active = (alpha_users * np.linspace(0.62, 0.70, 12)).round().astype(int)
    alpha_ret = np.linspace(0.45, 0.50, 12)
    alpha_churn = np.linspace(0.08, 0.06, 12)
    alpha_rev = np.linspace(20000, 42000, 12).round()
    alpha_marketing = np.linspace(8000, 12000, 12).round()
    hype_users = np.array([800, 820, 1000, 1800, 2600, 2800, 2200, 2400, 3200, 3500, 3000, 3100])
    hype_active = (hype_users * np.linspace(0.40, 0.55, 12)).round().astype(int)
    hype_ret = np.linspace(0.22, 0.18, 12)
    hype_churn = np.linspace(0.15, 0.18, 12)
    hype_rev = np.array([5000, 5200, 6000, 7000, 9000, 9500, 8000, 8200, 10000, 11000, 10500, 10800])
    hype_marketing = np.array([4000, 4500, 6000, 12000, 16000, 18000, 15000, 14000, 17000, 19000, 16000, 15000])
    alpha = pd.DataFrame({
        "date": dates,
        "company_name": ["AlphaSoft"] * 12,
        "users": alpha_users,
        "active_users": alpha_active,
        "retention_rate": alpha_ret,
        "churn_rate": alpha_churn,
        "revenue": alpha_rev,
        "marketing_spend": alpha_marketing,
    })
    hype = pd.DataFrame({
        "date": dates,
        "company_name": ["HypeCorp"] * 12,
        "users": hype_users,
        "active_users": hype_active,
        "retention_rate": hype_ret,
        "churn_rate": hype_churn,
        "revenue": hype_rev,
        "marketing_spend": hype_marketing,
    })
    df = pd.concat([alpha, hype], ignore_index=True)
    return df

