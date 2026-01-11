import io
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import simpleSplit
from utils import monthly_growth_rates, volatility_std, detect_spikes, share_profitable_months, is_spend_near_zero, arrow_from_trend

def clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))

def compute_retention_score(df: pd.DataFrame) -> Dict[str, Any]:
    ret = df["retention_rate"].astype(float).values
    if len(ret) < 2:
        return {"score": None, "value": None, "trend": None, "explanation": "Not enough data"}
    avg_ret = float(np.nanmean(ret))
    trend = float(ret[-1] - ret[0])
    base = 16 if avg_ret >= 0.40 else (10 if avg_ret >= 0.25 else 4)
    adj = 4 if trend >= 0.05 else (2 if -0.05 < trend < 0.05 else 0)
    score = clamp(base + adj, 0, 20)
    months = len(ret)
    change_pct = round((trend) * 100.0, 1)
    explanation = f"Retention {'increased' if trend>0 else 'decreased' if trend<0 else 'was flat'} by {abs(change_pct)}% over {months} months, indicating {'stronger' if trend>0 else 'weaker' if trend<0 else 'stable'} long-term value."
    explanation_ar = f"الاحتفاظ {'زاد' if trend>0 else 'انخفض' if trend<0 else 'كان مستقرًا'} بمقدار {abs(change_pct)}% خلال {months} شهرًا، ما يشير إلى قيمة طويلة الأجل {'أقوى' if trend>0 else 'أضعف' if trend<0 else 'مستقرة'}."
    arrow = arrow_from_trend(trend, 0.01, 0.01)
    return {"score": score, "value": round(avg_ret, 3), "trend": arrow, "explanation": explanation, "explanation_ar": explanation_ar}

def compute_churn_score(df: pd.DataFrame) -> Dict[str, Any]:
    churn = df["churn_rate"].astype(float).values
    if len(churn) < 2:
        return {"score": None, "value": None, "trend": None, "explanation": "Not enough data"}
    avg_churn = float(np.nanmean(churn))
    trend = float(churn[-1] - churn[0])
    base = 16 if avg_churn <= 0.05 else (10 if avg_churn <= 0.10 else 4)
    adj = 4 if trend <= -0.03 else (2 if -0.03 < trend < 0.03 else 0)
    score = clamp(base + adj, 0, 20)
    explanation = f"Churn {'decreased' if trend<0 else 'increased' if trend>0 else 'was flat'} from {round(churn[0]*100,1)}% to {round(churn[-1]*100,1)}%, showing {'fewer' if trend<0 else 'more' if trend>0 else 'steady'} users leaving over time."
    explanation_ar = f"التسرّب {'انخفض' if trend<0 else 'ارتفع' if trend>0 else 'كان مستقرًا'} من {round(churn[0]*100,1)}% إلى {round(churn[-1]*100,1)}%، مما يُظهر {'مستخدمين أقل يغادرون' if trend<0 else 'مستخدمين أكثر يغادرون' if trend>0 else 'استقرارًا'} عبر الزمن."
    arrow = arrow_from_trend(-trend, 0.01, 0.01)
    return {"score": score, "value": round(avg_churn, 3), "trend": arrow, "explanation": explanation, "explanation_ar": explanation_ar}

def volatility_trend(df: pd.DataFrame) -> float:
    rates = monthly_growth_rates(df["users"]) 
    if len(rates) < 2:
        return 0.0
    split = max(2, len(rates)//2)
    early = volatility_std(rates[:split])
    recent = volatility_std(rates[-split:])
    return float(recent - early)

def compute_volatility_score(df: pd.DataFrame) -> Dict[str, Any]:
    rates = monthly_growth_rates(df["users"]) 
    if len(rates) == 0:
        return {"score": None, "value": None, "trend": None, "explanation": "Not enough data", "spikes": 0, "volatility": None}
    vol = volatility_std(rates)
    spikes = detect_spikes(rates)
    base = 18 if vol <= 0.10 else (12 if vol <= 0.25 else 6)
    penalty = min(8, 2 * spikes)
    score = clamp(base - penalty, 0, 20)
    arrow = arrow_from_trend(-volatility_trend(df), 0.02, 0.02)
    explanation = f"User growth shows {'low' if vol<=0.10 else 'moderate' if vol<=0.25 else 'high'} volatility with {spikes} spike(s), suggesting {'stable' if vol<=0.10 else 'partly stable' if vol<=0.25 else 'short-term acquisition rather than stable adoption'}."
    explanation_ar = f"نمو المستخدمين يظهر {'منخفضًا' if vol<=0.10 else 'متوسطًا' if vol<=0.25 else 'مرتفعًا'} من حيث التقلب مع {spikes} قمة، ما يشير إلى {'اعتماد مستقر' if vol<=0.10 else 'اعتماد جزئي الاستقرار' if vol<=0.25 else 'استحواذ قصير الأجل بدلًا من اعتماد مستقر'}."
    return {"score": score, "value": round(vol, 3), "trend": arrow, "explanation": explanation, "explanation_ar": explanation_ar, "spikes": spikes, "volatility": vol}

def compute_revenue_alignment_score(df: pd.DataFrame) -> Dict[str, Any]:
    users = df["users"].astype(float).values
    rev = df["revenue"].astype(float).values
    if len(users) < 2 or len(rev) < 2:
        return {"score": None, "value": None, "trend": None, "explanation": "Not enough data"}
    ug = float((users[-1] - users[0]) / max(users[0], 1.0))
    first_rev = float(rev[0])
    last_rev = float(rev[-1])
    if first_rev == 0.0 and last_rev == 0.0:
        score = 6.0
        explanation = "Revenue is zero throughout; monetization cannot be validated."
        explanation_ar = "الإيرادات صفر طوال الفترة؛ لا يمكن التحقق من جدوى الربح."
        return {"score": score, "value": None, "trend": "flat", "explanation": explanation, "explanation_ar": explanation_ar}
    if first_rev == 0.0 and last_rev > 0.0:
        score = 12.0
        explanation = "Revenue emerged from zero; partial validation of monetization with limited history."
        explanation_ar = "الإيرادات بدأت بعد أن كانت صفر؛ تحقق جزئي للربحية مع تاريخ محدود."
        return {"score": score, "value": None, "trend": "up", "explanation": explanation, "explanation_ar": explanation_ar}
    rg = float((last_rev - first_rev) / max(first_rev, 1.0))
    gap = float(abs(rg - ug))
    score = 18.0 if gap <= 0.15 else (12.0 if gap <= 0.35 else 6.0)
    score = clamp(score, 0, 20)
    expl_gap = round(gap * 100.0, 1)
    explanation = f"User growth outpaced revenue growth by {expl_gap}%" if ug > rg else f"Revenue growth outpaced user growth by {expl_gap}%"
    explanation = explanation + ", indicating growth is not translating into financial value." if gap > 0.15 else explanation + ", indicating alignment between usage and monetization."
    explanation_ar = (f"نمو المستخدمين تفوّق على نمو الإيرادات بمقدار {expl_gap}%" if ug > rg else f"نمو الإيرادات تفوّق على نمو المستخدمين بمقدار {expl_gap}%") + (", ما يشير إلى أن النمو لا يتحوّل إلى قيمة مالية." if gap > 0.15 else ", ما يشير إلى توافق بين الاستخدام والربحية.")
    arrow = "flat"
    return {"score": score, "value": round(gap, 3), "trend": arrow, "explanation": explanation, "explanation_ar": explanation_ar}

def compute_marketing_efficiency_score(df: pd.DataFrame) -> Dict[str, Any]:
    rev = df["revenue"].astype(float)
    spend = df["marketing_spend"].astype(float)
    if len(rev) == 0:
        return {"score": None, "value": None, "trend": None, "explanation": "Not enough data"}
    near_zero = is_spend_near_zero(spend)
    if near_zero:
        score = 14.0
        pct = share_profitable_months(rev, spend)
        explanation = "Marketing spend is near zero; efficiency cannot be fully assessed."
        explanation_ar = "الإنفاق التسويقي قريب من الصفر؛ لا يمكن تقييم الكفاءة بشكل كامل."
        arrow = "flat"
        return {"score": score, "value": round(pct, 3), "trend": arrow, "explanation": explanation, "explanation_ar": explanation_ar}
    pct = share_profitable_months(rev, spend)
    score = 18.0 if pct >= 0.70 else (12.0 if 0.40 <= pct <= 0.69 else 6.0)
    score = clamp(score, 0, 20)
    explanation = f"Revenue exceeded marketing spend in {round(pct*100,1)}% of months, indicating {'efficient' if pct>=0.70 else 'partly efficient' if pct>=0.40 else 'inefficient'} acquisition."
    explanation_ar = f"تجاوزت الإيرادات الإنفاق التسويقي في {round(pct*100,1)}% من الأشهر، ما يدل على {'كفاءة' if pct>=0.70 else 'كفاءة جزئية' if pct>=0.40 else 'عدم كفاءة'} في الاستحواذ."
    arrow = "flat"
    return {"score": score, "value": round(pct, 3), "trend": arrow, "explanation": explanation, "explanation_ar": explanation_ar}

def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    r = compute_retention_score(df)
    c = compute_churn_score(df)
    v = compute_volatility_score(df)
    rr = compute_revenue_alignment_score(df)
    me = compute_marketing_efficiency_score(df)
    scores = {"Retention Quality": r["score"], "Churn Risk": c["score"], "Growth Volatility": v["score"], "Revenue Reality Check": rr["score"], "Marketing Efficiency": me["score"]}
    values = {"Retention Quality": r["value"], "Churn Risk": c["value"], "Growth Volatility": v.get("volatility"), "Revenue Reality Check": rr["value"], "Marketing Efficiency": me["value"]}
    trends = {"Retention Quality": r["trend"], "Churn Risk": c["trend"], "Growth Volatility": v["trend"], "Revenue Reality Check": rr["trend"], "Marketing Efficiency": me["trend"]}
    explanations = {"Retention Quality": r["explanation"], "Churn Risk": c["explanation"], "Growth Volatility": v["explanation"], "Revenue Reality Check": rr["explanation"], "Marketing Efficiency": me["explanation"]}
    explanations_ar = {"Retention Quality": r.get("explanation_ar"), "Churn Risk": c.get("explanation_ar"), "Growth Volatility": v.get("explanation_ar"), "Revenue Reality Check": rr.get("explanation_ar"), "Marketing Efficiency": me.get("explanation_ar")}
    return {"scores": scores, "values": values, "trends": trends, "explanations": explanations, "explanations_ar": explanations_ar, "spikes": v.get("spikes", 0), "volatility": v.get("volatility", None)}

def total_score_and_classification(scores: Dict[str, float]) -> Dict[str, Any]:
    total = 0.0
    missing = False
    for k, v in scores.items():
        if v is None:
            missing = True
        else:
            total += float(v)
    classification = None
    if not missing:
        classification = "Sustainable Success" if total >= 65.0 else "Success Bubble"
    return {"total": int(round(total)), "classification": classification}

def data_outliers_count(df: pd.DataFrame) -> int:
    rates = monthly_growth_rates(df["users"])
    if len(rates) == 0:
        return 0
    mean = float(np.nanmean(rates))
    std = float(np.nanstd(rates))
    if std == 0.0:
        return 0
    z = np.abs((rates - mean) / std)
    return int((z >= 3.0).sum())

def compute_confidence(df: pd.DataFrame, clean_summary: Dict[str, Any], volatility: float, outliers: int) -> Dict[str, Any]:
    months = clean_summary.get("months_coverage") or 0
    time_factor = "good" if months >= 9 else ("moderate" if 6 <= months <= 8 else "low")
    original = clean_summary.get("original_rows") or 0
    dropped = clean_summary.get("dropped_rows_total") or 0
    miss_ratio = float(dropped) / float(original) if original > 0 else 0.0
    missing_factor = "good" if miss_ratio <= 0.05 else ("moderate" if miss_ratio <= 0.15 else "low")
    stability_factor = "good" if (volatility is not None and volatility <= 0.25 and outliers <= 1) else ("moderate" if (volatility is not None and volatility <= 0.25) or outliers <= 1 else "low")
    factors = [time_factor, missing_factor, stability_factor]
    level = "High" if all(f == "good" for f in factors) else ("Medium" if ("low" not in factors and ("moderate" in factors)) else "Low")
    reason_parts = []
    if time_factor != "good":
        reason_parts.append(f"{months} months of data")
    else:
        reason_parts.append(f"{months} months of data")
    if missing_factor != "good":
        reason_parts.append(f"{int(round(miss_ratio*100))}% missing or dropped")
    else:
        reason_parts.append(f"{int(round(miss_ratio*100))}% missing or dropped")
    if stability_factor != "good":
        reason_parts.append("unstable volatility")
    else:
        reason_parts.append("stable volatility")
    reason = f"{level} confidence: " + ", ".join(reason_parts) + "."
    ar_vol = "تقلب مستقر" if stability_factor == "good" else "تقلب غير مستقر" if stability_factor == "low" else "تقلب متوسط"
    reason_ar = f"{ 'ثقة عالية' if level=='High' else 'ثقة متوسطة' if level=='Medium' else 'ثقة منخفضة' }: {months} شهر من البيانات، {int(round(miss_ratio*100))}% مفقود أو مُزال، {ar_vol}."
    return {"level": level, "reason": reason, "reason_ar": reason_ar}

def key_reasons(scores: Dict[str, float], explanations: Dict[str, str]) -> Dict[str, List[str]]:
    items = [(k, v) for k, v in scores.items() if v is not None]
    items_sorted = sorted(items, key=lambda x: x[1])
    negatives = [f"{items_sorted[i][0]}: {explanations[items_sorted[i][0]]}" for i in range(min(3, len(items_sorted)))]
    positives = [f"{items_sorted[-(i+1)][0]}: {explanations[items_sorted[-(i+1)][0]]}" for i in range(min(3, len(items_sorted)))]
    return {"negatives": negatives, "positives": positives}

def recommendations(scores: Dict[str, float]) -> List[str]:
    sorted_inds = sorted([(k, v) for k, v in scores.items() if v is not None], key=lambda x: x[1])
    recs = []
    names = [s[0] for s in sorted_inds[:5]]
    for name in names:
        if name == "Retention Quality":
            recs.append("Retention: strengthen onboarding, product stickiness, and long-term value features.")
        elif name == "Churn Risk":
            recs.append("Churn: investigate drivers across UX, pricing, and support; run cohort analysis.")
        elif name == "Growth Volatility":
            recs.append("Volatility: separate paid vs organic growth and monitor post-campaign retention.")
        elif name == "Revenue Reality Check":
            recs.append("Revenue alignment: refine monetization strategy and conversion funnel.")
        elif name == "Marketing Efficiency":
            recs.append("Marketing efficiency: optimize acquisition spend and emphasize higher-LTV channels.")
    if len(recs) < 3:
        recs = recs + ["Data coverage: extend time range to strengthen confidence."]
    return recs[:5]

def summary_text(classification: str, reasons: Dict[str, List[str]]) -> str:
    if classification == "Sustainable Success":
        return "The indicators collectively support sustainable performance with aligned retention, churn, and monetization, while operational variance remains acceptable."
    return "Indicators suggest a success bubble driven by volatility or weak retention/revenue alignment; focus on stabilizing core value and monetization."

def build_scores_dataframe(company: str, start: pd.Timestamp, end: pd.Timestamp, scores: Dict[str, float]) -> pd.DataFrame:
    data = {"company_name": [company], "start_date": [pd.to_datetime(start)], "end_date": [pd.to_datetime(end)]}
    for k, v in scores.items():
        data[k.replace(" ", "_").lower()] = [v]
    data["total_score"] = [int(round(sum([s for s in scores.values() if s is not None])))]
    return pd.DataFrame(data)

def make_pdf(company: str, period: str, classification: str, total: int, confidence: str, key_pos: List[str], key_neg: List[str], scores: Dict[str, float], recs: List[str], limitations: List[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    y = h - 0.75*inch
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.75*inch, y, "Success Bubble Detector — Summary")
    y -= 0.35*inch
    c.setFont("Helvetica", 11)
    c.drawString(0.75*inch, y, f"Company: {company}")
    y -= 0.25*inch
    c.drawString(0.75*inch, y, f"Period: {period}")
    y -= 0.25*inch
    c.drawString(0.75*inch, y, f"Classification: {classification}")
    y -= 0.25*inch
    c.drawString(0.75*inch, y, f"Total Score: {total}")
    y -= 0.25*inch
    c.drawString(0.75*inch, y, f"Confidence: {confidence}")
    y -= 0.35*inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.75*inch, y, "Score Breakdown")
    y -= 0.25*inch
    c.setFont("Helvetica", 10)
    for k in ["Retention Quality", "Churn Risk", "Growth Volatility", "Revenue Reality Check", "Marketing Efficiency"]:
        c.drawString(0.85*inch, y, f"{k}: {int(round(scores.get(k) or 0))}")
        y -= 0.18*inch
    y -= 0.20*inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.75*inch, y, "Key Reasons")
    y -= 0.22*inch
    c.setFont("Helvetica", 10)
    for s in key_pos:
        for line in simpleSplit(s, "Helvetica", 10, w - 1.5*inch):
            c.drawString(0.85*inch, y, f"+ {line}")
            y -= 0.16*inch
    for s in key_neg:
        for line in simpleSplit(s, "Helvetica", 10, w - 1.5*inch):
            c.drawString(0.85*inch, y, f"- {line}")
            y -= 0.16*inch
    y -= 0.18*inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.75*inch, y, "Recommendations")
    y -= 0.22*inch
    c.setFont("Helvetica", 10)
    for s in recs:
        for line in simpleSplit(s, "Helvetica", 10, w - 1.5*inch):
            c.drawString(0.85*inch, y, f"• {line}")
            y -= 0.16*inch
    y -= 0.18*inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.75*inch, y, "System Limitations")
    y -= 0.22*inch
    c.setFont("Helvetica", 9)
    for s in limitations:
        for line in simpleSplit(s, "Helvetica", 9, w - 1.5*inch):
            c.drawString(0.85*inch, y, f"- {line}")
            y -= 0.14*inch
    c.showPage()
    c.save()
    return buf.getvalue()

# --- Forecasting ---
def _linear_forecast(series: np.ndarray, periods: int, lo: float = None, hi: float = None) -> np.ndarray:
    n = len(series)
    if n < 2:
        last = series[-1] if n == 1 else 0.0
        return np.array([last for _ in range(periods)], dtype=float)
    x = np.arange(n, dtype=float)
    y = series.astype(float)
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs[0], coeffs[1]
    x_future = np.arange(n, n + periods, dtype=float)
    y_pred = slope * x_future + intercept
    if lo is not None or hi is not None:
        lo_v = -np.inf if lo is None else lo
        hi_v = np.inf if hi is None else hi
        y_pred = np.clip(y_pred, lo_v, hi_v)
    return y_pred

def forecast_series(series: pd.Series, periods: int, bounds: Dict[str, float] = None) -> np.ndarray:
    b_lo = None
    b_hi = None
    if bounds:
        b_lo = bounds.get("lo")
        b_hi = bounds.get("hi")
    return _linear_forecast(series.astype(float).values, periods, b_lo, b_hi)

def forecast_df(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    last_date = pd.to_datetime(df["date"].max())
    future_dates = pd.date_range(start=(last_date + pd.offsets.MonthBegin(1)), periods=periods, freq="MS")
    users_pred = forecast_series(df["users"], periods, bounds={"lo": 0.0})
    active_pred = forecast_series(df["active_users"], periods, bounds={"lo": 0.0})
    revenue_pred = forecast_series(df["revenue"], periods, bounds={"lo": 0.0})
    spend_pred = forecast_series(df["marketing_spend"], periods, bounds={"lo": 0.0})
    retention_pred = forecast_series(df["retention_rate"], periods, bounds={"lo": 0.0, "hi": 1.0})
    churn_pred = forecast_series(df["churn_rate"], periods, bounds={"lo": 0.0, "hi": 1.0})
    out = pd.DataFrame({
        "date": future_dates,
        "users": users_pred,
        "active_users": active_pred,
        "revenue": revenue_pred,
        "marketing_spend": spend_pred,
        "retention_rate": retention_pred,
        "churn_rate": churn_pred,
    })
    return out

def holt_linear_forecast(series: pd.Series, periods: int, alpha: float = 0.5, beta: float = 0.3, bounds: Dict[str, float] = None) -> np.ndarray:
    vals = series.astype(float).values
    n = len(vals)
    if n == 0:
        return np.zeros(periods, dtype=float)
    if n == 1:
        base = float(vals[0])
        out = np.array([base for _ in range(periods)], dtype=float)
        if bounds:
            lo = bounds.get("lo")
            hi = bounds.get("hi")
            lo_v = -np.inf if lo is None else lo
            hi_v = np.inf if hi is None else hi
            out = np.clip(out, lo_v, hi_v)
        return out
    l = float(vals[0])
    b = float(vals[1] - vals[0])
    for i in range(1, n):
        prev_l = l
        l = alpha * vals[i] + (1 - alpha) * (l + b)
        b = beta * (l - prev_l) + (1 - beta) * b
    out = np.array([l + b * (k + 1) for k in range(periods)], dtype=float)
    if bounds:
        lo = bounds.get("lo")
        hi = bounds.get("hi")
        lo_v = -np.inf if lo is None else lo
        hi_v = np.inf if hi is None else hi
        out = np.clip(out, lo_v, hi_v)
    return out

def seasonal_mean_forecast(series: pd.Series, periods: int, season: int = 12, bounds: Dict[str, float] = None) -> np.ndarray:
    vals = series.astype(float).values
    n = len(vals)
    if n < season:
        return forecast_series(series, periods, bounds)
    pattern = np.zeros(season, dtype=float)
    counts = np.zeros(season, dtype=int)
    for i in range(n):
        idx = i % season
        pattern[idx] += vals[i]
        counts[idx] += 1
    counts[counts == 0] = 1
    pattern = pattern / counts
    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, vals, 1)
    slope = float(coeffs[0])
    out = []
    for k in range(periods):
        base = pattern[(n + k) % season]
        trend = slope * (k + 1)
        out.append(base + trend)
    out = np.array(out, dtype=float)
    if bounds:
        lo = bounds.get("lo")
        hi = bounds.get("hi")
        lo_v = -np.inf if lo is None else lo
        hi_v = np.inf if hi is None else hi
        out = np.clip(out, lo_v, hi_v)
    return out

def forecast_with_confidence(series: pd.Series, periods: int, method: str, bounds: Dict[str, float] = None, method_params: Dict[str, float] = None) -> Dict[str, np.ndarray]:
    vals = series.astype(float).values
    n = len(vals)
    if method == "holt":
        alpha = 0.5
        beta = 0.3
        if method_params:
            alpha = float(method_params.get("alpha", alpha))
            beta = float(method_params.get("beta", beta))
        pred = holt_linear_forecast(series, periods, alpha=alpha, beta=beta, bounds=bounds)
        l = float(vals[0]) if n else 0.0
        b = float(vals[1] - vals[0]) if n >= 2 else 0.0
        alpha = 0.5
        beta = 0.3
        fitted = []
        for i in range(n):
            fitted.append(l + b)
            prev_l = l
            l = alpha * vals[i] + (1 - alpha) * (l + b)
            b = beta * (l - prev_l) + (1 - beta) * b
        resid = vals - np.array(fitted[:n], dtype=float)
    elif method == "seasonal":
        season_len = 12
        if method_params:
            season_len = int(method_params.get("season", season_len))
        pred = seasonal_mean_forecast(series, periods, season=season_len, bounds=bounds)
        season = 12
        fitted = []
        if n >= season:
            pattern = np.zeros(season, dtype=float)
            counts = np.zeros(season, dtype=int)
            for i in range(n):
                idx = i % season
                pattern[idx] += vals[i]
                counts[idx] += 1
            counts[counts == 0] = 1
            pattern = pattern / counts
            x = np.arange(n, dtype=float)
            coeffs = np.polyfit(x, vals, 1)
            slope = float(coeffs[0])
            for i in range(n):
                fitted.append(pattern[i % season] + slope * i)
        else:
            x = np.arange(n, dtype=float)
            coeffs = np.polyfit(x, vals, 1) if n >= 2 else np.array([0.0, vals[0] if n else 0.0], dtype=float)
            fitted = coeffs[0] * x + coeffs[1]
        resid = vals - np.array(fitted[:n], dtype=float)
    else:
        pred = forecast_series(series, periods, bounds=bounds)
        x = np.arange(n, dtype=float)
        coeffs = np.polyfit(x, vals, 1) if n >= 2 else np.array([0.0, vals[0] if n else 0.0], dtype=float)
        fitted = coeffs[0] * x + coeffs[1]
        resid = vals - fitted
    sigma = float(np.nanstd(resid)) if n > 1 else 0.0
    lo = pred - 1.96 * sigma
    hi = pred + 1.96 * sigma
    if bounds:
        lo_b = bounds.get("lo")
        hi_b = bounds.get("hi")
        lo_v = -np.inf if lo_b is None else lo_b
        hi_v = np.inf if hi_b is None else hi_b
        lo = np.clip(lo, lo_v, hi_v)
        hi = np.clip(hi, lo_v, hi_v)
    return {"pred": pred, "lo": lo, "hi": hi}

def forecast_df_confidence(df: pd.DataFrame, periods: int, method: str, method_params: Dict[str, float] = None) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    last_date = pd.to_datetime(df["date"].max())
    future_dates = pd.date_range(start=(last_date + pd.offsets.MonthBegin(1)), periods=periods, freq="MS")
    u = forecast_with_confidence(df["users"], periods, method, bounds={"lo": 0.0}, method_params=method_params)
    r = forecast_with_confidence(df["revenue"], periods, method, bounds={"lo": 0.0}, method_params=method_params)
    ret = forecast_with_confidence(df["retention_rate"], periods, method, bounds={"lo": 0.0, "hi": 1.0}, method_params=method_params)
    out = pd.DataFrame({
        "date": future_dates,
        "users": u["pred"],
        "users_lo": u["lo"],
        "users_hi": u["hi"],
        "revenue": r["pred"],
        "revenue_lo": r["lo"],
        "revenue_hi": r["hi"],
        "retention_rate": ret["pred"],
        "retention_lo": ret["lo"],
        "retention_hi": ret["hi"],
    })
    return out

def recommend_forecast_method(df: pd.DataFrame) -> str:
    n = len(df)
    if n < 12:
        return "linear"
    x = np.arange(n, dtype=float)
    users = df["users"].astype(float).values
    slope = float(np.polyfit(x, users, 1)[0]) if n >= 2 else 0.0
    months = pd.to_datetime(df["date"]).dt.month.values
    pm = np.zeros(12, dtype=float)
    cnt = np.zeros(12, dtype=int)
    for i in range(n):
        idx = int(months[i]) - 1
        pm[idx] += users[i]
        cnt[idx] += 1
    cnt[cnt == 0] = 1
    pm = pm / cnt
    overall_std = float(np.nanstd(users))
    seasonal_std = float(np.nanstd(pm))
    ratio = seasonal_std / overall_std if overall_std > 0 else 0.0
    if ratio >= 0.20:
        return "seasonal"
    if abs(slope) > 0.0:
        return "holt"
    return "linear"

def evaluate_forecast_series(series: pd.Series, method: str, holdout: int = 6, method_params: Dict[str, float] = None) -> Dict[str, float]:
    vals = series.astype(float).values
    n = len(vals)
    if n <= holdout:
        return {"mae": None, "mape": None}
    train = series.iloc[:n-holdout]
    actual = vals[n-holdout:]
    bounds = {"lo": 0.0}
    if series.name in ["retention_rate", "churn_rate"]:
        bounds = {"lo": 0.0, "hi": 1.0}
    pred = forecast_with_confidence(train, holdout, method, bounds=bounds, method_params=method_params)["pred"]
    mae = float(np.nanmean(np.abs(actual - pred)))
    denom = np.where(actual == 0, np.nan, actual)
    mape = float(np.nanmean(np.abs((actual - pred) / denom)))
    return {"mae": mae, "mape": mape}

def evaluate_forecast_df(df: pd.DataFrame, method: str, holdout: int = 6, method_params: Dict[str, float] = None) -> Dict[str, Dict[str, float]]:
    res_users = evaluate_forecast_series(df["users"], method, holdout, method_params)
    res_rev = evaluate_forecast_series(df["revenue"], method, holdout, method_params)
    res_ret = evaluate_forecast_series(df["retention_rate"], method, holdout, method_params)
    return {"users": res_users, "revenue": res_rev, "retention": res_ret}

def tune_holt_params(series: pd.Series, holdout: int = 6) -> Dict[str, float]:
    candidates_a = [0.2, 0.5, 0.8]
    candidates_b = [0.1, 0.3, 0.5]
    best = {"alpha": 0.5, "beta": 0.3, "mae": float("inf")}
    for a in candidates_a:
        for b in candidates_b:
            res = evaluate_forecast_series(series, "holt", holdout, {"alpha": a, "beta": b})
            mae = res["mae"] if res["mae"] is not None else float("inf")
            if mae < best["mae"]:
                best = {"alpha": a, "beta": b, "mae": mae}
    return {"alpha": best["alpha"], "beta": best["beta"]}
