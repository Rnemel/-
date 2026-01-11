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
    arrow = arrow_from_trend(trend, 0.01, 0.01)
    return {"score": score, "value": round(avg_ret, 3), "trend": arrow, "explanation": explanation}

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
    arrow = arrow_from_trend(-trend, 0.01, 0.01)
    return {"score": score, "value": round(avg_churn, 3), "trend": arrow, "explanation": explanation}

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
    return {"score": score, "value": round(vol, 3), "trend": arrow, "explanation": explanation, "spikes": spikes, "volatility": vol}

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
        return {"score": score, "value": None, "trend": "flat", "explanation": explanation}
    if first_rev == 0.0 and last_rev > 0.0:
        score = 12.0
        explanation = "Revenue emerged from zero; partial validation of monetization with limited history."
        return {"score": score, "value": None, "trend": "up", "explanation": explanation}
    rg = float((last_rev - first_rev) / max(first_rev, 1.0))
    gap = float(abs(rg - ug))
    score = 18.0 if gap <= 0.15 else (12.0 if gap <= 0.35 else 6.0)
    score = clamp(score, 0, 20)
    expl_gap = round(gap * 100.0, 1)
    explanation = f"User growth outpaced revenue growth by {expl_gap}%" if ug > rg else f"Revenue growth outpaced user growth by {expl_gap}%"
    explanation = explanation + ", indicating growth is not translating into financial value." if gap > 0.15 else explanation + ", indicating alignment between usage and monetization."
    arrow = "flat"
    return {"score": score, "value": round(gap, 3), "trend": arrow, "explanation": explanation}

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
        arrow = "flat"
        return {"score": score, "value": round(pct, 3), "trend": arrow, "explanation": explanation}
    pct = share_profitable_months(rev, spend)
    score = 18.0 if pct >= 0.70 else (12.0 if 0.40 <= pct <= 0.69 else 6.0)
    score = clamp(score, 0, 20)
    explanation = f"Revenue exceeded marketing spend in {round(pct*100,1)}% of months, indicating {'efficient' if pct>=0.70 else 'partly efficient' if pct>=0.40 else 'inefficient'} acquisition."
    arrow = "flat"
    return {"score": score, "value": round(pct, 3), "trend": arrow, "explanation": explanation}

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
    return {"scores": scores, "values": values, "trends": trends, "explanations": explanations, "spikes": v.get("spikes", 0), "volatility": v.get("volatility", None)}

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
    return {"level": level, "reason": reason}

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

