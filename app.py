import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from utils import validate_and_clean, filter_company_range, REQUIRED_COLUMNS, monthly_growth_rates
from sample_data import get_sample_data
from analysis_engine import compute_indicators, total_score_and_classification, compute_confidence, key_reasons, recommendations, build_scores_dataframe, make_pdf, data_outliers_count, summary_text, forecast_df, forecast_df_confidence, recommend_forecast_method, evaluate_forecast_df, tune_holt_params

st.set_page_config(page_title="Success Bubble Detector", layout="wide")

@st.cache_data(show_spinner=False)
def cached_compute_indicators(df: pd.DataFrame):
    return compute_indicators(df)

@st.cache_data(show_spinner=False)
def cached_growth_rates(series: pd.Series):
    return monthly_growth_rates(series)

def _lock_fig(fig):
    fig.update_layout(legend_itemclick=False, legend_itemdoubleclick=False)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig

plotly_config = {"displayModeBar": False, "scrollZoom": False}

def _hex_to_rgba(h, a):
    h = h.lstrip('#')
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"

@st.cache_data(show_spinner=False)
def parse_events_csv(file_bytes: bytes):
    import io
    ev = pd.read_csv(io.BytesIO(file_bytes))
    if "date" in ev.columns:
        ev["date"] = pd.to_datetime(ev["date"]) 
    return ev

if "data" not in st.session_state:
    st.session_state["data"] = None
if "clean_summary" not in st.session_state:
    st.session_state["clean_summary"] = None
if "company" not in st.session_state:
    st.session_state["company"] = None
if "date_range" not in st.session_state:
    st.session_state["date_range"] = None

st.sidebar.title("Success Bubble Detector")
lang = st.sidebar.selectbox("Language", ["English", "العربية"], index=0)
T = {
    "English": {
        "data_source": "Data Source",
        "upload_csv": "Upload CSV",
        "load_sample": "Load sample data",
        "setup": "Data Upload & Setup",
        "preview": "Dataset Preview",
        "rows": "Rows",
        "companies": "Companies",
        "start": "Start",
        "end": "End",
        "company": "Company",
        "date_range": "Date range",
        "metrics_tab": "Sustainability Metrics",
        "decision_tab": "Scoring & Decision",
        "dashboard_tab": "Visual Dashboard",
        "export_tab": "Recommendations & Export",
        "dataset_loaded": "Dataset loaded",
        "sample_loaded": "Sample data loaded",
        "need_data": "Upload a CSV or load sample data to begin.",
        "not_enough": "Not enough data in selected range.",
        "decision": "Decision",
        "classification": "Classification",
        "score": "Sustainability Score",
        "confidence": "Confidence",
        "key_reasons": "Key Reasons",
        "positives": "Positives",
        "negatives": "Negatives",
        "recommendations": "Recommendations",
        "export": "Export",
        "download_csv": "Download CSV",
        "download_pdf": "Download PDF",
        "download_json": "Download JSON",
        "limitations": "System Limitations",
        "theme": "Theme",
        "dark": "Dark",
        "light": "Light",
        "forecast_tab": "Forecast",
        "forecast_horizon": "Forecast horizon (months)",
        "forecast_note": "Indicative trend-only forecast; not predictive or guaranteed.",
        "users_vs_active": "Users vs Active Users",
        "retention_rate": "Retention Rate",
        "churn_rate": "Churn Rate",
        "rev_vs_marketing": "Revenue vs Marketing Spend",
        "ind_retention": "Retention Quality",
        "ind_churn": "Churn Risk",
        "ind_vol": "Growth Volatility",
        "ind_rev": "Revenue Reality Check",
        "ind_marketing": "Marketing Efficiency",
        "spikes": "Growth Spikes",
        "dash_options": "Dashboard Options",
        "smoothing_window": "Smoothing window (months)",
        "show_trend": "Show trend line",
        "segment_marketing": "Segment by marketing channels",
        "cumulative_revenue": "Cumulative Revenue",
        "campaign_events": "Campaign Events",
        "cohort_heatmap": "Retention Cohort Heatmap",
        "best_month": "Best Month",
        "worst_month": "Worst Month",
        "trend_summary": "Trend Summary",
        "color_scheme": "Chart Colors",
        "custom_palette": "Custom",
        "palette_input": "Hex colors (comma-separated)",
        "charts_tab": "Charts",
        "chart_type": "Chart type",
        "hist_users": "Users distribution",
        "hist_revenue": "Revenue distribution",
        "corr_heatmap": "Correlation heatmap",
        "scatter_rev_users": "Revenue vs Users",
        "rolling_metric": "Rolling averages",
        "roi_over_time": "Marketing ROI over time",
        "metric": "Metric",
        "window": "Window",
    },
    "العربية": {
        "data_source": "مصدر البيانات",
        "upload_csv": "رفع ملف CSV",
        "load_sample": "تحميل بيانات تجريبية",
        "setup": "رفع البيانات والإعداد",
        "preview": "معاينة البيانات",
        "rows": "عدد الصفوف",
        "companies": "عدد الشركات",
        "start": "بداية الفترة",
        "end": "نهاية الفترة",
        "company": "الشركة",
        "date_range": "نطاق التاريخ",
        "metrics_tab": "مقاييس الاستدامة",
        "decision_tab": "التقييم والقرار",
        "dashboard_tab": "لوحة الأدلة الرسومية",
        "export_tab": "التوصيات والتصدير",
        "dataset_loaded": "تم تحميل البيانات",
        "sample_loaded": "تم تحميل بيانات تجريبية",
        "need_data": "يرجى رفع CSV أو تحميل بيانات تجريبية للبدء.",
        "not_enough": "البيانات غير كافية في النطاق المحدد.",
        "decision": "القرار",
        "classification": "التصنيف",
        "score": "درجة الاستدامة",
        "confidence": "مستوى الثقة",
        "key_reasons": "أسباب رئيسية",
        "positives": "نقاط إيجابية",
        "negatives": "نقاط سلبية",
        "recommendations": "توصيات",
        "export": "تصدير",
        "download_csv": "تنزيل CSV",
        "download_pdf": "تنزيل PDF",
        "download_json": "تنزيل JSON",
        "limitations": "قيود النظام",
        "theme": "النمط",
        "dark": "داكن",
        "light": "فاتح",
        "forecast_tab": "التنبؤ",
        "forecast_horizon": "أفق التنبؤ (أشهر)",
        "forecast_note": "تنبؤ اتجاهي إرشادي فقط؛ ليس توقعًا مضمونًا.",
        "users_vs_active": "المستخدمون مقابل النشطين",
        "retention_rate": "معدل الاحتفاظ",
        "churn_rate": "معدل التسرّب",
        "rev_vs_marketing": "الإيرادات مقابل الإنفاق التسويقي",
        "ind_retention": "جودة الاحتفاظ",
        "ind_churn": "مخاطر التسرّب",
        "ind_vol": "تقلب النمو",
        "ind_rev": "واقعية الإيرادات",
        "ind_marketing": "كفاءة التسويق",
        "spikes": "قمم النمو",
        "dash_options": "خيارات الداشبورد",
        "smoothing_window": "نافذة التنعيم (أشهر)",
        "show_trend": "عرض خط الاتجاه",
        "segment_marketing": "تقسيم حسب قنوات التسويق",
        "cumulative_revenue": "الإيراد التراكمي",
        "campaign_events": "أحداث الحملات",
        "cohort_heatmap": "خريطة حرارية للدفعات (Cohort)",
        "best_month": "أفضل شهر",
        "worst_month": "أسوأ شهر",
        "trend_summary": "ملخّص الاتجاه",
        "color_scheme": "ألوان المخططات",
        "custom_palette": "مخصص",
        "palette_input": "ألوان Hex (مفصولة بفواصل)",
        "charts_tab": "الرسوم البيانية",
        "chart_type": "نوع الرسم",
        "hist_users": "توزيع المستخدمين",
        "hist_revenue": "توزيع الإيرادات",
        "corr_heatmap": "مصفوفة الارتباط",
        "scatter_rev_users": "الإيراد مقابل المستخدمين",
        "rolling_metric": "متوسطات متحركة",
        "roi_over_time": "عائد التسويق عبر الزمن",
        "metric": "المؤشر",
        "window": "النافذة",
    },
}
mode = st.sidebar.radio(T[lang]["data_source"], [T[lang]["upload_csv"], T[lang]["load_sample"]], index=1)
company_selection = None
start_date = None
end_date = None

if mode == T[lang]["upload_csv"]:
    uploaded = st.sidebar.file_uploader(T[lang]["upload_csv"], type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        clean_df, warnings, errors, summary = validate_and_clean(df)
        if errors:
            st.error(" ".join(errors))
        else:
            st.session_state["data"] = clean_df
            st.session_state["clean_summary"] = summary
            st.success(T[lang]["dataset_loaded"])
else:
    if st.sidebar.button(T[lang]["load_sample"]) or st.session_state["data"] is None:
        df = get_sample_data()
        clean_df, warnings, errors, summary = validate_and_clean(df)
        st.session_state["data"] = clean_df
        st.session_state["clean_summary"] = summary
        st.success(T[lang]["sample_loaded"])

data = st.session_state.get("data")
summary = st.session_state.get("clean_summary")

st.header(T[lang]["setup"])
if data is None:
    st.info(T[lang]["need_data"])
else:
    st.subheader(T[lang]["preview"])
    st.dataframe(data.head(10))
    cols = st.columns(4)
    cols[0].metric(T[lang]["rows"], summary["rows"]) 
    cols[1].metric(T[lang]["companies"], summary["companies"]) 
    cols[2].metric(T[lang]["start"], str(summary["min_date"]) if summary["min_date"] is not None else "-")
    cols[3].metric(T[lang]["end"], str(summary["max_date"]) if summary["max_date"] is not None else "-")
    miss_table = pd.DataFrame({"column": REQUIRED_COLUMNS, "missing": [summary["missing_counts"][c] for c in REQUIRED_COLUMNS]})
    st.table(miss_table)
    if summary["warnings"]:
        st.warning("; ".join(summary["warnings"]))
    companies = sorted(data["company_name"].unique().tolist())
    company_selection = st.selectbox(T[lang]["company"], companies, index=0)
    min_d = data["date"].min()
    max_d = data["date"].max()
    date_range = st.date_input(T[lang]["date_range"], value=(min_d, max_d))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
    st.session_state["company"] = company_selection
    st.session_state["date_range"] = (start_date, end_date)

theme_choice = st.sidebar.selectbox(T[lang]["theme"], [T[lang]["dark"], T[lang]["light"]], index=0)
if theme_choice == T[lang]["dark"]:
    st.markdown("""<style>
    .stApp { background-color: #0e1a2b; color: #e8edf5; }
    section[data-testid='stSidebar'] { background-color: #0b1424; }
    div[data-testid='stHeader'] { background-color: transparent; }
    </style>""", unsafe_allow_html=True)
    plotly_template = "plotly_dark"
else:
    st.markdown("""<style>
    .stApp { background-color: #ffffff; color: #0e1a2b; }
    section[data-testid='stSidebar'] { background-color: #f5f7fb; }
    div[data-testid='stHeader'] { background-color: transparent; }
    </style>""", unsafe_allow_html=True)
    plotly_template = "plotly"
scheme = st.sidebar.selectbox(T[lang]["color_scheme"], ["Classic", "Vibrant", "Cool", "Warm", "Monochrome", T[lang]["custom_palette"]], index=0)
if scheme == "Classic":
    palette = ["#4c9aff", "#f39c12", "#2ecc71", "#e74c3c", "#8e44ad", "#1abc9c"]
elif scheme == "Vibrant":
    palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f1c40f", "#e67e22"]
elif scheme == "Cool":
    palette = ["#2c3e50", "#3498db", "#16a085", "#8e44ad", "#95a5a6", "#1abc9c"]
elif scheme == "Warm":
    palette = ["#d35400", "#c0392b", "#e67e22", "#f1c40f", "#8e44ad", "#e74c3c"]
elif scheme == "Monochrome":
    palette = ["#111111", "#333333", "#555555", "#777777", "#999999", "#bbbbbb"]
else:
    pal_str = st.sidebar.text_input(T[lang]["palette_input"], "#4c9aff,#f39c12,#2ecc71,#e74c3c,#8e44ad,#1abc9c")
    palette = [c.strip() for c in pal_str.split(",") if c.strip().startswith("#")]
    if len(palette) == 0:
        palette = ["#4c9aff", "#f39c12", "#2ecc71", "#e74c3c", "#8e44ad", "#1abc9c"]
tabs = st.tabs([T[lang]["metrics_tab"], T[lang]["decision_tab"], T[lang]["dashboard_tab"], T[lang]["forecast_tab"], T[lang]["export_tab"], T[lang]["charts_tab"]]) 

if data is not None and company_selection is not None and start_date is not None and end_date is not None:
    subset = filter_company_range(data, company_selection, start_date, end_date)
    if len(subset) < 2:
        with tabs[0]:
            st.error(T[lang]["not_enough"])
    else:
        inds = cached_compute_indicators(subset)
        with tabs[0]:
            cards = [
                (T[lang]["ind_retention"], inds["values"]["Retention Quality"], inds["trends"]["Retention Quality"], inds["explanations"]["Retention Quality"]),
                (T[lang]["ind_churn"], inds["values"]["Churn Risk"], inds["trends"]["Churn Risk"], inds["explanations"]["Churn Risk"]),
                (T[lang]["ind_vol"], inds["values"]["Growth Volatility"], inds["trends"]["Growth Volatility"], inds["explanations"]["Growth Volatility"]),
                (T[lang]["ind_rev"], inds["values"]["Revenue Reality Check"], inds["trends"]["Revenue Reality Check"], inds["explanations"]["Revenue Reality Check"]),
                (T[lang]["ind_marketing"], inds["values"]["Marketing Efficiency"], inds["trends"]["Marketing Efficiency"], inds["explanations"]["Marketing Efficiency"]),
            ]
            cols = st.columns(5)
            for i, c in enumerate(cards):
                with cols[i]:
                    v = c[1]
                    t = c[2]
                    st.metric(c[0], value="-" if v is None else round(v, 3), delta=t)
                    if lang == "العربية" and inds.get("explanations_ar"):
                        key_map = {
                            T[lang]["ind_retention"]: "Retention Quality",
                            T[lang]["ind_churn"]: "Churn Risk",
                            T[lang]["ind_vol"]: "Growth Volatility",
                            T[lang]["ind_rev"]: "Revenue Reality Check",
                            T[lang]["ind_marketing"]: "Marketing Efficiency",
                        }
                        st.caption(inds["explanations_ar"].get(key_map[c[0]]) or c[3])
                    else:
                        st.caption(c[3])
        scoring = total_score_and_classification(inds["scores"]) 
        conf = compute_confidence(subset, summary, inds.get("volatility"), outliers=data_outliers_count(subset))
        kr = key_reasons(inds["scores"], inds["explanations"]) 
        kr_local = kr
        if lang == "العربية" and inds.get("explanations_ar"):
            items = [(k, v) for k, v in inds["scores"].items() if v is not None]
            items_sorted = sorted(items, key=lambda x: x[1])
            name_map_ar = {
                "Retention Quality": T[lang]["ind_retention"],
                "Churn Risk": T[lang]["ind_churn"],
                "Growth Volatility": T[lang]["ind_vol"],
                "Revenue Reality Check": T[lang]["ind_rev"],
                "Marketing Efficiency": T[lang]["ind_marketing"],
            }
            negatives_ar = [f"{name_map_ar[items_sorted[i][0]]}: {inds['explanations_ar'][items_sorted[i][0]]}" for i in range(min(3, len(items_sorted)))]
            positives_ar = [f"{name_map_ar[items_sorted[-(i+1)][0]]}: {inds['explanations_ar'][items_sorted[-(i+1)][0]]}" for i in range(min(3, len(items_sorted)))]
            kr_local = {"negatives": negatives_ar, "positives": positives_ar}
        recs = recommendations(inds["scores"]) 
        with tabs[1]:
            st.subheader(T[lang]["decision"])
            if scoring["classification"] is None:
                st.error("Cannot compute classification due to insufficient data.")
            else:
                st.metric(T[lang]["classification"], scoring["classification"], delta=None)
                st.metric(T[lang]["score"], scoring["total"], delta=None)
                st.metric(T[lang]["confidence"], conf["level"], delta=None)
                st.write(conf["reason"] if lang == "English" else conf.get("reason_ar", conf["reason"])) 
                if summary.get("months_coverage") and summary["months_coverage"] < 6:
                    st.warning("Short coverage (<6 months); confidence reduced." if lang == "English" else "تغطية زمنية قصيرة (<6 أشهر)؛ تم خفض مستوى الثقة.")
                if subset["revenue"].sum() == 0:
                    st.warning("Revenue is all zeros; monetization cannot be validated." if lang == "English" else "الإيرادات كلها صفراً؛ لا يمكن التحقق من جدوى الربح.")
                if (subset["marketing_spend"] == 0).all():
                    st.warning("Marketing spend is all zeros; efficiency assessment is limited." if lang == "English" else "الإنفاق التسويقي كله صفراً؛ تقييم الكفاءة محدود.")
                if scoring["classification"]:
                    summary_en = summary_text(scoring["classification"], kr)
                    summary_ar = "المؤشرات تشير إلى أداء مستدام مع توافق بين الاحتفاظ والتسرّب والربحية وتقلب مقبول." if scoring["classification"] == "Sustainable Success" else "المؤشرات تشير إلى فقاعة نجاح مدفوعة بالتقلب أو ضعف الاحتفاظ/توافق الإيرادات؛ ركّز على استقرار القيمة والربحية."
                    st.write(summary_en if lang == "English" else summary_ar)
                    negs = kr.get("negatives", [])
                    final_expl_en = ("Top issues: " + "; ".join(negs[:2])) if len(negs) > 0 else ""
                    final_expl_ar = ("أبرز المشاكل: " + "; ".join(negs[:2])) if len(negs) > 0 else ""
                    if final_expl_en:
                        st.write(final_expl_en if lang == "English" else final_expl_ar)
                bar_df = pd.DataFrame({"Indicator": list(inds["scores"].keys()), "Score": [inds["scores"][k] for k in inds["scores"]]})
                fig_dec = go.Figure(data=[go.Bar(x=bar_df["Indicator"], y=bar_df["Score"])])
                fig_dec.update_layout(template=plotly_template, margin=dict(l=40, r=20, t=40, b=40))
                _lock_fig(fig_dec)
                st.plotly_chart(fig_dec, use_container_width=True, config=plotly_config)
                table_df = bar_df.copy()
                table_df.loc[len(table_df)] = ["Total", scoring["total"]]
                st.table(table_df)
                st.subheader(T[lang]["key_reasons"])
                st.write(T[lang]["positives"])
                for s in (kr["positives"] if lang == "English" else kr_local["positives"]):
                    st.write(f"• {s}")
                st.write(T[lang]["negatives"])
                for s in (kr["negatives"] if lang == "English" else kr_local["negatives"]):
                    st.write(f"• {s}")
with tabs[2]:
            with st.expander(T[lang]["dash_options"], expanded=True):
                smooth_win = st.slider(T[lang]["smoothing_window"], min_value=1, max_value=6, value=1)
                show_trend = st.checkbox(T[lang]["show_trend"], value=True)
                seg_marketing = st.checkbox(T[lang]["segment_marketing"], value=False)
                events_file = st.file_uploader(("Campaign events CSV" if lang=="English" else "ملف أحداث الحملات (CSV)"), type=["csv"]) 
                import io as _io
                _buf = _io.StringIO()
                _buf.write("date,label\n")
                try:
                    sample_dates = list(subset["date"].iloc[:2])
                except Exception:
                    sample_dates = []
                if len(sample_dates) == 0:
                    _buf.write("2025-01-01,Sample Event\n")
                    _buf.write("2025-02-01,Sample Event 2\n")
                else:
                    for _d in sample_dates:
                        _buf.write(f"{pd.to_datetime(_d).date()},Sample Event\n")
                template_csv_bytes = _buf.getvalue().encode("utf-8")
                st.download_button(("Download campaign events CSV template" if lang=="English" else "تنزيل قالب CSV لأحداث الحملات"), template_csv_bytes, file_name="campaign_events_template.csv", mime="text/csv")
            def _smooth(s):
                return s.rolling(window=smooth_win, min_periods=1).mean() if smooth_win > 1 else s
            u_series = _smooth(subset["users"]) 
            a_series = _smooth(subset["active_users"]) 
            fig1 = px.line(pd.DataFrame({"date": subset["date"], "users": u_series, "active_users": a_series}), x="date", y=["users", "active_users"], title=T[lang]["users_vs_active"], template=plotly_template, color_discrete_sequence=palette)
            fig1.update_layout(hovermode="x unified")
            fig1.update_layout(margin=dict(l=40, r=20, t=60, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig1.update_xaxes(showgrid=True, gridcolor="rgba(127,127,127,0.2)")
            fig1.update_yaxes(showgrid=True, gridcolor="rgba(127,127,127,0.2)")
            rates = cached_growth_rates(subset["users"])
            if len(rates) > 0:
                spike_mask = rates >= 0.40
                spike_dates = subset["date"].iloc[1:][spike_mask]
                spike_vals = subset["users"].iloc[1:][spike_mask]
                if len(spike_dates) > 0:
                    fig1.add_trace(go.Scatter(x=spike_dates, y=spike_vals, mode="markers", name=T[lang]["spikes"], marker=dict(color="red", size=8)))
            if show_trend:
                x_idx = np.arange(len(subset))
                coeff = np.polyfit(x_idx, subset["users"].astype(float).values, 1) if len(subset) >= 2 else np.array([0.0, subset["users"].iloc[0] if len(subset) else 0.0])
                trend = coeff[0] * x_idx + coeff[1]
                fig1.add_trace(go.Scatter(x=subset["date"], y=trend, name=("Users trend" if lang=="English" else "اتجاه المستخدمين"), line=dict(dash="dot", color="orange")))
            ev_range = None
            if events_file is not None:
                try:
                    ev = parse_events_csv(events_file.getvalue())
                    if "date" in ev.columns:
                        ev = ev.sort_values("date")
                        ev_range = ev[(ev["date"] >= subset["date"].min()) & (ev["date"] <= subset["date"].max())]
                except Exception:
                    ev_range = None
            if ev_range is not None and len(ev_range) > 0:
                map_u = pd.DataFrame({"date": subset["date"], "y": u_series})
                ev_u = ev_range.merge(map_u, on="date", how="left")
                fig1.add_trace(go.Scatter(x=ev_u["date"], y=ev_u["y"], mode="markers", name=T[lang]["campaign_events"], marker=dict(color="purple", size=9), text=ev_range["label"] if "label" in ev_range.columns else None))
            _lock_fig(fig1)
            st.plotly_chart(fig1, use_container_width=True, config=plotly_config)
            k1 = (subset["active_users"] / subset["users"].replace(0, pd.NA)).clip(lower=0, upper=1).dropna().mean()
            k2 = (subset["revenue"] / subset["users"].replace(0, pd.NA)).dropna().mean()
            spend_mask = subset["marketing_spend"] > 0
            k3 = (subset.loc[spend_mask, "revenue"] / subset.loc[spend_mask, "marketing_spend"]).dropna().mean() if spend_mask.any() else None
            k4 = data_outliers_count(subset)
            k_cols = st.columns(4)
            k_cols[0].metric("Engagement Ratio" if lang=="English" else "نسبة الارتباط", value="-" if pd.isna(k1) else round(float(k1), 3))
            k_cols[1].metric("ARPU" if lang=="English" else "إيراد لكل مستخدم", value="-" if pd.isna(k2) else round(float(k2), 2))
            k_cols[2].metric("Marketing ROI" if lang=="English" else "عائد التسويق", value="-" if k3 is None or pd.isna(k3) else round(float(k3), 2))
            k_cols[3].metric("Outliers" if lang=="English" else "قيم شاذة", value=int(k4))
            fig2 = px.line(subset, x="date", y="retention_rate", title=T[lang]["retention_rate"], template=plotly_template, color_discrete_sequence=[palette[3]])
            fig2.update_layout(hovermode="x unified")
            fig2.update_layout(margin=dict(l=40, r=20, t=60, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig2.update_xaxes(showgrid=True, gridcolor="rgba(127,127,127,0.2)")
            fig2.update_yaxes(showgrid=True, gridcolor="rgba(127,127,127,0.2)", tickformat=".1%")
            _lock_fig(fig2)
            st.plotly_chart(fig2, use_container_width=True, config=plotly_config)
            fig3 = px.line(subset, x="date", y="churn_rate", title=T[lang]["churn_rate"], template=plotly_template, color_discrete_sequence=[palette[4]])
            fig3.update_layout(hovermode="x unified")
            fig3.update_layout(margin=dict(l=40, r=20, t=60, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig3.update_xaxes(showgrid=True, gridcolor="rgba(127,127,127,0.2)")
            fig3.update_yaxes(showgrid=True, gridcolor="rgba(127,127,127,0.2)", tickformat=".1%")
            _lock_fig(fig3)
            st.plotly_chart(fig3, use_container_width=True, config=plotly_config)
            rev_series = _smooth(subset["revenue"]) 
            spend_series = _smooth(subset["marketing_spend"]) 
            fig4 = px.line(pd.DataFrame({"date": subset["date"], "revenue": rev_series, "marketing_spend": spend_series}), x="date", y=["revenue", "marketing_spend"], title=T[lang]["rev_vs_marketing"], template=plotly_template, color_discrete_sequence=palette)
            fig4.update_layout(hovermode="x unified")
            fig4.update_layout(margin=dict(l=40, r=20, t=60, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig4.update_xaxes(showgrid=True, gridcolor="rgba(127,127,127,0.2)")
            fig4.update_yaxes(showgrid=True, gridcolor="rgba(127,127,127,0.2)")
            if ev_range is not None and len(ev_range) > 0:
                map_r = pd.DataFrame({"date": subset["date"], "y": rev_series})
                ev_r = ev_range.merge(map_r, on="date", how="left")
                fig4.add_trace(go.Scatter(x=ev_r["date"], y=ev_r["y"], mode="markers", name=T[lang]["campaign_events"], marker=dict(color="purple", size=9), text=ev_range["label"] if "label" in ev_range.columns else None))
            _lock_fig(fig4)
            st.plotly_chart(fig4, use_container_width=True, config=plotly_config)
            cum_df = pd.DataFrame({"date": subset["date"], "revenue_cum": subset["revenue"].cumsum(), "spend_cum": subset["marketing_spend"].cumsum()})
            fig_cum = px.line(cum_df, x="date", y=["revenue_cum", "spend_cum"], title=T[lang]["cumulative_revenue"], template=plotly_template, color_discrete_sequence=palette)
            fig_cum.update_layout(hovermode="x unified")
            fig_cum.update_layout(margin=dict(l=40, r=20, t=60, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig_cum.update_xaxes(showgrid=True, gridcolor="rgba(127,127,127,0.2)")
            fig_cum.update_yaxes(showgrid=True, gridcolor="rgba(127,127,127,0.2)")
            spend_growth = cached_growth_rates(subset["marketing_spend"]) if len(subset) > 1 else np.array([])
            if len(spend_growth) > 0:
                big_inc = spend_growth >= 0.30
                ev_dates = subset["date"].iloc[1:][big_inc]
                ev_vals = cum_df["revenue_cum"].iloc[1:][big_inc]
                if len(ev_dates) > 0:
                    fig_cum.add_trace(go.Scatter(x=ev_dates, y=ev_vals, mode="markers", name=T[lang]["campaign_events"], marker=dict(color="purple", size=7)))
            if ev_range is not None and len(ev_range) > 0:
                map_c = pd.DataFrame({"date": cum_df["date"], "y": cum_df["revenue_cum"]})
                ev_c = ev_range.merge(map_c, on="date", how="left")
                fig_cum.add_trace(go.Scatter(x=ev_c["date"], y=ev_c["y"], mode="markers", name=T[lang]["campaign_events"], marker=dict(color="purple", size=9), text=ev_range["label"] if "label" in ev_range.columns else None))
            _lock_fig(fig_cum)
            st.plotly_chart(fig_cum, use_container_width=True, config=plotly_config)
            if seg_marketing and {"paid_marketing_spend", "organic_marketing_spend"}.issubset(set(subset.columns)):
                seg_df = pd.DataFrame({"date": subset["date"], "paid": subset["paid_marketing_spend"], "organic": subset["organic_marketing_spend"]})
                fig_seg = px.area(seg_df, x="date", y=["paid", "organic"], title=("Marketing channels" if lang=="English" else "قنوات التسويق"), template=plotly_template, color_discrete_sequence=palette[:2])
                _lock_fig(fig_seg)
                st.plotly_chart(fig_seg, use_container_width=True, config=plotly_config)
            elif seg_marketing:
                st.info(("Segmentation requires paid/organic columns" if lang=="English" else "التقسيم يتطلب أعمدة للمدفوع/العضوي"))
            max_h = min(6, len(subset))
            if max_h >= 2:
                rows = []
                labels = []
                for i in range(len(subset) - 1):
                    horizon = min(max_h, len(subset) - i)
                    vals = []
                    for j in range(horizon):
                        vals.append(float(subset["retention_rate"].iloc[i+j]))
                    rows.append(vals + [np.nan]*(max_h - horizon))
                    labels.append(str(subset["date"].iloc[i].date()))
                heat = pd.DataFrame(rows, index=labels, columns=[f"t+{k}" for k in range(max_h)])
                fig_co = px.imshow(heat, aspect="auto", color_continuous_scale="Viridis", origin="lower")
                fig_co.update_layout(title=T[lang]["cohort_heatmap"], template=plotly_template)
                _lock_fig(fig_co)
                st.plotly_chart(fig_co, use_container_width=True, config=plotly_config)
            bm_cols = st.columns(3)
            best_rev_idx = int(subset["revenue"].astype(float).idxmax()) if len(subset) else 0
            worst_rev_idx = int(subset["revenue"].astype(float).idxmin()) if len(subset) else 0
            bm_cols[0].metric(T[lang]["best_month"], str(subset["date"].iloc[best_rev_idx].date()) if len(subset) else "-")
            bm_cols[1].metric(T[lang]["worst_month"], str(subset["date"].iloc[worst_rev_idx].date()) if len(subset) else "-")
            prev_mean = float(subset["revenue"].astype(float).iloc[:-3].mean()) if len(subset) > 3 else float(subset["revenue"].astype(float).mean())
            recent_mean = float(subset["revenue"].astype(float).iloc[-3:].mean()) if len(subset) >= 3 else float(subset["revenue"].astype(float).mean())
            trend_txt = ("Up" if recent_mean > prev_mean else "Down" if recent_mean < prev_mean else "Flat") if lang=="English" else ("صاعد" if recent_mean > prev_mean else "هابط" if recent_mean < prev_mean else "مستقر")
            bm_cols[2].metric(T[lang]["trend_summary"], trend_txt)
            with tabs[3]:
                st.subheader(T[lang]["forecast_tab"])
                horizon = st.slider(T[lang]["forecast_horizon"], min_value=3, max_value=12, value=6)
                suggested = recommend_forecast_method(subset)
                default_idx = 0 if suggested=="linear" else (1 if suggested=="holt" else 2)
                method = st.selectbox(("Method" if lang=="English" else "الأسلوب"), ["Linear", "Holt", "Seasonal+Trend"], index=default_idx)
                st.caption(T[lang]["forecast_note"])
                show_bands = st.checkbox(("Show confidence bands" if lang=="English" else "عرض نطاقات الثقة"), value=True)
                m_key = "linear" if method == "Linear" else ("holt" if method == "Holt" else "seasonal")
                method_params = {}
                if m_key == "holt":
                    auto_tune = st.checkbox(("Auto-tune Holt (alpha/beta)" if lang=="English" else "ضبط تلقائي لهولت (alpha/beta)"), value=True)
                    if auto_tune:
                        tuned = tune_holt_params(subset["users"]) 
                        method_params.update(tuned)
                    else:
                        a = st.slider("alpha", 0.1, 0.9, 0.5)
                        b = st.slider("beta", 0.1, 0.9, 0.3)
                        method_params.update({"alpha": float(a), "beta": float(b)})
                if m_key == "seasonal":
                    season_len = st.slider(("Season length" if lang=="English" else "طول الموسم"), min_value=3, max_value=12, value=12)
                    method_params.update({"season": int(season_len)})
                fdf = forecast_df_confidence(subset, horizon, m_key, method_params)
                evals = evaluate_forecast_df(subset, m_key, method_params=method_params)
            if len(fdf) == 0:
                st.info(T[lang]["not_enough"])
            else:
                figf1 = go.Figure()
                figf1.add_trace(go.Scatter(x=subset["date"], y=subset["users"], name=(("Actual users") if lang=="English" else "المستخدمون الفعليون"), line=dict(color=palette[0])))
                figf1.add_trace(go.Scatter(x=fdf["date"], y=fdf["users"], name=(("Forecast users") if lang=="English" else "تنبؤ المستخدمين"), line=dict(dash="dash", color=palette[0])))
                if show_bands:
                    figf1.add_trace(go.Scatter(x=fdf["date"], y=fdf["users_lo"], name="lo", line=dict(color=_hex_to_rgba(palette[0], 0.12))))
                    figf1.add_trace(go.Scatter(x=fdf["date"], y=fdf["users_hi"], name="hi", fill="tonexty", line=dict(color=_hex_to_rgba(palette[0], 0.12))))
                figf1.update_layout(title=("Users forecast" if lang=="English" else "تنبؤ المستخدمين"), template=plotly_template)
                _lock_fig(figf1)
                st.plotly_chart(figf1, use_container_width=True, config=plotly_config)
                figf2 = go.Figure()
                figf2.add_trace(go.Scatter(x=subset["date"], y=subset["revenue"], name=(("Actual revenue") if lang=="English" else "الإيراد الفعلي"), line=dict(color=palette[2])))
                figf2.add_trace(go.Scatter(x=fdf["date"], y=fdf["revenue"], name=(("Forecast revenue") if lang=="English" else "تنبؤ الإيرادات"), line=dict(dash="dash", color=palette[2])))
                if show_bands:
                    figf2.add_trace(go.Scatter(x=fdf["date"], y=fdf["revenue_lo"], name="lo", line=dict(color=_hex_to_rgba(palette[2], 0.12))))
                    figf2.add_trace(go.Scatter(x=fdf["date"], y=fdf["revenue_hi"], name="hi", fill="tonexty", line=dict(color=_hex_to_rgba(palette[2], 0.12))))
                figf2.update_layout(title=("Revenue forecast" if lang=="English" else "تنبؤ الإيرادات"), template=plotly_template)
                _lock_fig(figf2)
                st.plotly_chart(figf2, use_container_width=True, config=plotly_config)
                figf3 = go.Figure()
                figf3.add_trace(go.Scatter(x=subset["date"], y=subset["retention_rate"], name=(("Actual retention") if lang=="English" else "الاحتفاظ الفعلي"), line=dict(color=palette[3])))
                figf3.add_trace(go.Scatter(x=fdf["date"], y=fdf["retention_rate"], name=(("Forecast retention") if lang=="English" else "تنبؤ الاحتفاظ"), line=dict(dash="dash", color=palette[3])))
                if show_bands:
                    figf3.add_trace(go.Scatter(x=fdf["date"], y=fdf["retention_lo"], name="lo", line=dict(color=_hex_to_rgba(palette[3], 0.12))))
                    figf3.add_trace(go.Scatter(x=fdf["date"], y=fdf["retention_hi"], name="hi", fill="tonexty", line=dict(color=_hex_to_rgba(palette[3], 0.12))))
                figf3.update_layout(title=("Retention forecast" if lang=="English" else "تنبؤ الاحتفاظ"), template=plotly_template)
                _lock_fig(figf3)
                st.plotly_chart(figf3, use_container_width=True, config=plotly_config)
                m_cols = st.columns(3)
                u_mae = evals["users"]["mae"]
                u_mape = evals["users"]["mape"]
                r_mae = evals["revenue"]["mae"]
                r_mape = evals["revenue"]["mape"]
                ret_mae = evals["retention"]["mae"]
                ret_mape = evals["retention"]["mape"]
                m_cols[0].metric(("Users MAE" if lang=="English" else "MAE للمستخدمين"), value="-" if u_mae is None else round(float(u_mae),2))
                m_cols[0].metric(("Users MAPE" if lang=="English" else "MAPE للمستخدمين"), value="-" if u_mape is None or np.isnan(u_mape) else f"{round(float(u_mape*100),1)}%")
                m_cols[1].metric(("Revenue MAE" if lang=="English" else "MAE للإيراد"), value="-" if r_mae is None else round(float(r_mae),2))
                m_cols[1].metric(("Revenue MAPE" if lang=="English" else "MAPE للإيراد"), value="-" if r_mape is None or np.isnan(r_mape) else f"{round(float(r_mape*100),1)}%")
                m_cols[2].metric(("Retention MAE" if lang=="English" else "MAE للاحتفاظ"), value="-" if ret_mae is None else round(float(ret_mae),3))
                m_cols[2].metric(("Retention MAPE" if lang=="English" else "MAPE للاحتفاظ"), value="-" if ret_mape is None or np.isnan(ret_mape) else f"{round(float(ret_mape*100),1)}%")
                f_cols = st.columns(3)
                f_cols[0].metric("Users (end)" if lang=="English" else "المستخدمون (نهاية)", value=round(float(fdf["users"].iloc[-1]),2))
                f_cols[1].metric("Revenue (end)" if lang=="English" else "الإيراد (نهاية)", value=round(float(fdf["revenue"].iloc[-1]),2))
                f_cols[2].metric("Retention (end)" if lang=="English" else "الاحتفاظ (نهاية)", value=round(float(fdf["retention_rate"].iloc[-1]),3))
                fdf_json = fdf.copy()
                if "date" in fdf_json.columns:
                    fdf_json["date"] = pd.to_datetime(fdf_json["date"]).dt.strftime("%Y-%m-%d")
                f_payload = {
                    "horizon_months": horizon,
                    "method": m_key,
                    "forecast": fdf_json.to_dict(orient="records"),
                }
                import json as _json
                f_json = _json.dumps(f_payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
                st.download_button(("Download forecast JSON" if lang=="English" else "تنزيل JSON للتنبؤ"), f_json, file_name="forecast.json", mime="application/json")
                f_csv = fdf.to_csv(index=False).encode("utf-8")
                st.download_button(("Download forecast CSV" if lang=="English" else "تنزيل CSV للتنبؤ"), f_csv, file_name="forecast.csv", mime="text/csv")
            with tabs[4]:
                st.subheader(T[lang]["recommendations"])
                for r in recs:
                    st.write(f"• {r}")
                st.subheader(T[lang]["export"])
                csv_df = build_scores_dataframe(company_selection, start_date, end_date, inds["scores"]) 
            csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
            st.download_button(T[lang]["download_csv"], csv_bytes, file_name="scores.csv", mime="text/csv")
            payload = {
                "company": company_selection,
                "start_date": str(start_date.date()),
                "end_date": str(end_date.date()),
                "scores": inds["scores"],
                "total": int(scoring["total"]),
                "classification": scoring["classification"],
                "confidence": {"level": conf["level"], "reason": conf["reason"], "reason_ar": conf.get("reason_ar")},
                "reasons": (kr if lang == "English" else kr_local),
                "recommendations": recs,
            }
            import json as _json
            json_bytes = _json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button(T[lang]["download_json"], json_bytes, file_name="summary.json", mime="application/json")
            limitations = (
                [
                    "Decision-support only, not predictive.",
                    "No forecast or guarantee of future performance.",
                    "No financial/legal/investment advice.",
                    "Excludes external market/competitor/news factors.",
                    "Reliability depends on data quality and coverage.",
                    "Cannot determine root causes beyond metric insights.",
                    "Uses fixed rules and does not learn automatically.",
                ]
                if lang == "English"
                else [
                    "الأداة دعم قرار وليست تنبؤية.",
                    "لا توجد توقعات أو ضمانات للأداء المستقبلي.",
                    "ليست نصيحة مالية/قانونية/استثمارية.",
                    "لا تشمل عوامل السوق/المنافسين/الأخبار.",
                    "تعتمد الموثوقية على جودة البيانات والتغطية.",
                    "لا تحدد الأسباب الجذرية خارج رؤى المقاييس.",
                    "تستخدم قواعد ثابتة ولا تتعلم تلقائيًا.",
                ]
            )
            pdf_conf_reason = conf["reason"] if lang == "English" else conf.get("reason_ar", conf["reason"]) 
            pdf_pos = kr["positives"] if lang == "English" else kr_local["positives"]
            pdf_neg = kr["negatives"] if lang == "English" else kr_local["negatives"]
            pdf_bytes = make_pdf(company_selection, f"{str(start_date.date())} to {str(end_date.date())}", scoring["classification"] or "-", scoring["total"], pdf_conf_reason, pdf_pos, pdf_neg, inds["scores"], recs, limitations)
            st.download_button(T[lang]["download_pdf"], data=pdf_bytes, file_name="summary.pdf", mime="application/pdf")
            st.subheader(T[lang]["limitations"])
            for l in limitations:
                st.write(f"- {l}")

            with tabs[5]:
                st.subheader(T[lang]["charts_tab"])
                chart_opts = [
                    T[lang]["hist_users"],
                    T[lang]["hist_revenue"],
                    T[lang]["corr_heatmap"],
                    T[lang]["scatter_rev_users"],
                    T[lang]["rolling_metric"],
                    T[lang]["roi_over_time"],
                ]
                preset_opts = ["None" if lang=="English" else "لا شيء", ("Growth Overview" if lang=="English" else "نظرة عامة على النمو"), ("Monetization Overview" if lang=="English" else "نظرة عامة على الربحية"), ("Retention Overview" if lang=="English" else "نظرة عامة على الاحتفاظ")]
                preset = st.selectbox(("Preset" if lang=="English" else "مخططات جاهزة"), preset_opts, index=0)
                ctype = st.selectbox(T[lang]["chart_type"], chart_opts, index=0)
                if preset != ("None" if lang=="English" else "لا شيء"):
                    if preset.startswith("Growth") or preset.startswith("نظرة عامة على النمو"):
                        cols_p = st.columns(2)
                        with cols_p[0]:
                            w = 3
                            uu = subset["users"].astype(float).rolling(window=w, min_periods=1).mean()
                            aa = subset["active_users"].astype(float).rolling(window=w, min_periods=1).mean()
                            fig = px.line(pd.DataFrame({"date": subset["date"], "users": uu, "active_users": aa}), x="date", y=["users", "active_users"], title=T[lang]["users_vs_active"], template=plotly_template, color_discrete_sequence=palette)
                            fig.update_layout(hovermode="x unified", margin=dict(l=40,r=20,t=60,b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                            fig.update_xaxes(showgrid=True, gridcolor="rgba(127,127,127,0.2)")
                            fig.update_yaxes(showgrid=True, gridcolor="rgba(127,127,127,0.2)")
                            _lock_fig(fig)
                            st.plotly_chart(fig, use_container_width=True, config=plotly_config)
                        with cols_p[1]:
                            rates = cached_growth_rates(subset["users"]) if len(subset) > 1 else np.array([])
                            gr = pd.DataFrame({"date": subset["date"].iloc[1:], "growth": rates}) if len(rates) > 0 else pd.DataFrame({"date": [], "growth": []})
                            fig = px.bar(gr, x="date", y="growth", title=("Monthly user growth" if lang=="English" else "نمو المستخدمين الشهري"), template=plotly_template, color_discrete_sequence=[palette[4]])
                            fig.update_yaxes(tickformat=".1%")
                            fig.update_layout(hovermode="x unified", margin=dict(l=40,r=20,t=60,b=40))
                            _lock_fig(fig)
                            st.plotly_chart(fig, use_container_width=True, config=plotly_config)
                    elif preset.startswith("Monetization") or preset.startswith("نظرة عامة على الربحية"):
                        cols_p = st.columns(2)
                        with cols_p[0]:
                            w = 3
                            rr = subset["revenue"].astype(float).rolling(window=w, min_periods=1).mean()
                            ss = subset["marketing_spend"].astype(float).rolling(window=w, min_periods=1).mean()
                            fig = px.line(pd.DataFrame({"date": subset["date"], "revenue": rr, "marketing_spend": ss}), x="date", y=["revenue", "marketing_spend"], title=T[lang]["rev_vs_marketing"], template=plotly_template, color_discrete_sequence=palette)
                            fig.update_layout(hovermode="x unified", margin=dict(l=40,r=20,t=60,b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                            fig.update_xaxes(showgrid=True, gridcolor="rgba(127,127,127,0.2)")
                            fig.update_yaxes(showgrid=True, gridcolor="rgba(127,127,127,0.2)")
                            _lock_fig(fig)
                            st.plotly_chart(fig, use_container_width=True, config=plotly_config)
                        with cols_p[1]:
                            cum_df = pd.DataFrame({"date": subset["date"], "revenue_cum": subset["revenue"].cumsum(), "spend_cum": subset["marketing_spend"].cumsum()})
                            fig = px.line(cum_df, x="date", y=["revenue_cum", "spend_cum"], title=T[lang]["cumulative_revenue"], template=plotly_template, color_discrete_sequence=palette)
                            fig.update_layout(hovermode="x unified", margin=dict(l=40,r=20,t=60,b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                            fig.update_xaxes(showgrid=True, gridcolor="rgba(127,127,127,0.2)")
                            fig.update_yaxes(showgrid=True, gridcolor="rgba(127,127,127,0.2)")
                            _lock_fig(fig)
                            st.plotly_chart(fig, use_container_width=True, config=plotly_config)
                    elif preset.startswith("Retention") or preset.startswith("نظرة عامة على الاحتفاظ"):
                        cols_p = st.columns(2)
                        with cols_p[0]:
                            fig = px.line(subset, x="date", y="retention_rate", title=T[lang]["retention_rate"], template=plotly_template, color_discrete_sequence=[palette[3]])
                            fig.update_yaxes(tickformat=".1%")
                            fig.update_layout(hovermode="x unified", margin=dict(l=40,r=20,t=60,b=40))
                            _lock_fig(fig)
                            st.plotly_chart(fig, use_container_width=True, config=plotly_config)
                        with cols_p[1]:
                            fig = px.line(subset, x="date", y="churn_rate", title=T[lang]["churn_rate"], template=plotly_template, color_discrete_sequence=[palette[4]])
                            fig.update_yaxes(tickformat=".1%")
                            fig.update_layout(hovermode="x unified", margin=dict(l=40,r=20,t=60,b=40))
                            _lock_fig(fig)
                            st.plotly_chart(fig, use_container_width=True, config=plotly_config)
                else:
                    if ctype == T[lang]["hist_users"]:
                        edges = np.histogram_bin_edges(subset["users"].astype(float).values, bins="auto") if len(subset) > 1 else np.array([0,1])
                        fig = px.histogram(subset, x="users", nbins=max(5, min(30, len(edges)-1)), title=T[lang]["hist_users"], template=plotly_template, color_discrete_sequence=[palette[0]])
                        _lock_fig(fig)
                        st.plotly_chart(fig, use_container_width=True, config=plotly_config)
                    elif ctype == T[lang]["hist_revenue"]:
                        edges = np.histogram_bin_edges(subset["revenue"].astype(float).values, bins="auto") if len(subset) > 1 else np.array([0,1])
                        fig = px.histogram(subset, x="revenue", nbins=max(5, min(30, len(edges)-1)), title=T[lang]["hist_revenue"], template=plotly_template, color_discrete_sequence=[palette[2]])
                        _lock_fig(fig)
                        st.plotly_chart(fig, use_container_width=True, config=plotly_config)
                    elif ctype == T[lang]["corr_heatmap"]:
                        cols = ["users", "active_users", "revenue", "marketing_spend", "retention_rate", "churn_rate"]
                        dfc = subset[cols].astype(float).corr()
                        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale="RdBu", title=T[lang]["corr_heatmap"])
                        fig.update_layout(template=plotly_template)
                        _lock_fig(fig)
                        st.plotly_chart(fig, use_container_width=True, config=plotly_config)
                    elif ctype == T[lang]["scatter_rev_users"]:
                        fig = go.Figure()
                        fig.add_trace(go.Scattergl(x=subset["users"], y=subset["revenue"], mode="markers", name=T[lang]["scatter_rev_users"], marker=dict(color=palette[1])))
                        if len(subset) >= 2:
                            x = subset["users"].astype(float).values
                            y = subset["revenue"].astype(float).values
                            a, b = np.polyfit(x, y, 1)
                            xr = np.linspace(float(np.min(x)), float(np.max(x)), 50)
                            yr = a * xr + b
                            fig.add_trace(go.Scatter(x=xr, y=yr, name=("Trend" if lang=="English" else "اتجاه"), line=dict(color="orange")))
                        fig.update_layout(title=T[lang]["scatter_rev_users"], template=plotly_template)
                        _lock_fig(fig)
                        st.plotly_chart(fig, use_container_width=True, config=plotly_config)
                    elif ctype == T[lang]["rolling_metric"]:
                        m = st.selectbox(T[lang]["metric"], ["users", "active_users", "revenue", "marketing_spend", "retention_rate", "churn_rate"], index=0)
                        w = st.slider(T[lang]["window"], min_value=2, max_value=6, value=3)
                        base = subset[m].astype(float)
                        roll = base.rolling(window=w, min_periods=1).mean()
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=subset["date"], y=base, name=m, line=dict(color=palette[0])))
                        fig.add_trace(go.Scatter(x=subset["date"], y=roll, name=("Rolling" if lang=="English" else "متوسط متحرك"), line=dict(dash="dash", color=palette[1])))
                        fig.update_layout(title=T[lang]["rolling_metric"], template=plotly_template)
                        _lock_fig(fig)
                        st.plotly_chart(fig, use_container_width=True, config=plotly_config)
                    elif ctype == T[lang]["roi_over_time"]:
                        spend_mask = subset["marketing_spend"].astype(float) > 0
                        roi = (subset.loc[spend_mask, "revenue"].astype(float) / subset.loc[spend_mask, "marketing_spend"].astype(float))
                        dfroi = pd.DataFrame({"date": subset.loc[spend_mask, "date"], "roi": roi})
                        fig = px.line(dfroi, x="date", y="roi", title=T[lang]["roi_over_time"], template=plotly_template, color_discrete_sequence=[palette[5]])
                        _lock_fig(fig)
                        st.plotly_chart(fig, use_container_width=True, config=plotly_config)

st.sidebar.markdown("Run: pip install -r requirements.txt")
st.sidebar.markdown("Run: streamlit run app.py")
