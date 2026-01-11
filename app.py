import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from utils import validate_and_clean, filter_company_range, REQUIRED_COLUMNS
from sample_data import get_sample_data
from analysis_engine import compute_indicators, total_score_and_classification, compute_confidence, key_reasons, recommendations, build_scores_dataframe, make_pdf, data_outliers_count

st.set_page_config(page_title="Success Bubble Detector", layout="wide")
st.set_option('browser.gatherUsageStats', False)

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
        "limitations": "System Limitations",
        "users_vs_active": "Users vs Active Users",
        "retention_rate": "Retention Rate",
        "churn_rate": "Churn Rate",
        "rev_vs_marketing": "Revenue vs Marketing Spend",
        "ind_retention": "Retention Quality",
        "ind_churn": "Churn Risk",
        "ind_vol": "Growth Volatility",
        "ind_rev": "Revenue Reality Check",
        "ind_marketing": "Marketing Efficiency",
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
        "limitations": "قيود النظام",
        "users_vs_active": "المستخدمون مقابل النشطين",
        "retention_rate": "معدل الاحتفاظ",
        "churn_rate": "معدل التسرّب",
        "rev_vs_marketing": "الإيرادات مقابل الإنفاق التسويقي",
        "ind_retention": "جودة الاحتفاظ",
        "ind_churn": "مخاطر التسرّب",
        "ind_vol": "تقلب النمو",
        "ind_rev": "واقعية الإيرادات",
        "ind_marketing": "كفاءة التسويق",
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

tabs = st.tabs([T[lang]["metrics_tab"], T[lang]["decision_tab"], T[lang]["dashboard_tab"], T[lang]["export_tab"]]) 

if data is not None and company_selection is not None and start_date is not None and end_date is not None:
    subset = filter_company_range(data, company_selection, start_date, end_date)
    if len(subset) < 2:
        with tabs[0]:
            st.error(T[lang]["not_enough"])
    else:
        inds = compute_indicators(subset)
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
                    st.caption(c[3])
        scoring = total_score_and_classification(inds["scores"]) 
        conf = compute_confidence(subset, summary, inds.get("volatility"), outliers=data_outliers_count(subset))
        kr = key_reasons(inds["scores"], inds["explanations"]) 
        recs = recommendations(inds["scores"]) 
        with tabs[1]:
            st.subheader(T[lang]["decision"])
            if scoring["classification"] is None:
                st.error("Cannot compute classification due to insufficient data.")
            else:
                st.metric(T[lang]["classification"], scoring["classification"], delta=None)
                st.metric(T[lang]["score"], scoring["total"], delta=None)
                st.metric(T[lang]["confidence"], conf["level"], delta=None)
                st.write(conf["reason"]) 
                bar_df = pd.DataFrame({"Indicator": list(inds["scores"].keys()), "Score": [inds["scores"][k] for k in inds["scores"]]})
                st.bar_chart(bar_df.set_index("Indicator"))
                st.subheader(T[lang]["key_reasons"])
                st.write(T[lang]["positives"])
                for s in kr["positives"]:
                    st.write(f"• {s}")
                st.write(T[lang]["negatives"])
                for s in kr["negatives"]:
                    st.write(f"• {s}")
        with tabs[2]:
            fig1 = px.line(subset, x="date", y=["users", "active_users"], title=T[lang]["users_vs_active"])
            st.plotly_chart(fig1, use_container_width=True)
            fig2 = px.line(subset, x="date", y="retention_rate", title=T[lang]["retention_rate"])
            st.plotly_chart(fig2, use_container_width=True)
            fig3 = px.line(subset, x="date", y="churn_rate", title=T[lang]["churn_rate"])
            st.plotly_chart(fig3, use_container_width=True)
            fig4 = px.line(subset, x="date", y=["revenue", "marketing_spend"], title=T[lang]["rev_vs_marketing"])
            st.plotly_chart(fig4, use_container_width=True)
        with tabs[3]:
            st.subheader(T[lang]["recommendations"])
            for r in recs:
                st.write(f"• {r}")
            st.subheader(T[lang]["export"])
            csv_df = build_scores_dataframe(company_selection, start_date, end_date, inds["scores"]) 
            csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
            st.download_button(T[lang]["download_csv"], csv_bytes, file_name="scores.csv", mime="text/csv")
            limitations = [
                "Decision-support only, not predictive.",
                "No forecast or guarantee of future performance.",
                "No financial/legal/investment advice.",
                "Excludes external market/competitor/news factors.",
                "Reliability depends on data quality and coverage.",
                "Cannot determine root causes beyond metric insights.",
                "Uses fixed rules and does not learn automatically.",
            ]
            pdf_bytes = make_pdf(company_selection, f"{str(start_date.date())} to {str(end_date.date())}", scoring["classification"] or "-", scoring["total"], conf["reason"], kr["positives"], kr["negatives"], inds["scores"], recs, limitations)
            st.download_button(T[lang]["download_pdf"], data=pdf_bytes, file_name="summary.pdf", mime="application/pdf")
            st.subheader(T[lang]["limitations"])
            for l in limitations:
                st.write(f"- {l}")

st.sidebar.markdown("Run: pip install -r requirements.txt")
st.sidebar.markdown("Run: streamlit run app.py")
