import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import REQUIRED_DATASETS, get_data_dir
from src.ingestion.loader import check_dataset_files, load_dataset
from src.ingestion.audit import audit_all_datasets
from src.intelligence.sentiment import analyze_sentiment
from src.intelligence.pain_points import detect_pain_points, summarize_pain_points
from src.intelligence.semantic_clustering import cluster_reviews
from src.intelligence.market_gap_analysis import (
    attach_product_categories, summarize_category_opportunities
)
from src.intelligence.market_research import (
    build_competitor_comparison, build_market_snapshot, score_opportunities,
    make_opportunity_briefs
)
from src.rag.generation import generate_opportunity_report, get_llm_configuration_status

st.set_page_config(page_title="MarketGap AI", page_icon="📊", layout="wide")
st.title("MarketGap AI")
st.caption("Generative AI-Based Market Opportunity Discovery System")
st.write("Discover customer pain points, explore market signals, compare competitor evidence, and generate business opportunity hypotheses.")

with st.sidebar:
    st.header("Research configuration")
    industry = st.selectbox("Market / industry", ["E-commerce", "Healthcare", "Education", "Fitness", "Food delivery", "Other"])
    focus = st.text_input("Research focus", placeholder="e.g. affordable delivery, accessible fitness")
    st.caption("The included Olist data supports Brazilian e-commerce analysis. For other industries, upload a suitable dataset or provide independently verified research evidence.")
    st.divider()
    st.caption("Configured dataset directory")
    st.code(str(get_data_dir()))

tabs = st.tabs([
    "Overview", "Data Audit", "Dataset Explorer", "Customer Pain Points",
    "Market Gap Detector", "Competitor Comparison", "GenAI Opportunity Engine",
    "Testing & Deployment"
])

def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")

def get_review_analysis():
    if "pain_reviews" not in st.session_state:
        st.session_state["pain_reviews"] = detect_pain_points(analyze_sentiment(load_dataset("order_reviews")))
    return st.session_state["pain_reviews"]

def run_market_analysis(sample_size, n_clusters):
    reviews = load_dataset("order_reviews")
    clustered, cluster_summary = cluster_reviews(reviews, max_reviews=sample_size, n_clusters=n_clusters)
    orders = load_dataset("orders")
    items = load_dataset("order_items")
    products = load_dataset("products")
    translations = load_dataset("category_translation")
    mapped = attach_product_categories(clustered, orders, items, products, translations)
    category_summary = summarize_category_opportunities(mapped)
    st.session_state["semantic_result"] = (clustered, cluster_summary, category_summary)
    st.session_state["market_snapshot"] = build_market_snapshot(
        detect_pain_points(analyze_sentiment(reviews)), category_summary, focus
    )
    st.session_state["opportunity_briefs"] = make_opportunity_briefs(category_summary, cluster_summary, focus)
    return clustered, cluster_summary, category_summary

with tabs[0]:
    st.header("Market Research Dashboard")
    st.write(f"**Selected industry:** {industry}")
    st.write(f"**Research focus:** {focus or 'General opportunity discovery'}")
    if industry != "E-commerce":
        st.warning("The current source dataset is Olist e-commerce. The selected industry is a research brief only; findings are not industry-specific until relevant evidence is added.")
    st.info("Historical Olist Brazilian e-commerce data (primarily 2016–2018) supports exploratory hypotheses, not live market sizing or proof of profitability.")
    availability = check_dataset_files()
    found = sum(x["exists"] for x in availability.values())
    a,b,c = st.columns(3)
    a.metric("Expected CSV files", len(availability))
    b.metric("Files found", found)
    c.metric("Files missing", len(availability)-found)
    st.dataframe(pd.DataFrame([{"Dataset":k,**v} for k,v in availability.items()]), use_container_width=True, hide_index=True)
    st.subheader("Project capabilities")
    st.markdown("""
    - Customer review sentiment and recurring complaint themes
    - Semantic clustering of multilingual customer feedback
    - Product-category opportunity screening with transparent scores
    - Competitor feature comparison from supplied and verified evidence
    - Generative AI opportunity reports, customer profiles, MVP features, monetization hypotheses, and validation plans
    - CSV exports, audit reports, starter tests, and deployment guidance
    """)

with tabs[1]:
    st.header("Data Quality Audit")
    st.caption("CSV files are processed in chunks. Duplicate counts are within chunks and can miss duplicates across chunk boundaries.")
    if st.button("Run complete data audit", type="primary"):
        with st.spinner("Auditing all expected CSV files..."):
            st.session_state["audit_df"] = audit_all_datasets()
    audit = st.session_state.get("audit_df")
    if audit is not None:
        st.dataframe(audit, use_container_width=True, hide_index=True)
        passed = audit[audit["Status"] == "Passed"]
        a,b,c = st.columns(3)
        a.metric("Files passed", len(passed))
        b.metric("Rows across files", f'{int(passed["Rows"].fillna(0).sum()):,}')
        c.metric("Missing cells", f'{int(passed["Missing Cells"].fillna(0).sum()):,}')
        if not passed.empty:
            st.plotly_chart(px.bar(passed, x="Dataset", y="Rows", title="Records per dataset"), use_container_width=True)
        details = st.selectbox("Column-level audit details", ["None"] + passed["Dataset"].tolist())
        if details != "None":
            from src.ingestion.audit import audit_dataset_file
            detail = audit_dataset_file(details)
            st.dataframe(pd.DataFrame([{"Column":k,**v} for k,v in detail["column_details"].items()]), use_container_width=True, hide_index=True)
        st.download_button("Download audit report", csv_bytes(audit), "marketgap_audit.csv", "text/csv")

with tabs[2]:
    st.header("Dataset Explorer")
    dataset = st.selectbox("Choose dataset", list(REQUIRED_DATASETS.keys()), format_func=lambda k:f"{k} — {REQUIRED_DATASETS[k]}")
    limit = st.selectbox("Preview rows", [10,25,50,100], index=0)
    if st.button("Load dataset preview"):
        try:
            st.session_state["preview"] = (dataset, load_dataset(dataset, nrows=limit))
        except Exception as exc:
            st.error(str(exc))
    preview = st.session_state.get("preview")
    if preview and preview[0] == dataset:
        st.dataframe(preview[1], use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame({"Column":preview[1].columns, "Inferred type":preview[1].dtypes.astype(str)}), use_container_width=True, hide_index=True)

with tabs[3]:
    st.header("Customer Pain Point Analyzer")
    st.caption("Sentiment uses rating labels; pain-point extraction uses transparent keyword rules. These are baseline methods, not a trained sentiment model.")
    if st.button("Analyze customer reviews", type="primary"):
        try:
            with st.spinner("Analyzing review ratings and text..."):
                st.session_state["pain_reviews"] = detect_pain_points(analyze_sentiment(load_dataset("order_reviews")))
        except Exception as exc:
            st.error(f"Review analysis failed: {exc}")
    reviews = st.session_state.get("pain_reviews")
    if reviews is not None:
        sentiment = reviews["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Reviews")
        a,b,c = st.columns(3)
        a.metric("Total reviews", f"{len(reviews):,}")
        a.metric("Negative ratings", f'{int(reviews["sentiment"].eq("Negative").sum()):,}')
        b.metric("Reviews with text", f'{int(reviews["has_comment"].sum()):,}')
        c.metric("Reviews matching pain themes", f'{int(reviews["has_pain_point"].sum()):,}')
        st.plotly_chart(px.pie(sentiment, names="Sentiment", values="Reviews", hole=.4, title="Rating-based sentiment"), use_container_width=True)
        pains = summarize_pain_points(reviews)
        if not pains.empty:
            st.plotly_chart(px.bar(pains, x="Review matches", y="Pain point", orientation="h", title="Recurring complaint themes"), use_container_width=True)
            st.dataframe(pains, use_container_width=True, hide_index=True)
        selected = st.selectbox("Review sentiment", ["All","Negative","Neutral","Positive","Unknown"])
        filtered = reviews if selected == "All" else reviews[reviews["sentiment"] == selected]
        st.dataframe(filtered[[x for x in ["review_score","sentiment","review_text","pain_points"] if x in filtered]].head(250), use_container_width=True, hide_index=True)
        st.download_button("Export analyzed reviews", csv_bytes(reviews), "marketgap_reviews.csv", "text/csv")

with tabs[4]:
    st.header("Competitor and Market Gap Detector")
    st.write("Run semantic clustering to connect recurring review themes with product categories and create a ranked list of opportunities to investigate.")
    sample = st.select_slider("Review sample size", options=[300,500,1000,2500,5000], value=1000)
    cluster_n = st.slider("Semantic clusters", min_value=3, max_value=15, value=8)
    if st.button("Run market gap analysis", type="primary"):
        try:
            with st.spinner("Encoding review text, clustering, and joining category evidence..."):
                run_market_analysis(sample, cluster_n)
        except Exception as exc:
            st.error(f"Market analysis failed: {exc}")
    result = st.session_state.get("semantic_result")
    if result:
        clustered, cluster_summary, categories = result
        a,b,c = st.columns(3)
        a.metric("Reviews clustered", f"{len(clustered):,}")
        b.metric("Clusters discovered", int(clustered["cluster_id"].nunique()))
        c.metric("Low-rating reviews in sample", int(clustered["is_negative"].sum()))
        st.warning("Sampling prioritizes low-rated reviews. Cluster rates are exploratory and should not be interpreted as population rates.")
        st.subheader("Cluster summaries")
        st.dataframe(cluster_summary, use_container_width=True, hide_index=True)
        chosen = st.selectbox("Inspect cluster ID", sorted(clustered["cluster_id"].unique().tolist()))
        st.dataframe(clustered.loc[clustered["cluster_id"] == chosen, [x for x in ["review_score","review_text","is_negative"] if x in clustered]].head(25), use_container_width=True, hide_index=True)
        st.subheader("Ranked category opportunity signals")
        if categories.empty:
            st.info("No category met the minimum association count.")
        else:
            ranked = score_opportunities(categories)
            st.dataframe(ranked, use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(ranked.head(15), x="opportunity_signal_score", y="category", orientation="h", title="Heuristic opportunity signal"), use_container_width=True)
            st.download_button("Export opportunity ranking", csv_bytes(ranked), "marketgap_opportunity_ranking.csv", "text/csv")
        st.download_button("Export cluster summary", csv_bytes(cluster_summary), "marketgap_cluster_summary.csv", "text/csv")
        briefs = st.session_state.get("opportunity_briefs", [])
        if briefs:
            st.subheader("Opportunity briefs")
            st.dataframe(pd.DataFrame(briefs), use_container_width=True, hide_index=True)

with tabs[5]:
    st.header("Competitor Comparison")
    st.write("Olist does not provide a verified competitor feature catalog. Enter competitor data from sources you have checked. Unknown is not the same as absent.")
    st.caption("Status values: 1 = confirmed present, 0 = confirmed absent, blank = unknown.")
    template = pd.DataFrame([
        {"Competitor":"Competitor A","Feature":"Fast delivery","Status":""},
        {"Competitor":"Competitor A","Feature":"Easy returns","Status":""},
        {"Competitor":"Competitor A","Feature":"Customer support","Status":""},
        {"Competitor":"Competitor B","Feature":"Fast delivery","Status":""},
        {"Competitor":"Competitor B","Feature":"Easy returns","Status":""},
        {"Competitor":"Competitor B","Feature":"Customer support","Status":""},
    ])
    if "competitor_input" not in st.session_state:
        st.session_state["competitor_input"] = template
    edited = st.data_editor(st.session_state["competitor_input"], num_rows="dynamic", use_container_width=True, key="competitor_edit", column_config={"Status":st.column_config.SelectboxColumn("Status", options=["","1","0"])})
    st.session_state["competitor_input"] = edited
    st.download_button("Download blank competitor template", csv_bytes(template), "competitor_template.csv", "text/csv")
    if st.button("Compare competitors"):
        try:
            st.session_state["competitor_matrix"] = build_competitor_comparison(edited)
        except Exception as exc:
            st.error(str(exc))
    matrix = st.session_state.get("competitor_matrix")
    if matrix is not None:
        st.dataframe(matrix, use_container_width=True, hide_index=True)
        st.info("Verify each entry against an official product page, documentation, or another reliable source before drawing conclusions.")
        st.download_button("Export competitor comparison", csv_bytes(matrix), "competitor_comparison.csv", "text/csv")

with tabs[6]:
    st.header("Generative AI Opportunity Engine")
    st.write("Build a structured opportunity report from current review, category, and competitor evidence.")
    if not get_llm_configuration_status():
        st.warning("Gemini API key is not configured. Add GEMINI_API_KEY to .env locally or to your cloud host's secrets.")
    if st.button("Generate opportunity report", type="primary"):
        evidence = []
        pain = st.session_state.get("pain_reviews")
        semantic = st.session_state.get("semantic_result")
        competitor = st.session_state.get("competitor_matrix")
        if pain is not None:
            evidence.append("Sentiment counts:\n" + pain["sentiment"].value_counts().to_string())
            evidence.append("Complaint themes:\n" + summarize_pain_points(pain).to_string(index=False))
        if semantic:
            _, clusters, categories = semantic
            evidence.append("Semantic clusters:\n" + clusters.to_string(index=False))
            evidence.append("Category signals:\n" + categories.head(20).to_string(index=False))
        if competitor is not None:
            evidence.append("User-entered competitor matrix (not independently verified):\n" + competitor.to_string(index=False))
        if not evidence:
            st.warning("Run customer pain point or market-gap analysis first.")
        else:
            try:
                with st.spinner("Generating report..."):
                    st.session_state["genai_report"] = generate_opportunity_report("\n\n".join(evidence), focus=focus, industry=industry)
            except Exception as exc:
                st.error(f"GenAI report failed: {exc}")
    report = st.session_state.get("genai_report")
    if report:
        st.markdown(report)
        st.download_button("Download opportunity report", report.encode("utf-8"), "marketgap_opportunity_report.md", "text/markdown")

with tabs[7]:
    st.header("Testing & Cloud Delivery")
    st.markdown("""
    **Windows setup**
    ```powershell
    py -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1
    pip install -r requirements.txt
    python -m pytest
    python -m streamlit run app.py
    ```
    **Deployment checklist**
    1. Push source code to a Git repository; do not commit CSVs or secrets.
    2. Configure `OLIST_DATA_DIR` to cloud-accessible storage.
    3. Configure `GEMINI_API_KEY` in the host's secret manager.
    4. Check memory and startup time for the multilingual embedding model.
    5. Run smoke checks with real data before sharing the deployment URL.
    """)
    st.write("Dataset directory exists:", get_data_dir().exists())
    st.write("Gemini key configured:", get_llm_configuration_status())
    if st.button("Run smoke checks"):
        rows = [{"Check":f"Dataset file: {k}","Passed":v["exists"],"Details":v["filename"]} for k,v in check_dataset_files().items()]
        rows.append({"Check":"Gemini API key","Passed":get_llm_configuration_status(),"Details":"Optional until GenAI report is used"})
        checks = pd.DataFrame(rows)
        st.dataframe(checks, use_container_width=True, hide_index=True)
        st.metric("Passed", f'{int(checks["Passed"].sum())}/{len(checks)}')

st.divider()
st.caption("MarketGap AI | Opportunity scores are heuristic screening signals, not validated demand, market size, or profitability.")
