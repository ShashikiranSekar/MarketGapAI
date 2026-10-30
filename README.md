# MarketGap AI — Core Implementation (Phases 1–6)

Streamlit dashboard for exploratory market opportunity discovery from Olist Brazilian e-commerce reviews and product categories.

## Implemented core features
- Industry/focus selector with explicit dataset scope warnings
- Nine-file availability check and chunked data-quality audit
- Dataset preview and inferred column types
- Rating-based sentiment and keyword complaint-theme analysis
- Multilingual sentence-transformer embeddings and MiniBatchKMeans clustering
- Review-to-product-category joins and heuristic opportunity scoring
- Competitor feature matrix from user-supplied, unverified entries
- Gemini-generated opportunity report with customer profiles, MVP features, monetization hypotheses, risks, and validation experiments
- CSV/Markdown exports, starter tests, and cloud deployment instructions

## Dataset
Put these files in `data/raw/olist/`, or set `OLIST_DATA_DIR` to the directory containing them:
`olist_customers_dataset.csv`, `olist_geolocation_dataset.csv`, `olist_order_items_dataset.csv`, `olist_order_payments_dataset.csv`, `olist_order_reviews_dataset.csv`, `olist_orders_dataset.csv`, `olist_products_dataset.csv`, `olist_sellers_dataset.csv`, `product_category_name_translation.csv`.

## Windows install and run
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest
python -m streamlit run app.py
```

## GenAI
Copy `.env.example` to `.env`, set `GEMINI_API_KEY`, and keep the key private. In cloud hosting, use the platform's secret manager instead of committing `.env`.

## Cloud deployment
This package is deployment-ready in structure, but it has not been deployed from this environment. The loader currently expects filesystem-accessible CSVs. Configure `OLIST_DATA_DIR` to a mounted volume or extend the loader for your cloud object store. Do not commit dataset files or credentials. Verify memory/cold-start constraints for downloading the embedding model and test with the actual host.

## Interpretation limits
- Olist is historical Brazil-specific e-commerce data, mainly 2016–2018; it does not provide current market size.
- Industry selector is a research brief; other industries need suitable datasets.
- Sentiment labels derive from ratings, not a trained text sentiment classifier.
- Complaint themes use keyword matching and may miss complaints not in the vocabulary.
- Cluster names are inferred by human interpretation of examples; clustering is exploratory.
- Negative-review sampling is intentionally prioritized and not population-representative.
- One order can contain multiple products, so review-to-category joins can associate one review with multiple categories.
- Opportunity scores are heuristic screening signals, not proof of market demand or profitability.
- Competitor feature statuses are manual and need source verification.
- GenAI output may be inaccurate and should be validated with customer research.

## Tests
```powershell
python -m pytest
```
