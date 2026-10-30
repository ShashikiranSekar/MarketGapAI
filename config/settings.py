import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "olist"

def get_data_dir():
    value = os.getenv("OLIST_DATA_DIR", "").strip()
    return Path(value).expanduser() if value else DEFAULT_DATA_DIR

AUDIT_CHUNK_SIZE = int(os.getenv("AUDIT_CHUNK_SIZE", "50000"))
REQUIRED_DATASETS = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}
