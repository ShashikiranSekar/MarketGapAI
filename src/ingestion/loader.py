from pathlib import Path
import pandas as pd
import streamlit as st
from config.settings import REQUIRED_DATASETS, get_data_dir

class DatasetError(RuntimeError):
    pass

def get_dataset_path(name):
    if name not in REQUIRED_DATASETS:
        raise DatasetError(f"Unknown dataset: {name}")
    return get_data_dir() / REQUIRED_DATASETS[name]

def check_dataset_files():
    base = get_data_dir()
    result = {}
    for name, filename in REQUIRED_DATASETS.items():
        path = base / filename
        exists = path.is_file()
        result[name] = {"filename": filename, "exists": exists, "size_mb": round(path.stat().st_size/(1024*1024),2) if exists else None}
    return result

@st.cache_data(show_spinner=False, max_entries=10)
def load_dataset(name, nrows=None):
    path = get_dataset_path(name)
    if not path.is_file():
        raise DatasetError(f"Missing {path.name} at {path}. Set OLIST_DATA_DIR or put the CSV in data/raw/olist.")
    try:
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, nrows=nrows, encoding="latin-1", low_memory=False)
