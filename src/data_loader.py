# src/data_loader.py
import pandas as pd
import sys
from src import config

def load_raw_data() -> pd.DataFrame:
    """Memuat dataset mentah dari folder data/raw/."""
    try:
        print(f"Membaca data dari: {config.RAW_DATA_PATH}")
        return pd.read_csv(config.RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di '{config.RAW_DATA_PATH}'")
        sys.exit()