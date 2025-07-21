# src/config.py (LENGKAP DAN BENAR)

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path Data
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "apachejit_total.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")

# Path Output
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_PATH = os.path.join(RESULTS_DIR, "plots")
MODELS_PATH = os.path.join(RESULTS_DIR, "models")

# Otomatis buat folder jika belum ada
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# Daftar Fitur Lengkap
FEATURES = [
    'la', 'ld', 'nf', 'ent', 'code_churn', 'nd', 'ns',
    'aexp', 'arexp', 'asexp', 'fix_int'
]
TARGET = 'buggy'

# Daftar Model yang Akan Dijalankan <-- VARIABEL INI YANG SEBELUMNYA HILANG
MODELS_TO_RUN = [
    "RandomForest",
    "LogisticRegression",
    "KNeighbors",
    "LightGBM",
    "XGBoost"
]

TEST_SIZE = 0.2
RANDOM_STATE = 42