# src/utils.py
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
from src import config

# --- FUNGSI PREPROCESSING (DIPERBARUI) ---
def preprocess_data(df: pd.DataFrame) -> tuple:
    """Membersihkan, rekayasa fitur, memisahkan, dan menormalkan data."""
    print("Memulai preprocessing data...")
    # Hapus baris dengan NaN pada kolom-kolom penting
    cols_to_check_na = ['la', 'ld', 'nf', 'ent', 'buggy', 'aexp', 'arexp', 'asexp', 'nd', 'ns']
    df_processed = df.dropna(subset=cols_to_check_na).copy()

    # Feature Engineering
    df_processed['author_date'] = pd.to_datetime(df_processed['author_date'], errors='coerce')
    df_processed.dropna(subset=['author_date'], inplace=True)
    df_processed['code_churn'] = df_processed['la'] + df_processed['ld']
    df_processed['fix_int'] = df_processed['fix'].astype(int)
    df_processed['buggy'] = df_processed['buggy'].astype(bool)

    X = df_processed[config.FEATURES]
    y = df_processed[config.TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Preprocessing selesai. Menggunakan {len(config.FEATURES)} fitur.")
    return X_train_scaled, X_test_scaled, y_train, y_test

# --- FUNGSI EDA (BARU) ---
def run_eda_visualizations(df: pd.DataFrame):
    """Menjalankan dan menampilkan visualisasi EDA utama."""
    print("\n--> Menampilkan visualisasi Exploratory Data Analysis (EDA)...")
    
    # DAFTAR FITUR YANG ADA DI DATA MENTAH (TANPA 'code_churn' dan 'fix_int')
    raw_features_for_eda = ['la', 'ld', 'nf', 'ent', 'nd', 'ns', 'aexp', 'arexp', 'asexp']
    
    # Pastikan semua kolom ini ada sebelum melanjutkan
    existing_features = [col for col in raw_features_for_eda if col in df.columns]
    
    X = df[existing_features]
    y = df[config.TARGET]

    # Histogram
    X.hist(bins=30, figsize=(15, 10), layout=(3, 3))
    plt.suptitle("Histogram Distribusi Fitur", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
    
    # Heatmap Korelasi
    plt.figure(figsize=(12, 10))
    correlation_data = X.copy()
    correlation_data['buggy'] = y.astype(int)
    sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Heatmap Korelasi Antar Fitur", fontsize=16); plt.show()

# --- FUNGSI EVALUASI (DIPERBARUI) ---
def evaluate_models_text(models_dict: dict, X_test, y_test):
    """Mencetak laporan klasifikasi untuk semua model."""
    print("\n\n--- B. Laporan Lengkap Kinerja Model ---")
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, target_names=['Not Buggy (0)', 'Buggy (1)']))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

def plot_comparison_results(models_dict: dict, X_test, y_test):
    """Membuat dan menyimpan plot perbandingan untuk semua model."""
    print("\n\n--- C. Visualisasi Hasil Evaluasi Perbandingan ---")
    # Plot Perbandingan Confusion Matrix
    fig_cm, axes_cm = plt.subplots(2, 3, figsize=(18, 10))
    axes_cm = axes_cm.flatten()
    fig_cm.suptitle('Perbandingan Confusion Matrix', fontsize=20)
    for i, (name, model) in enumerate(models_dict.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[i])
        axes_cm[i].set_title(name)
        axes_cm[i].set_xlabel('Predicted'); axes_cm[i].set_ylabel('Actual')
    for j in range(len(models_dict), len(axes_cm)): fig_cm.delaxes(axes_cm[j])
    cm_path = os.path.join(config.PLOTS_PATH, "all_confusion_matrix.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(cm_path); plt.show()
    print(f"Plot perbandingan Confusion Matrix disimpan di: {cm_path}")

    # Plot Perbandingan ROC Curve
    fig_roc, ax_roc = plt.subplots(figsize=(12, 9))
    ax_roc.plot([0, 1], [0, 1], 'r--', label='No Skill')
    for name, model in models_dict.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax_roc)
    ax_roc.set_title('Perbandingan ROC Curve dari Lima Model'); ax_roc.legend(); ax_roc.grid(True)
    roc_path = os.path.join(config.PLOTS_PATH, "all_roc_curve.png")
    plt.savefig(roc_path); plt.show()
    print(f"Plot perbandingan ROC Curve disimpan di: {roc_path}")