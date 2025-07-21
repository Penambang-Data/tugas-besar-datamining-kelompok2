# src/main.py
from src import config
from src.data_loader import load_raw_data
from src.utils import preprocess_data, run_eda_visualizations, evaluate_models_text, plot_comparison_results
from src.model import train_and_save_model

def main():
    """Pipeline utama proyek Data Mining."""
    print("="*60)
    print("      MEMULAI PIPELINE ANALISIS BUG PREDICTION (v2.0)")
    print("="*60)

    # Langkah 1: Muat Data Mentah
    raw_df = load_raw_data()
    
    # Langkah 2: Exploratory Data Analysis (EDA)
    run_eda_visualizations(raw_df)

    # Langkah 3: Preprocessing Data
    X_train, X_test, y_train, y_test = preprocess_data(raw_df)
    
    # Langkah 4: Latih Semua Model
    trained_models = {}
    for model_name in config.MODELS_TO_RUN:
        model = train_and_save_model(X_train, y_train, model_name)
        trained_models[model_name] = model
    
    # Langkah 5: Evaluasi Semua Model
    evaluate_models_text(trained_models, X_test, y_test)
    plot_comparison_results(trained_models, X_test, y_test)

    print("\n" + "="*60)
    print("             PIPELINE SELESAI DENGAN SUKSES")
    print("="*60)

if __name__ == "__main__":
    main()