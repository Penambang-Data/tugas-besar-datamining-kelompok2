# src/model.py
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
import xgboost as xgb
from src import config

def train_and_save_model(X_train, y_train, model_name: str):
    """Melatih model berdasarkan nama, lalu menyimpannya."""
    print(f"\n--- Melatih Model {model_name} ---")
    
    if model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE, class_weight='balanced', n_jobs=-1)
    elif model_name == "LogisticRegression":
        model = LogisticRegression(random_state=config.RANDOM_STATE, class_weight='balanced')
    elif model_name == "KNeighbors":
        model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    elif model_name == "LightGBM":
        model = lgb.LGBMClassifier(random_state=config.RANDOM_STATE, class_weight='balanced')
    elif model_name == "XGBoost":
        # Menghitung scale_pos_weight untuk handle class imbalance
        scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
        model = xgb.XGBClassifier(random_state=config.RANDOM_STATE, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    else:
        raise ValueError(f"Model '{model_name}' tidak dikenal.")

    model.fit(X_train, y_train)
    save_path = os.path.join(config.MODELS_PATH, f"{model_name.lower()}.joblib")
    joblib.dump(model, save_path)
    print(f"Model {model_name} selesai dilatih dan disimpan.")
    return model