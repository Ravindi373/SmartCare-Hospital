"""
CCS3440 - Task 05: Model Training (Option C - Disease Risk Classification)
Trains Logistic Regression, Random Forest, and XGBoost on the disease risk
target, then saves the best model + preprocessing artifacts used by the
Streamlit prototype (app/app.py).

Usage:
    python train_model.py
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from feature_engineering import build_features
from preprocessing import load_and_clean_data


def train_all_models(X: pd.DataFrame, y: pd.Series, numeric_cols: list, random_state: int = 42):
    """Train Logistic Regression, Random Forest, and XGBoost. Returns dict of fitted models + splits."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    models = {}

    # Logistic Regression — needs scaled features; scikit-learn auto-selects
    # a multinomial objective for multi-class targets
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    lr.fit(X_train_scaled, y_train)
    models["Logistic Regression"] = (lr, X_test_scaled)

    # Random Forest — tree-based, scaling not required
    rf_params = {"n_estimators": [100, 200], "max_depth": [5, 10, None]}
    rf_grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=random_state),
        rf_params, cv=3, scoring="f1_macro")
    rf_grid.fit(X_train, y_train)
    models["Random Forest"] = (rf_grid.best_estimator_, X_test)

    # XGBoost — native multi-class support
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         objective="multi:softprob", num_class=3,
                         eval_metric="mlogloss", random_state=random_state)
    xgb.fit(X_train, y_train)
    models["XGBoost"] = (xgb, X_test)

    return models, scaler, X_train, X_test, y_train, y_test


def save_artifacts(best_model, scaler, feature_columns, encoding_maps, out_dir="../models"):
    joblib.dump(best_model, f"{out_dir}/best_model_lr_optionC.pkl")
    joblib.dump(scaler, f"{out_dir}/scaler_optionC.pkl")
    joblib.dump(feature_columns, f"{out_dir}/feature_columns_optionC.pkl")
    joblib.dump(encoding_maps, f"{out_dir}/encoding_maps_optionC.pkl")
    print(f"Artifacts saved to {out_dir}/")


if __name__ == "__main__":
    df_clean = load_and_clean_data("../data/raw/smartcare_ai_dataset_1000.csv")
    X, y, feature_columns, encoding_maps = build_features(df_clean)
    numeric_cols = encoding_maps["numeric_cols"]

    models, scaler, X_train, X_test, y_train, y_test = train_all_models(X, y, numeric_cols)

    # Logistic Regression was found to be the best model by macro F1 (see
    # evaluate_model.py) — unlike Option B, this is a genuine result, not a
    # degenerate one, so it is saved as the prototype's model.
    best_model, _ = models["Logistic Regression"]
    save_artifacts(best_model, scaler, feature_columns, encoding_maps)

    joblib.dump(
        {"models": models, "y_test": y_test},
        "../models/_eval_cache_optionC.pkl")
    print("Training complete.")
