"""
CCS3440 Artificial Intelligence Coursework | Group 02
Option C: Disease Risk Classification - SmartCare Hospital
Task 03 – Data Preprocessing and Feature Engineering
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import shared modules
from preprocessing import load_and_clean_data
from feature_engineering import engineer_features, fit_feature_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "smartcare_ai_dataset_1000.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
APP_DIR = BASE_DIR / "app"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
APP_DIR.mkdir(parents=True, exist_ok=True)


def run_task03():
    print("==================================================")
    print("  Task 03: Data Preprocessing & Feature Engineering")
    print("==================================================")

    # 1. Load and clean raw data
    print(f"\n[1] Loading raw dataset: {RAW_DATA_PATH}")
    df_clean = load_and_clean_data(RAW_DATA_PATH)
    print(f"Cleaned dataset shape: {df_clean.shape}")

    # 2. Engineer features for exploratory/human-readable analysis
    df_feat = engineer_features(df_clean)
    cleaned_csv_path = PROCESSED_DIR / "smartcare_cleaned.csv"
    df_feat.to_csv(cleaned_csv_path, index=False)
    print(f"Saved human-readable cleaned & engineered data to: {cleaned_csv_path}")

    # 3. Fit full feature engineering & scaling pipeline (Select K=15 Best Features)
    print("\n[2] Executing Categorical Encoding, Feature Selection & Standard Scaling...")
    X_scaled, y_encoded, pipeline_artifacts = fit_feature_pipeline(df_clean, k=15)

    print(f"\nSelected Top 15 Features:")
    for rank, feat in enumerate(pipeline_artifacts["selected_features"], 1):
        score = pipeline_artifacts["feature_scores"].loc[
            pipeline_artifacts["feature_scores"]["feature"] == feat, "score"
        ].values[0]
        print(f"  {rank:2d}. {feat:25s} (ANOVA F-score: {score:.2f})")

    # 4. Stratified Train/Test Split (80/20)
    print("\n[3] Splitting into 80% Train and 20% Held-Out Test Set (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Train Shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test Shape:  X={X_test.shape}, y={y_test.shape}")

    # 5. Save model-ready CSVs
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    print(f"Saved split CSVs to: {PROCESSED_DIR}")

    # 6. Save preprocessing and feature engineering pipeline artifacts
    joblib.dump(pipeline_artifacts, MODELS_DIR / "feature_artifacts.joblib")
    joblib.dump(pipeline_artifacts["scaler"], MODELS_DIR / "feature_scaler.pkl")
    joblib.dump(pipeline_artifacts["scaler"], APP_DIR / "feature_scaler.pkl")
    print(f"Saved feature artifacts bundle to: {MODELS_DIR / 'feature_artifacts.joblib'}")

    print("\n[SUCCESS] Task 03 completed successfully!\n")
    return pipeline_artifacts


if __name__ == "__main__":
    run_task03()