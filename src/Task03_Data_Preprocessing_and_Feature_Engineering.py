"""
CCS3440 Artificial Intelligence Coursework | Group 02
Option C: Disease Risk Classification - SmartCare Hospital
Task 03 – Data Preprocessing and Feature Engineering (Leakage-Free Pipeline)
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import load_and_clean_data
from feature_engineering import (
    engineer_features,
    fit_and_transform_pipeline,
    TARGET_MAP,
    TARGET_CLASSES,
)

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
    print("  (Zero Data Leakage: Split First -> Fit on Train)")
    print("==================================================")

    # 1. Load and clean raw data
    print(f"\n[1] Loading raw dataset (N=1000): {RAW_DATA_PATH}")
    df_clean = load_and_clean_data(RAW_DATA_PATH)
    print(f"Cleaned dataset shape: {df_clean.shape}")

    # 2. Save human-readable cleaned & engineered dataset for EDA
    df_feat = engineer_features(df_clean)
    cleaned_csv_path = PROCESSED_DIR / "smartcare_cleaned.csv"
    df_feat.to_csv(cleaned_csv_path, index=False)
    print(f"Saved human-readable cleaned data to: {cleaned_csv_path}")

    # 3. Stratified Train/Test Split (80% Train, 20% Held-Out Test) BEFORE Fitting Transformers
    print("\n[2] Performing Stratified 80/20 Train/Test Split on Raw Data...")
    X_raw = df_clean.drop(columns=["disease_risk_level"])
    y_raw = df_clean["disease_risk_level"]
    y_encoded = pd.Series(y_raw.map(TARGET_MAP).astype(int), name="disease_risk_level")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Train Raw Shape: {X_train_raw.shape}, y_train: {y_train.shape}")
    print(f"Test Raw Shape:  {X_test_raw.shape}, y_test: {y_test.shape}")
    print("\nTarget Class Distribution:")
    for cls_name, cls_idx in TARGET_MAP.items():
        n_train = (y_train == cls_idx).sum()
        n_test = (y_test == cls_idx).sum()
        print(f"  Class {cls_idx} ({cls_name:6s}): Train = {n_train:3d} ({n_train/len(y_train):.1%}) | Test = {n_test:2d} ({n_test/len(y_test):.1%})")

    # 4. Fit Feature Engineering, One-Hot Encoding, Selection & Scaling ONLY on Training Set
    print("\n[3] Fitting One-Hot Encoding, ANOVA F-Score Selection (K=15), & Standard Scaling strictly on Train...")
    X_train_scaled, X_test_scaled, pipeline_artifacts = fit_and_transform_pipeline(
        X_train_raw, y_train, X_test_raw, k=15
    )

    print("\nSelected Top 15 Features:")
    for rank, feat in enumerate(pipeline_artifacts["selected_features"], 1):
        score = pipeline_artifacts["feature_scores"].loc[
            pipeline_artifacts["feature_scores"]["feature"] == feat, "score"
        ].values[0]
        print(f"  {rank:2d}. {feat:32s} (ANOVA F-score: {score:.2f})")

    # 5. Save model-ready CSVs
    X_train_scaled.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test_scaled.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    print(f"\nSaved model-ready split CSVs to: {PROCESSED_DIR}")

    # 6. Save preprocessing and feature engineering pipeline artifacts
    joblib.dump(pipeline_artifacts, MODELS_DIR / "feature_artifacts.joblib")
    joblib.dump(pipeline_artifacts["scaler"], MODELS_DIR / "feature_scaler.pkl")
    joblib.dump(pipeline_artifacts["scaler"], APP_DIR / "feature_scaler.pkl")
    print(f"Saved feature artifacts bundle to: {MODELS_DIR / 'feature_artifacts.joblib'}")

    print("\n[SUCCESS] Task 03 completed successfully with zero data leakage!\n")
    return pipeline_artifacts


if __name__ == "__main__":
    run_task03()