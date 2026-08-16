"""
CCS3440 Artificial Intelligence Coursework | Group 02
Option C: Disease Risk Classification - SmartCare Hospital
Task 03 – Data Preprocessing and Feature Engineering (Leak-Free Workflow)
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Import shared modules
from preprocessing import load_and_clean_data
from feature_engineering import engineer_features, fit_feature_pipeline, TARGET_MAP

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

    # 1. Load raw dataset & handle missing values + remove duplicates
    print(f"\n[1] Loading raw dataset & cleaning duplicates: {RAW_DATA_PATH}")
    df_clean = load_and_clean_data(RAW_DATA_PATH)
    print(f"Cleaned dataset shape: {df_clean.shape}")

    # Save human-readable engineered data for EDA (Section 4)
    df_feat_full = engineer_features(df_clean)
    cleaned_csv_path = PROCESSED_DIR / "smartcare_cleaned.csv"
    df_feat_full.to_csv(cleaned_csv_path, index=False)
    print(f"Saved human-readable cleaned & engineered data to: {cleaned_csv_path}")

    # 2. STRATIFIED TRAIN / TEST SPLIT IMMEDIATELY AFTER DUPLICATE DETECTION
    # Prevents data leakage into feature encoding, selection, and scaling.
    print("\n[2] Performing Stratified Train/Test Split (80% Train, 20% Test)...")
    df_train, df_test = train_test_split(
        df_clean,
        test_size=0.2,
        random_state=42,
        stratify=df_clean["disease_risk_level"]
    )

    print(f"Train Raw Shape: {df_train.shape}")
    print(f"Test Raw Shape:  {df_test.shape}")
    print("\nTrain target counts:\n", df_train["disease_risk_level"].value_counts())
    print("\nTest target counts:\n", df_test["disease_risk_level"].value_counts())

    # 3. Fit OneHotEncoder, SelectKBest, and StandardScaler STRICTLY on df_train
    print("\n[3] Fitting OneHotEncoder, SelectKBest (K=15), & StandardScaler ONLY on Training Data...")
    X_train, y_train, X_test, y_test, pipeline_artifacts = fit_feature_pipeline(df_train, df_test, k=15)

    print("\nSelected Top 15 Features:")
    for rank, feat in enumerate(pipeline_artifacts["selected_features"], 1):
        score_val = pipeline_artifacts["feature_scores"].loc[
            pipeline_artifacts["feature_scores"]["feature"] == feat, "score"
        ].values
        score = score_val[0] if len(score_val) > 0 else 0.0
        print(f"  {rank:2d}. {feat:35s} (ANOVA F-score: {score:.2f})")

    print(f"\nTransformed Train Shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Transformed Test Shape:  X={X_test.shape}, y={y_test.shape}")

    # 4. Save model-ready split CSVs
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    print(f"\nSaved split CSVs to: {PROCESSED_DIR}")

    # 5. Save preprocessing and feature engineering pipeline artifacts
    joblib.dump(pipeline_artifacts, MODELS_DIR / "feature_artifacts.joblib")
    joblib.dump(pipeline_artifacts["scaler"], MODELS_DIR / "feature_scaler.pkl")
    joblib.dump(pipeline_artifacts["scaler"], APP_DIR / "feature_scaler.pkl")
    print(f"Saved feature artifacts bundle to: {MODELS_DIR / 'feature_artifacts.joblib'}")

    print("\n[SUCCESS] Task 03 completed successfully!\n")
    return pipeline_artifacts


if __name__ == "__main__":
    run_task03()