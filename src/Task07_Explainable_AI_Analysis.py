"""
CCS3440 Artificial Intelligence Coursework | Group 02
Option C: Disease Risk Classification - SmartCare Hospital
Task 07 – Explainable AI Analysis (SHAP)
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from preprocessing import load_and_clean_data
from feature_engineering import fit_feature_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "smartcare_ai_dataset_1000.csv"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["Low", "Medium", "High"]


def run_task07():
    print("==================================================")
    print("  Task 07: Explainable AI Analysis (SHAP)")
    print("==================================================")

    # 1. Load data and extract scaled features
    df_clean = load_and_clean_data(DATA_PATH)
    X_scaled, y_encoded, artifacts = fit_feature_pipeline(df_clean, k=15)
    feature_names = artifacts["selected_features"]

    # 2. Train an XGBoost model for fast TreeExplainer SHAP computation
    print("Training XGBoost model for SHAP analysis...")
    xgb_model = XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        random_state=42
    )
    xgb_model.fit(X_scaled, y_encoded)
    joblib.dump(xgb_model, MODELS_DIR / "xgb_shap_model.pkl")

    # 3. Compute SHAP Values
    print("Computing TreeExplainer SHAP values...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_scaled)
    shap_array = np.array(shap_values)

    # Overall feature importance
    if shap_array.ndim == 3:
        mean_abs = np.abs(shap_array).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_array).mean(axis=0)

    importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    print("\nTop 10 Clinical Features Driving Risk Classification (SHAP):")
    for r, (feat, val) in enumerate(importance.head(10).items(), 1):
        print(f"  {r:2d}. {feat:25s} (Mean |SHAP|: {val:.4f})")

    importance.to_csv(REPORTS_DIR / "shap_feature_importance.csv")

    # 4. Multi-class summary plot
    plt.figure()
    if shap_array.ndim == 3:
        sv_list = [shap_array[:, :, i] for i in range(shap_array.shape[2])]
        shap.summary_plot(sv_list, X_scaled, class_names=LABEL_NAMES, show=False)
    else:
        shap.summary_plot(shap_values, X_scaled, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary_multiclass.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 5. Feature Importance for High-Risk Category
    high_idx = LABEL_NAMES.index("High")
    plt.figure()
    if shap_array.ndim == 3:
        shap.summary_plot(shap_array[:, :, high_idx], X_scaled, plot_type="bar", show=False)
    plt.title("Key Feature Drivers for High-Risk Classification")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_high_risk_importance.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 6. Single Patient Waterfall Plot (High-Risk patient case)
    high_risk_patient_idx = int(np.where(y_encoded == high_idx)[0][0])
    base_val = explainer.expected_value[high_idx] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value
    sv_patient = shap_array[high_risk_patient_idx, :, high_idx] if shap_array.ndim == 3 else shap_values[high_risk_patient_idx]

    exp = shap.Explanation(
        values=sv_patient,
        base_values=base_val,
        data=X_scaled.iloc[high_risk_patient_idx],
        feature_names=feature_names
    )
    plt.figure()
    shap.plots.waterfall(exp, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_waterfall_patient_example.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\n[SUCCESS] Task 07 completed! SHAP visualizations saved to: {REPORTS_DIR}\n")


if __name__ == "__main__":
    run_task07()
