"""
CCS3440 Artificial Intelligence Coursework | Group 02
Option C: Disease Risk Classification - SmartCare Hospital
Task 07 – Explainable AI Analysis (SHAP) using Saved Best Pipeline Model
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Low", "Medium", "High"]


def run_task07():
    print("==================================================")
    print("  Task 07: Explainable AI Analysis (SHAP)")
    print("==================================================")

    # 1. Load train & held-out test dataset + pipeline bundle
    x_train_path = DATA_DIR / "X_train.csv"
    x_test_path = DATA_DIR / "X_test.csv"
    y_test_path = DATA_DIR / "y_test.csv"
    bundle_path = MODELS_DIR / "pipeline_bundle.joblib"

    if not (x_test_path.exists() and bundle_path.exists()):
        from Task05_Model_Development import run_task05
        run_task05()

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")
    bundle = joblib.load(bundle_path)

    best_model_name = bundle.get("best_model_name", "Logistic Regression")
    best_model = bundle.get("best_model")
    feature_names = X_test.columns.tolist()

    print(f"Using saved best model for SHAP explanations: {best_model_name}")
    print(f"Test set shape for SHAP: X={X_test.shape}, y={y_test.shape}")

    # 2. Select appropriate fast SHAP Explainer
    print("Computing SHAP values on test set...")
    if isinstance(best_model, LogisticRegression) or hasattr(best_model, "coef_"):
        explainer = shap.LinearExplainer(best_model, X_train)
        shap_values = explainer.shap_values(X_test)
        shap_array = np.array(shap_values)
    elif "Forest" in best_model_name or "Tree" in best_model_name or "XGB" in best_model_name:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        shap_array = np.array(shap_values)
    else:
        explainer = shap.Explainer(best_model, X_train)
        shap_explanation = explainer(X_test)
        shap_array = shap_explanation.values
        shap_values = shap_array

    # 3. Overall Feature Importance (Mean |SHAP|)
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif shap_array.ndim == 3:
        mean_abs = np.abs(shap_array).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_array).mean(axis=0)

    importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    print("\nTop 10 Clinical Features Driving Risk Classification (SHAP):")
    for r, (feat, val) in enumerate(importance.head(10).items(), 1):
        print(f"  {r:2d}. {feat:35s} (Mean |SHAP|: {val:.4f})")

    importance.to_csv(REPORTS_DIR / "shap_feature_importance.csv")

    # 4. Multi-class Summary Plot
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values, X_test, class_names=CLASS_NAMES, show=False)
    elif shap_array.ndim == 3:
        sv_list = [shap_array[:, :, i] for i in range(shap_array.shape[2])]
        shap.summary_plot(sv_list, X_test, class_names=CLASS_NAMES, show=False)
    else:
        shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary_multiclass.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 5. Feature Importance for High-Risk Category
    high_idx = CLASS_NAMES.index("High")
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[high_idx], X_test, plot_type="bar", show=False)
    elif shap_array.ndim == 3:
        shap.summary_plot(shap_array[:, :, high_idx], X_test, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Key Feature Drivers for High-Risk Classification", fontsize=12)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_high_risk_importance.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 6. Single Patient Waterfall Plot (High-Risk Patient Case)
    high_risk_indices = np.where(y_test == high_idx)[0]
    patient_test_idx = int(high_risk_indices[0]) if len(high_risk_indices) > 0 else 0

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)) and len(base_val) > high_idx:
        base_val_patient = float(base_val[high_idx])
    elif isinstance(base_val, (list, np.ndarray)):
        base_val_patient = float(base_val[0])
    else:
        base_val_patient = float(base_val)

    if isinstance(shap_values, list):
        sv_patient = shap_values[high_idx][patient_test_idx]
    elif shap_array.ndim == 3:
        sv_patient = shap_array[patient_test_idx, :, high_idx]
    else:
        sv_patient = shap_array[patient_test_idx]

    exp = shap.Explanation(
        values=sv_patient,
        base_values=base_val_patient,
        data=X_test.iloc[patient_test_idx].values,
        feature_names=feature_names
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_waterfall_patient_example.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\n[SUCCESS] Task 07 completed! SHAP visualizations saved to: {REPORTS_DIR}\n")


if __name__ == "__main__":
    run_task07()
