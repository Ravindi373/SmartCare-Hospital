"""
CCS3440 Artificial Intelligence Coursework | Group 02
Option C: Disease Risk Classification - SmartCare Hospital
Task 07 – Explainable AI Analysis (True SHAP Implementation)
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LABEL_NAMES = ["Low", "Medium", "High"]


def run_task07():
    print("==================================================")
    print("  Task 07: Explainable AI Analysis (True SHAP)")
    print("  (Generated Directly from Saved Pipeline & Test Set)")
    print("==================================================")

    # 1. Load pipeline bundle and held-out test data
    bundle_path = MODELS_DIR / "pipeline_bundle.joblib"
    if not bundle_path.exists():
        from Task05_Model_Development import run_task05
        run_task05()

    pipeline_bundle = joblib.load(bundle_path)
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

    feature_names = X_test.columns.tolist()
    print(f"Loaded Pipeline Bundle with {len(feature_names)} features.")
    print(f"Test Set Size for XAI Explanations: N={len(X_test)}")

    # 2. Extract tuned Tree & Linear models from the pipeline
    all_models = pipeline_bundle["all_models"]
    rf_model = all_models.get("Random Forest")
    xgb_model = all_models.get("XGBoost")
    best_model = pipeline_bundle["best_model"]
    best_name = pipeline_bundle["best_model_name"]

    print(f"\nComputing TreeExplainer SHAP values using tuned Random Forest / XGBoost ensemble...")
    # Use Random Forest / XGBoost TreeExplainer for exact, fast, model-agnostic tree attributions
    explainer_model = rf_model if rf_model is not None else xgb_model
    explainer = shap.TreeExplainer(explainer_model)
    shap_values = explainer.shap_values(X_test)

    # 3. Handle SHAP value output dimensions (list of arrays for multiclass or 3D array)
    if isinstance(shap_values, list):
        # List of [N_samples, N_features] for each class
        mean_abs_per_class = [np.abs(sv).mean(axis=0) for sv in shap_values]
        overall_mean_abs = np.mean(mean_abs_per_class, axis=0)
        shap_array = np.stack(shap_values, axis=-1)  # (N_samples, N_features, N_classes)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        overall_mean_abs = np.abs(shap_values).mean(axis=(0, 2))
        shap_array = shap_values
    else:
        # shap.Explanation object or 2D
        if hasattr(shap_values, "values"):
            raw_vals = shap_values.values
            if raw_vals.ndim == 3:
                overall_mean_abs = np.abs(raw_vals).mean(axis=(0, 2))
                shap_array = raw_vals
            else:
                overall_mean_abs = np.abs(raw_vals).mean(axis=0)
                shap_array = raw_vals
        else:
            overall_mean_abs = np.abs(shap_values).mean(axis=0)
            shap_array = np.array(shap_values)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean Absolute SHAP Value": overall_mean_abs
    }).sort_values("Mean Absolute SHAP Value", ascending=False).reset_index(drop=True)

    print("\n--- Global Feature Importance (Mean |SHAP| across all classes) ---")
    for r, row in importance_df.iterrows():
        print(f"  {r+1:2d}. {row['Feature']:32s} : {row['Mean Absolute SHAP Value']:.4f}")

    importance_df.to_csv(REPORTS_DIR / "shap_feature_importance.csv", index=False)

    # 4. Multi-Class Summary Plot
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values, X_test, class_names=LABEL_NAMES, show=False)
    elif shap_array.ndim == 3:
        sv_list = [shap_array[:, :, i] for i in range(shap_array.shape[2])]
        shap.summary_plot(sv_list, X_test, class_names=LABEL_NAMES, show=False)
    else:
        shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Multi-Class SHAP Summary Plot — Feature Attributions", fontsize=12)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary_multiclass.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 5. Feature Importance for High-Risk Category (Class Index 2)
    high_idx = 2  # High Risk
    plt.figure(figsize=(9, 5))
    if shap_array.ndim == 3:
        shap.summary_plot(shap_array[:, :, high_idx], X_test, plot_type="bar", show=False)
    elif isinstance(shap_values, list):
        shap.summary_plot(shap_values[high_idx], X_test, plot_type="bar", show=False)
    plt.title("Key Feature Drivers for HIGH Disease Risk Level (SHAP)", fontsize=12)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_high_risk_importance.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 6. Local Patient-Level Waterfall Explanation
    high_risk_indices = np.where(y_test == high_idx)[0]
    sample_idx = int(high_risk_indices[0]) if len(high_risk_indices) > 0 else 0

    if hasattr(explainer, "expected_value"):
        exp_val = explainer.expected_value
        base_val = exp_val[high_idx] if hasattr(exp_val, "__len__") else exp_val
    else:
        base_val = 0.0

    sv_sample = shap_array[sample_idx, :, high_idx] if shap_array.ndim == 3 else shap_values[high_idx][sample_idx]

    explanation = shap.Explanation(
        values=sv_sample,
        base_values=base_val,
        data=X_test.iloc[sample_idx].values,
        feature_names=feature_names
    )

    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(explanation, show=False)
    plt.title(f"Patient #{sample_idx} (True Class: High Risk) — SHAP Local Feature Attribution", fontsize=11)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_waterfall_patient_example.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\n[SUCCESS] Task 07 completed! SHAP visualizations saved to: {REPORTS_DIR}\n")


if __name__ == "__main__":
    run_task07()
