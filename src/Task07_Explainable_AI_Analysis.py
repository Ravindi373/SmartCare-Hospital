"""
CCS3440 - Task 07: Explainable AI Analysis (Option C - Disease Risk Classification)
Uses SHAP (SHapley Additive exPlanations) on a trained XGBoost model to interpret
which features drive disease risk classification, at both the global (feature
importance per class) and individual (single-patient) level.

Usage:
    python explainability.py
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from feature_engineering import build_features
from preprocessing import load_and_clean_data

LABEL_NAMES = ["Low", "Medium", "High"]


def train_xgb_for_shap(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> XGBClassifier:
    """
    Train an XGBoost model for SHAP analysis.

    Note: XGBoost (not Logistic Regression) is used here even though Logistic
    Regression was selected as the prototype's model — SHAP's TreeExplainer
    gives exact, fast per-class explanations for tree models, and tree-based
    feature importance is generally easier to interpret clinically than
    linear coefficients on standardised features.
    """
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                           objective="multi:softprob", num_class=3,
                           eval_metric="mlogloss", random_state=random_state)
    model.fit(X, y)
    return model


def compute_shap_values(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values using TreeExplainer.
    Returns an array of shape (n_samples, n_features, n_classes).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return np.array(shap_values), explainer


def overall_feature_importance(shap_values: np.ndarray, feature_names: list) -> pd.Series:
    """
    Mean absolute SHAP value per feature, averaged across all samples AND
    all three classes — a single overall importance ranking.
    """
    mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    return pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)


def plot_summary(shap_values: np.ndarray, X: pd.DataFrame, out_path: str = "shap_summary_c.png"):
    """
    Multi-class SHAP summary plot. Requires a LIST of 2D arrays (one per
    class), NOT the raw 3D array — passing the 3D array directly produces
    a broken/misleading plot.
    """
    sv_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    plt.figure()
    shap.summary_plot(sv_list, X, class_names=LABEL_NAMES, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_class_importance_bar(shap_values: np.ndarray, X: pd.DataFrame, class_idx: int,
                               out_path: str = "shap_importance_bar_c.png"):
    """Feature importance bar chart for a single class (e.g. High risk)."""
    plt.figure()
    shap.summary_plot(shap_values[:, :, class_idx], X, plot_type="bar", show=False)
    plt.title(f"Feature Importance for {LABEL_NAMES[class_idx]}-Risk Classification")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_patient_waterfall(shap_values: np.ndarray, explainer, X: pd.DataFrame,
                            patient_idx: int, class_idx: int,
                            out_path: str = "shap_waterfall_example_c.png"):
    """SHAP waterfall plot explaining one patient's prediction for one class."""
    sv_patient = shap_values[patient_idx][:, class_idx]
    base = (explainer.expected_value[class_idx]
            if hasattr(explainer.expected_value, "__len__")
            else explainer.expected_value)
    exp = shap.Explanation(values=sv_patient, base_values=base,
                            data=X.iloc[patient_idx], feature_names=X.columns.tolist())
    plt.figure()
    shap.plots.waterfall(exp, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def run_full_shap_analysis(csv_path: str = "../data/raw/smartcare_ai_dataset_1000.csv",
                            model_out: str = "../models/best_model_xgboost_optionC.pkl"):
    """Full Task 07 pipeline: train XGBoost, compute SHAP, save all figures."""
    df_clean = load_and_clean_data(csv_path)
    X, y, feature_columns, encoding_maps = build_features(df_clean)

    model = train_xgb_for_shap(X, y)
    joblib.dump(model, model_out)
    print(f"XGBoost model saved to {model_out}")

    shap_values, explainer = compute_shap_values(model, X)
    print(f"SHAP values shape: {shap_values.shape}  (n_samples, n_features, n_classes)")

    importance = overall_feature_importance(shap_values, feature_columns)
    print("\nTop 10 features overall (mean |SHAP value| across all classes):")
    print(importance.head(10))

    plot_summary(shap_values, X)
    high_risk_idx = LABEL_NAMES.index("High")
    plot_class_importance_bar(shap_values, X, high_risk_idx)
    plot_patient_waterfall(shap_values, explainer, X, patient_idx=0, class_idx=high_risk_idx)

    return importance


if __name__ == "__main__":
    run_full_shap_analysis()
