"""
CCS3440 Artificial Intelligence Coursework | Group 02
Option C: Disease Risk Classification - SmartCare Hospital
Task 06 – Multi-Class Model Evaluation, Benchmarking & Diagnostics
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = ["Low", "Medium", "High"]


def run_task06():
    print("==================================================")
    print("  Task 06: Model Evaluation & Benchmarking")
    print("  (Harmonized with Task 05 Held-Out Test Split)")
    print("==================================================")

    # 1. Load test data
    x_test_path = DATA_DIR / "X_test.csv"
    y_test_path = DATA_DIR / "y_test.csv"
    if not (x_test_path.exists() and y_test_path.exists()):
        from Task03_Data_Preprocessing_and_Feature_Engineering import run_task03
        run_task03()

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    # 2. Load models
    models_path = MODELS_DIR / "all_tuned_models.pkl"
    if not models_path.exists():
        from Task05_Model_Development import run_task05
        run_task05()

    fitted_models = joblib.load(models_path)
    print(f"Evaluating {len(fitted_models)} tuned models on held-out test set (N={len(y_test)})...")
    print("Ground truth test class support:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  {name:6s} (Class {idx}): N = {(y_test == idx).sum()}")

    summary_rows = []
    per_class_rows = []
    predictions = {}
    probabilities = {}

    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        # Calculate prediction probabilities if supported
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            probabilities[name] = y_proba
            roc_auc_macro = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        else:
            roc_auc_macro = np.nan

        # Per-class metrics strictly ordered 0=Low, 1=Medium, 2=High
        prec, rec, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2], zero_division=0)
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            per_class_rows.append({
                "Model": name,
                "Class": cls_name,
                "Precision": prec[cls_idx],
                "Recall": rec[cls_idx],
                "F1": f1[cls_idx],
                "Support": support[cls_idx]
            })

        summary_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "ROC-AUC (OvR macro)": roc_auc_macro
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
    summary_df.insert(0, "Rank", range(1, len(summary_df) + 1))
    per_class_df = pd.DataFrame(per_class_rows)

    print("\n--- Model Comparison Benchmark Table ---")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(REPORTS_DIR / "task06_model_comparison_table.csv", index=False)
    per_class_df.to_csv(REPORTS_DIR / "task06_per_class_metrics.csv", index=False)

    best_model_name = summary_df.iloc[0]["Model"]
    print(f"\n[TOP MODEL] Top Model: {best_model_name}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions[best_model_name], target_names=CLASS_NAMES, digits=4))

    # ROC Curves for Best Model
    if best_model_name in probabilities:
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        y_score = probabilities[best_model_name]

        plt.figure(figsize=(7, 6))
        colors = ["#4C956C", "#F2A541", "#D64550"]
        for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc_cls = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f"{cls_name} Risk (AUC = {roc_auc_cls:.3f})")

        plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC = 0.500)")
        plt.xlabel("False Positive Rate", fontsize=11)
        plt.ylabel("True Positive Rate", fontsize=11)
        plt.title(f"One-vs-Rest ROC Curves — {best_model_name} (Held-Out Test Set)", fontsize=12)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "eval_roc_curves_best_model.png", dpi=120)
        plt.close()

    print(f"\n[SUCCESS] Task 06 completed! Evaluation reports saved to: {REPORTS_DIR}\n")


if __name__ == "__main__":
    run_task06()
