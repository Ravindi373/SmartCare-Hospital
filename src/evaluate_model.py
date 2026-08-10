"""
CCS3440 - Task 06: Model Evaluation (Option C - Disease Risk Classification)
Computes Accuracy, macro-averaged Precision/Recall/F1, per-class report, and
confusion matrices for each trained model, and identifies the best performer.

Usage:
    python evaluate_model.py
"""

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score)

LABEL_NAMES = ["Low", "Medium", "High"]


def evaluate_models(models: dict, y_test: pd.Series) -> pd.DataFrame:
    """Build a comparison table across all trained models using macro-averaged metrics."""
    rows = []
    for name, (model, X_te) in models.items():
        y_pred = model.predict(X_te)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Macro Precision": precision_score(y_test, y_pred, average="macro"),
            "Macro Recall": recall_score(y_test, y_pred, average="macro"),
            "Macro F1 Score": f1_score(y_test, y_pred, average="macro"),
        })
    return pd.DataFrame(rows).set_index("Model").round(3)


def get_confusion_matrices(models: dict, y_test: pd.Series) -> dict:
    """Return {model_name: confusion_matrix} for all models, ordered Low/Medium/High."""
    return {
        name: confusion_matrix(y_test, model.predict(X_te), labels=[0, 1, 2])
        for name, (model, X_te) in models.items()
    }


def best_model_by_macro_f1(results_df: pd.DataFrame) -> str:
    """
    Identify the best model by macro F1 Score.

    Justification: disease_risk_level is moderately imbalanced (Low = 13.1%
    of records), so macro-averaging — which weights each class equally
    regardless of size — better reflects performance on the clinically
    important minority Low-risk class than raw accuracy would.
    """
    return results_df["Macro F1 Score"].idxmax()


def per_class_report(model, X_test, y_test) -> str:
    """Detailed per-class precision/recall/F1 for the given model."""
    return classification_report(y_test, model.predict(X_test), target_names=LABEL_NAMES)


if __name__ == "__main__":
    cache = joblib.load("../models/_eval_cache_optionC.pkl")
    models, y_test = cache["models"], cache["y_test"]

    results_df = evaluate_models(models, y_test)
    print("Model comparison:\n", results_df)

    best = best_model_by_macro_f1(results_df)
    print(f"\nBest model by Macro F1 Score: {best}")

    best_model, best_Xte = models[best]
    print(f"\nPer-class report for {best}:")
    print(per_class_report(best_model, best_Xte, y_test))

    cms = get_confusion_matrices(models, y_test)
    for name, cm in cms.items():
        print(f"\n{name} confusion matrix (rows=true, cols=pred, order=Low/Medium/High):\n{cm}")
