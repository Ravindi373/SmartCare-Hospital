"""
CCS3440 Artificial Intelligence Coursework | Group 02
Option C: Disease Risk Classification - SmartCare Hospital
Task 05 – Machine Learning Model Development, Hyperparameter Tuning & Class-Weighting Ablation
"""

from pathlib import Path
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
    from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
APP_DIR = BASE_DIR / "app"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
APP_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CLASS_NAMES = ["Low", "Medium", "High"]


def run_task05():
    print("==================================================")
    print("  Task 05: ML Model Development & Tuning (Leak-Free)")
    print("==================================================")

    # 1. Load prepared datasets
    x_train_path = DATA_DIR / "X_train.csv"
    x_test_path = DATA_DIR / "X_test.csv"
    y_train_path = DATA_DIR / "y_train.csv"
    y_test_path = DATA_DIR / "y_test.csv"

    if not (x_train_path.exists() and y_train_path.exists()):
        print("Prepared datasets not found. Running Task 03...")
        from Task03_Data_Preprocessing_and_Feature_Engineering import run_task03
        run_task03()

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns")
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    print(f"Loaded Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Loaded Test:  X={X_test.shape}, y={y_test.shape}")
    print("\nTrain class distribution:\n", y_train.value_counts(normalize=True).sort_index().round(3))
    print("\nTest class distribution:\n", y_test.value_counts().sort_index())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tuned_models = {}
    cv_summary = []

    def tune_and_report(name, estimator, param_grid, fit_params=None):
        start = time.time()
        grid = GridSearchCV(estimator, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
        grid.fit(X_train, y_train, **(fit_params or {}))
        elapsed = time.time() - start
        best_idx = grid.best_index_
        best_std = grid.cv_results_["std_test_score"][best_idx]
        print(f"{name:25s} | CV Macro-F1 = {grid.best_score_:.4f} ± {best_std:.4f} | {elapsed:.1f}s | Params: {grid.best_params_}")
        tuned_models[name] = grid.best_estimator_
        cv_summary.append({
            "Model": name,
            "Best CV Macro-F1": grid.best_score_,
            "CV Std": best_std,
            "Best Params": str(grid.best_params_)
        })
        return grid.best_estimator_

    # -------------------------------------------------------------
    # 2. Hyperparameter Tuning for 5 Primary Model Families
    # -------------------------------------------------------------
    print("\n--- Tuning Primary Class-Weighted Model Families (Stratified 5-Fold CV) ---")

    # Model 1: Logistic Regression
    lr_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [2000],
        "class_weight": ["balanced"],
    }
    tune_and_report("Logistic Regression", LogisticRegression(random_state=RANDOM_STATE), lr_grid)

    # Model 2: Decision Tree
    dt_grid = {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_leaf": [1, 5, 10],
        "min_samples_split": [2, 10],
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced"],
    }
    tune_and_report("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE), dt_grid)

    # Model 3: Random Forest
    rf_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 3, 5],
        "class_weight": ["balanced"],
    }
    tune_and_report("Random Forest", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), rf_grid)

    # Model 4: Support Vector Machine (SVC)
    svm_grid = {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
        "class_weight": ["balanced"],
        "probability": [True],
    }
    tune_and_report("SVM", SVC(random_state=RANDOM_STATE), svm_grid)

    # Model 5: XGBoost
    xgb_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1, 0.2],
    }
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    tune_and_report(
        "XGBoost",
        XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1),
        xgb_grid,
        fit_params={"sample_weight": sample_weights}
    )

    # -------------------------------------------------------------
    # 3. Class-Weighting Ablation Study (Unweighted Baselines)
    # -------------------------------------------------------------
    print("\n--- Training Unweighted Baseline Models (Ablation Study) ---")
    ablation_models = {
        "Logistic Regression (Unweighted)": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree (Unweighted)": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest (Unweighted)": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "SVM (Unweighted)": SVC(probability=True, random_state=RANDOM_STATE),
        "XGBoost (Unweighted)": XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1),
    }

    ablation_rows = []
    for name, model in ablation_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        weighted_version = name.replace(" (Unweighted)", "")
        weighted_pred = tuned_models[weighted_version].predict(X_test)

        ablation_rows.append({
            "Model Family": weighted_version,
            "Unweighted Acc": accuracy_score(y_test, y_pred),
            "Unweighted Macro-F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "Unweighted Low-F1": f1_score(y_test, y_pred, average=None, zero_division=0)[0],
            "Weighted Acc": accuracy_score(y_test, weighted_pred),
            "Weighted Macro-F1": f1_score(y_test, weighted_pred, average="macro", zero_division=0),
            "Weighted Low-F1": f1_score(y_test, weighted_pred, average=None, zero_division=0)[0],
        })

    ablation_df = pd.DataFrame(ablation_rows)
    print("\n--- Class Weighting Ablation Impact on Held-Out Test Set ---")
    print(ablation_df.round(4).to_string(index=False))
    ablation_df.to_csv(REPORTS_DIR / "class_weighting_ablation_study.csv", index=False)

    # -------------------------------------------------------------
    # 4. Comparative Evaluation on Held-Out Test Set
    # -------------------------------------------------------------
    print("\n--- Evaluating Primary 5 Models on Held-Out Test Set ---")
    test_results = []
    predictions = {}
    probabilities = {}

    for name, model in tuned_models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            probabilities[name] = y_proba
            roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        else:
            roc_auc = np.nan

        test_results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "ROC-AUC (OvR macro)": roc_auc
        })

    test_results_df = pd.DataFrame(test_results).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
    print("\nRanked Test Set Performance:\n", test_results_df.round(4))

    best_model_name = test_results_df.iloc[0]["Model"]
    best_estimator = tuned_models[best_model_name]
    print(f"\n[BEST MODEL] Best Model Selected: {best_model_name}")
    print("\nClassification Report for Best Model:")
    print(classification_report(y_test, predictions[best_model_name], target_names=CLASS_NAMES, digits=4))

    # Confusion matrix plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for i, (name, model) in enumerate(tuned_models.items()):
        cm = confusion_matrix(y_test, predictions[name])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
        axes[i].set_title(f"{name}\n(F1={test_results_df.loc[test_results_df['Model']==name, 'F1 (macro)'].values[0]:.3f})")
    axes[-1].axis("off")
    plt.suptitle("Confusion Matrices — All 5 Models (Held-Out Test Set)", fontsize=14)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrices_all_models.png", dpi=120)
    plt.close()

    # -------------------------------------------------------------
    # 5. Serialize Artifacts & Unified Pipeline Bundle
    # -------------------------------------------------------------
    feature_artifacts_path = MODELS_DIR / "feature_artifacts.joblib"
    if feature_artifacts_path.exists():
        pipeline_bundle = joblib.load(feature_artifacts_path)
    else:
        from Task03_Data_Preprocessing_and_Feature_Engineering import run_task03
        pipeline_bundle = run_task03()

    pipeline_bundle["best_model_name"] = best_model_name
    pipeline_bundle["best_model"] = best_estimator
    pipeline_bundle["all_models"] = tuned_models
    pipeline_bundle["test_results"] = test_results_df
    pipeline_bundle["ablation_results"] = ablation_df

    joblib.dump(best_estimator, MODELS_DIR / "best_model.pkl")
    joblib.dump(best_estimator, APP_DIR / "disease_risk_model.pkl")
    joblib.dump(tuned_models, MODELS_DIR / "all_tuned_models.pkl")
    joblib.dump(pipeline_bundle, MODELS_DIR / "pipeline_bundle.joblib")
    joblib.dump(pipeline_bundle, APP_DIR / "pipeline_bundle.joblib")
    test_results_df.to_csv(REPORTS_DIR / "task05_model_comparison_results.csv", index=False)

    print(f"\nSaved models and unified pipeline bundle to:\n  - {MODELS_DIR / 'pipeline_bundle.joblib'}\n  - {APP_DIR / 'pipeline_bundle.joblib'}")
    print("\n[SUCCESS] Task 05 completed successfully!\n")


if __name__ == "__main__":
    run_task05()
