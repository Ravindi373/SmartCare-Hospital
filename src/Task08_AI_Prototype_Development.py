"""
CCS3440 Artificial Intelligence Coursework | Group 02
Option C: Disease Risk Classification - SmartCare Hospital
Task 08 – AI Prototype Prediction Module & 5-Feature Prototype Evaluation
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from feature_engineering import transform_single_patient

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
APP_DIR = BASE_DIR / "app"
REPORTS_DIR = BASE_DIR / "reports"

BUNDLE_PATH = MODELS_DIR / "pipeline_bundle.joblib"
LABEL_NAMES = ["Low", "Medium", "High"]


def load_pipeline_bundle(bundle_path: Path = BUNDLE_PATH):
    """Load the complete pipeline bundle containing model, scaler, and encoders."""
    if not bundle_path.exists():
        from Task05_Model_Development import run_task05
        run_task05()
    return joblib.load(bundle_path)


def evaluate_prototype_model(bundle):
    """
    Train and evaluate a simplified 5-feature prototype model on held-out test data,
    comparing its generalization performance against the full 15-feature pipeline.
    """
    x_train_path = DATA_DIR / "X_train.csv"
    x_test_path = DATA_DIR / "X_test.csv"
    y_train_path = DATA_DIR / "y_train.csv"
    y_test_path = DATA_DIR / "y_test.csv"

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns")
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    # Select top 5 features
    top5_features = ["age", "blood_sugar_mg_dl", "cholesterol_mg_dl", "bmi", "previous_admissions"]
    # Check if all top5 features exist in X_train, else pick top 5 selected features
    actual_top5 = [f for f in top5_features if f in X_train.columns]
    if len(actual_top5) < 5:
        actual_top5 = X_train.columns[:5].tolist()

    X_train_5 = X_train[actual_top5]
    X_test_5 = X_test[actual_top5]

    # Train prototype model (Logistic Regression / Random Forest) on 5 features
    proto_model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    proto_model.fit(X_train_5, y_train)
    y_pred_5 = proto_model.predict(X_test_5)

    # Full model performance on test set
    best_full_model = bundle["best_model"]
    y_pred_full = best_full_model.predict(X_test)

    comparison_df = pd.DataFrame([
        {
            "Model System": "Full 15-Feature Pipeline",
            "Features Used": 15,
            "Accuracy": accuracy_score(y_test, y_pred_full),
            "Precision (macro)": precision_score(y_test, y_pred_full, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_test, y_pred_full, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_test, y_pred_full, average="macro", zero_division=0),
        },
        {
            "Model System": "5-Feature Prototype Model",
            "Features Used": 5,
            "Accuracy": accuracy_score(y_test, y_pred_5),
            "Precision (macro)": precision_score(y_test, y_pred_5, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_test, y_pred_5, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_test, y_pred_5, average="macro", zero_division=0),
        }
    ])

    print("\n--- Prototype vs Full Model Generalization Comparison (Held-Out Test Set) ---")
    print(comparison_df.round(4).to_string(index=False))

    comparison_df.to_csv(REPORTS_DIR / "task08_prototype_comparison.csv", index=False)
    joblib.dump(proto_model, MODELS_DIR / "prototype_5feature_model.pkl")
    return comparison_df


def predict_patient_risk(raw_patient: dict, bundle=None) -> dict:
    """
    Given a raw patient dictionary, transforms and scales all features using OneHotEncoder bundle,
    and returns predicted risk level and class probabilities.
    """
    if bundle is None:
        bundle = load_pipeline_bundle()

    model = bundle["best_model"]
    X_input = transform_single_patient(raw_patient, bundle)

    pred_idx = int(model.predict(X_input)[0])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
    else:
        probs = [1.0 if i == pred_idx else 0.0 for i in range(3)]

    return {
        "prediction": LABEL_NAMES[pred_idx],
        "prediction_index": pred_idx,
        "probabilities": {label: float(p) for label, p in zip(LABEL_NAMES, probs)},
        "model_used": bundle.get("best_model_name", "Logistic Regression"),
        "transformed_features": X_input.iloc[0].to_dict()
    }


def run_task08():
    print("==================================================")
    print("  Task 08: Prototype Prediction & Evaluation")
    print("==================================================")

    bundle = load_pipeline_bundle()
    print(f"Loaded Primary Model: {bundle.get('best_model_name', 'Logistic Regression')}")
    print(f"Features in Pipeline: {len(bundle.get('selected_features', []))} features\n")

    # Evaluate 5-feature prototype on held-out test data
    evaluate_prototype_model(bundle)

    # Verification profiles
    test_profiles = [
        {
            "name": "Profile 1: Healthy Young Outpatient (Expected Low Risk)",
            "data": {
                "age": 22, "gender": "Female", "blood_group": "O+", "department": "General Medicine",
                "diagnosis": "Fever", "appointment_status": "Completed", "admitted": 0, "room_type": "Not Admitted",
                "payment_status": "Paid", "payment_method": "Cash", "waiting_days": 1, "previous_appointments": 1,
                "missed_previous_appointments": 0, "length_of_stay_days": 0, "previous_admissions": 0,
                "systolic_bp": 115, "diastolic_bp": 75, "blood_sugar_mg_dl": 85, "cholesterol_mg_dl": 160,
                "bmi": 21.5, "lab_tests_count": 1, "treatments_count": 1, "consultation_fee_lkr": 2000,
                "room_charge_lkr": 0, "lab_charge_lkr": 1500, "medicine_charge_lkr": 1000
            }
        },
        {
            "name": "Profile 2: Middle-Aged Patient with Mild Hypertension (Expected Medium Risk)",
            "data": {
                "age": 48, "gender": "Male", "blood_group": "A+", "department": "Cardiology",
                "diagnosis": "Hypertension", "appointment_status": "Completed", "admitted": 0, "room_type": "Not Admitted",
                "payment_status": "Paid", "payment_method": "Card", "waiting_days": 4, "previous_appointments": 3,
                "missed_previous_appointments": 1, "length_of_stay_days": 0, "previous_admissions": 1,
                "systolic_bp": 135, "diastolic_bp": 85, "blood_sugar_mg_dl": 115, "cholesterol_mg_dl": 205,
                "bmi": 26.2, "lab_tests_count": 2, "treatments_count": 2, "consultation_fee_lkr": 2500,
                "room_charge_lkr": 0, "lab_charge_lkr": 3000, "medicine_charge_lkr": 3500
            }
        },
        {
            "name": "Profile 3: Elderly Admitted Patient with Complex Comorbidities (Expected High Risk)",
            "data": {
                "age": 72, "gender": "Male", "blood_group": "B+", "department": "Cardiology",
                "diagnosis": "Diabetes", "appointment_status": "Completed", "admitted": 1, "room_type": "ICU",
                "payment_status": "Paid", "payment_method": "Insurance", "waiting_days": 8, "previous_appointments": 6,
                "missed_previous_appointments": 2, "length_of_stay_days": 7, "previous_admissions": 4,
                "systolic_bp": 165, "diastolic_bp": 100, "blood_sugar_mg_dl": 240, "cholesterol_mg_dl": 290,
                "bmi": 33.4, "lab_tests_count": 6, "treatments_count": 5, "consultation_fee_lkr": 3000,
                "room_charge_lkr": 35000, "lab_charge_lkr": 12000, "medicine_charge_lkr": 18000
            }
        }
    ]

    print("\n--- Single Patient Prediction Verification ---")
    for item in test_profiles:
        res = predict_patient_risk(item["data"], bundle)
        print(f"--- {item['name']} ---")
        print(f"  Predicted Risk: {res['prediction']}")
        print(f"  Probabilities:  Low: {res['probabilities']['Low']:.1%}, Medium: {res['probabilities']['Medium']:.1%}, High: {res['probabilities']['High']:.1%}\n")

    print("\n[SUCCESS] Task 08 completed!\n")


if __name__ == "__main__":
    run_task08()
