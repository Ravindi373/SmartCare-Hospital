"""
CCS3440 Artificial Intelligence Coursework | Group 02
Option C: Disease Risk Classification - SmartCare Hospital
Task 08 – AI Model Artefact & Prototype Decision Support Module
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler

from feature_engineering import transform_single_patient, TARGET_MAP, TARGET_CLASSES

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
APP_DIR = BASE_DIR / "app"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
APP_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PROTOTYPE_5_FEATURES = ["blood_sugar_mg_dl", "cholesterol_mg_dl", "age", "bmi", "systolic_bp"]
LABEL_NAMES = ["Low", "Medium", "High"]


def evaluate_and_export_prototype_model():
    """
    Perform held-out test evaluation on the lightweight 5-feature model
    to benchmark generalisation performance against the full 15-feature pipeline.
    """
    print("==================================================")
    print("  Task 08: Model Artefact & Prototype Evaluation")
    print("==================================================")

    # 1. Load raw split data to extract clean 5-feature inputs
    df_raw = pd.read_csv(DATA_DIR / "raw" / "smartcare_ai_dataset_1000.csv")
    y_raw = df_raw["disease_risk_level"].map(TARGET_MAP).astype(int)

    from sklearn.model_selection import train_test_split
    X_proto_raw = df_raw[PROTOTYPE_5_FEATURES]

    X_train_p, X_test_p, y_train, y_test = train_test_split(
        X_proto_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )

    print(f"\n[1] Evaluating 5-Feature Lightweight Model on Held-Out Test Set (N=200)...")
    print(f"Features: {PROTOTYPE_5_FEATURES}")

    # Scale strictly on train
    scaler_proto = StandardScaler()
    X_train_p_scaled = scaler_proto.fit_transform(X_train_p)
    X_test_p_scaled = scaler_proto.transform(X_test_p)

    # Fit lightweight Logistic Regression model
    lr_proto = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
    lr_proto.fit(X_train_p_scaled, y_train)

    y_pred_proto = lr_proto.predict(X_test_p_scaled)

    acc = accuracy_score(y_test, y_pred_proto)
    prec = precision_score(y_test, y_pred_proto, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred_proto, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred_proto, average="macro", zero_division=0)

    print(f"\n5-Feature Model Held-Out Test Metrics:")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Precision (macro): {prec:.4f}")
    print(f"  Recall (macro):    {rec:.4f}")
    print(f"  Macro-F1:          {f1:.4f}")

    print("\nClassification Report (5-Feature Model):")
    print(classification_report(y_test, y_pred_proto, target_names=LABEL_NAMES, digits=4))

    # Benchmark vs Full 15-Feature Pipeline
    pipeline_bundle = joblib.load(MODELS_DIR / "pipeline_bundle.joblib")
    best_15_model = pipeline_bundle["best_model"]
    best_15_name = pipeline_bundle["best_model_name"]
    X_test_15 = pd.read_csv(DATA_DIR / "processed" / "X_test.csv")
    y_pred_15 = best_15_model.predict(X_test_15)

    comparison_df = pd.DataFrame([
        {
            "Architecture": f"Full 15-Feature Pipeline ({best_15_name})",
            "Features Count": 15,
            "Accuracy": accuracy_score(y_test, y_pred_15),
            "Precision (macro)": precision_score(y_test, y_pred_15, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_test, y_pred_15, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_test, y_pred_15, average="macro", zero_division=0),
        },
        {
            "Architecture": "Lightweight 5-Feature Model (Logistic Regression)",
            "Features Count": 5,
            "Accuracy": acc,
            "Precision (macro)": prec,
            "Recall (macro)": rec,
            "F1 (macro)": f1,
        }
    ])

    print("\n--- Generalisation Trade-Off Comparison ---")
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(REPORTS_DIR / "task08_prototype_evaluation.csv", index=False)

    # 2. Export artefacts
    joblib.dump(lr_proto, MODELS_DIR / "disease_risk_model.pkl")
    joblib.dump(scaler_proto, MODELS_DIR / "feature_scaler.pkl")
    joblib.dump(lr_proto, APP_DIR / "disease_risk_model.pkl")
    joblib.dump(scaler_proto, APP_DIR / "feature_scaler.pkl")

    # Update pipeline bundle with prototype metadata
    pipeline_bundle["prototype_5_model"] = lr_proto
    pipeline_bundle["prototype_5_scaler"] = scaler_proto
    pipeline_bundle["prototype_5_features"] = PROTOTYPE_5_FEATURES
    joblib.dump(pipeline_bundle, MODELS_DIR / "pipeline_bundle.joblib")
    joblib.dump(pipeline_bundle, APP_DIR / "pipeline_bundle.joblib")

    print(f"\nSaved deployment-ready artefacts to {MODELS_DIR} and {APP_DIR}")


def predict_patient_risk(raw_patient: dict, bundle=None) -> dict:
    """
    Given a raw patient dictionary, transforms and scales all features,
    and returns predicted risk level and class probabilities using the full pipeline.
    """
    if bundle is None:
        bundle = joblib.load(MODELS_DIR / "pipeline_bundle.joblib")

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


def run_sample_verification():
    print("\n--- Running Sample Patient Profile Predictions ---")
    bundle = joblib.load(MODELS_DIR / "pipeline_bundle.joblib")

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
            "name": "Profile 3: Elderly Admitted Patient with High Biomarkers (Expected High Risk)",
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

    for item in test_profiles:
        res = predict_patient_risk(item["data"], bundle)
        print(f"\n{item['name']}:")
        print(f"  Predicted Risk: {res['prediction']}")
        print(f"  Probabilities:  Low: {res['probabilities']['Low']:.1%}, Medium: {res['probabilities']['Medium']:.1%}, High: {res['probabilities']['High']:.1%}")

    print("\n[SUCCESS] Task 08 prototype evaluation completed successfully!\n")


if __name__ == "__main__":
    evaluate_and_export_prototype_model()
    run_sample_verification()
