"""
CCS3440 - Task 08 support module (Option C - Disease Risk Classification)
Reusable single-patient prediction function. This is the same logic used
inside app/app.py, factored out so it can be imported/tested independently
of the Streamlit UI.

Usage:
    from predict import predict_disease_risk
    result = predict_disease_risk(patient_dict)
"""

import joblib
import pandas as pd

from feature_engineering import NOMINAL_COLS

MODEL_DIR = "../models"
LABEL_NAMES = ["Low", "Medium", "High"]


def load_artifacts(model_dir: str = MODEL_DIR):
    model = joblib.load(f"{model_dir}/best_model_lr_optionC.pkl")
    scaler = joblib.load(f"{model_dir}/scaler_optionC.pkl")
    feature_columns = joblib.load(f"{model_dir}/feature_columns_optionC.pkl")
    encoding_maps = joblib.load(f"{model_dir}/encoding_maps_optionC.pkl")
    return model, scaler, feature_columns, encoding_maps


def engineer_single_patient(raw: dict, encoding_maps: dict) -> dict:
    """Apply the same feature engineering used in training to one raw patient record."""
    bmi_order = encoding_maps["bmi_order"]
    age_order = encoding_maps["age_order"]

    bmi = raw["bmi"]
    if bmi < 18.5:
        bmi_cat = "Underweight"
    elif bmi < 25:
        bmi_cat = "Normal"
    elif bmi < 30:
        bmi_cat = "Overweight"
    else:
        bmi_cat = "Obese"

    age = raw["age"]
    if age <= 18:
        age_grp = "0-18"
    elif age <= 35:
        age_grp = "19-35"
    elif age <= 50:
        age_grp = "36-50"
    elif age <= 65:
        age_grp = "51-65"
    else:
        age_grp = "65+"

    high_bp_flag = int(raw["systolic_bp"] >= 140 or raw["diastolic_bp"] >= 90)
    high_sugar_flag = int(raw["blood_sugar_mg_dl"] >= 126)
    high_chol_flag = int(raw["cholesterol_mg_dl"] >= 240)

    engineered = dict(raw)
    engineered["bmi_category"] = bmi_order[bmi_cat]
    engineered["age_group"] = age_order[age_grp]
    engineered["high_bp_flag"] = high_bp_flag
    engineered["high_sugar_flag"] = high_sugar_flag
    engineered["high_chol_flag"] = high_chol_flag
    engineered["risk_flag_count"] = high_bp_flag + high_sugar_flag + high_chol_flag
    engineered["prior_utilization"] = raw["previous_admissions"] + raw["previous_appointments"]
    engineered["care_intensity"] = raw["lab_tests_count"] + raw["treatments_count"]
    engineered["missed_appointment_rate"] = (
        raw["missed_previous_appointments"] / raw["previous_appointments"]
        if raw["previous_appointments"] > 0 else 0.0
    )
    return engineered


def predict_disease_risk(raw_patient: dict, artifacts=None) -> dict:
    """
    Predict disease risk level for a single patient.

    raw_patient must contain all base fields (age, gender, blood_group,
    department, diagnosis, admitted, room_type, appointment_status,
    payment_status, payment_method, and all numeric clinical/billing fields).

    Returns {"prediction": "Low"/"Medium"/"High", "probabilities": {label: prob}}.
    """
    if artifacts is None:
        artifacts = load_artifacts()
    model, scaler, feature_columns, encoding_maps = artifacts
    numeric_cols = encoding_maps["numeric_cols"]

    engineered = engineer_single_patient(raw_patient, encoding_maps)
    df = pd.DataFrame([engineered])
    df_encoded = pd.get_dummies(df, columns=NOMINAL_COLS)
    df_final = df_encoded.reindex(columns=feature_columns, fill_value=0).astype(float)

    # Logistic Regression was trained on SCALED numeric features
    df_final[numeric_cols] = scaler.transform(df_final[numeric_cols])

    pred_idx = int(model.predict(df_final)[0])
    probs = model.predict_proba(df_final)[0]

    return {
        "prediction": LABEL_NAMES[pred_idx],
        "probabilities": {name: float(p) for name, p in zip(LABEL_NAMES, probs)},
    }


if __name__ == "__main__":
    sample_patient = {
        "age": 65, "waiting_days": 5, "previous_appointments": 3,
        "missed_previous_appointments": 0, "admitted": 1,
        "length_of_stay_days": 4, "previous_admissions": 2,
        "systolic_bp": 150, "diastolic_bp": 95, "blood_sugar_mg_dl": 160,
        "cholesterol_mg_dl": 260, "bmi": 31.0, "lab_tests_count": 4, "treatments_count": 3,
        "consultation_fee_lkr": 2000, "room_charge_lkr": 5000, "lab_charge_lkr": 3000,
        "medicine_charge_lkr": 4000, "total_bill_lkr": 14000,
        "gender": "Male", "blood_group": "A+", "department": "Cardiology",
        "diagnosis": "Hypertension", "appointment_status": "Completed",
        "room_type": "General Ward", "payment_status": "Paid", "payment_method": "Insurance",
    }
    result = predict_disease_risk(sample_patient)
    print(result)
