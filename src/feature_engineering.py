"""
Feature Engineering Module for SmartCare Hospital AI Project
Provides clinical feature construction, categorical encoding, feature selection,
and unified single-patient inference transformation with explicit target mapping.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

BASE_DIR = Path(__file__).resolve().parent.parent

# Explicit target mapping: 0 = Low, 1 = Medium, 2 = High
TARGET_MAP = {"Low": 0, "Medium": 1, "High": 2}
INV_TARGET_MAP = {0: "Low", 1: "Medium", 2: "High"}
TARGET_CLASSES = ["Low", "Medium", "High"]

CATEGORICAL_COLS = [
    "gender", "blood_group", "department", "diagnosis", "appointment_status",
    "room_type", "payment_status", "payment_method", "age_group", "bmi_category", "bp_category"
]


def classify_bp(systolic: float, diastolic: float) -> str:
    """Classify blood pressure into clinical staging (systolic-led)."""
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif systolic < 130 and diastolic < 80:
        return "Elevated"
    else:
        return "Stage 1 Hypertension"


def classify_age_group(age: float) -> str:
    """Classify age into clinical age bands."""
    if age <= 17:
        return "Under 18"
    elif age <= 35:
        return "18-35"
    elif age <= 50:
        return "36-50"
    elif age <= 65:
        return "51-65"
    else:
        return "65+"


def classify_bmi_category(bmi: float) -> str:
    """Classify BMI into WHO standard categories."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal"
    elif bmi < 30.0:
        return "Overweight"
    else:
        return "Obese"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain-specific clinical, operational, and seasonal features.
    """
    df_feat = df.copy()

    # Age group
    if "age" in df_feat.columns:
        df_feat["age_group"] = pd.cut(
            df_feat["age"], bins=[0, 17, 35, 50, 65, 120],
            labels=["Under 18", "18-35", "36-50", "51-65", "65+"]
        ).astype(str)

    # BMI category
    if "bmi" in df_feat.columns:
        df_feat["bmi_category"] = pd.cut(
            df_feat["bmi"], bins=[0, 18.5, 25, 30, 100],
            labels=["Underweight", "Normal", "Overweight", "Obese"]
        ).astype(str)

    # Blood pressure category
    if "systolic_bp" in df_feat.columns and "diastolic_bp" in df_feat.columns:
        df_feat["bp_category"] = df_feat.apply(
            lambda r: classify_bp(r["systolic_bp"], r["diastolic_bp"]), axis=1
        )

    # Missed-appointment rate
    if "previous_appointments" in df_feat.columns and "missed_previous_appointments" in df_feat.columns:
        df_feat["missed_appointment_rate"] = np.where(
            df_feat["previous_appointments"] > 0,
            df_feat["missed_previous_appointments"] / df_feat["previous_appointments"],
            0.0
        )

    # Chronic patient flag
    if "previous_admissions" in df_feat.columns:
        df_feat["is_chronic_patient"] = (df_feat["previous_admissions"] >= 2).astype(int)

    # Care intensity
    if "lab_tests_count" in df_feat.columns and "treatments_count" in df_feat.columns:
        df_feat["care_intensity"] = df_feat["lab_tests_count"] + df_feat["treatments_count"]

    # Appointment date seasonality if available
    if "appointment_date" in df_feat.columns:
        try:
            dates = pd.to_datetime(df_feat["appointment_date"])
            df_feat["appointment_month"] = dates.dt.month
            df_feat["appointment_quarter"] = dates.dt.quarter
        except Exception:
            pass

    return df_feat


def fit_feature_pipeline(df_clean: pd.DataFrame, k: int = 15):
    """
    Execute full feature engineering, encoding, selection, and scaling on cleaned data.
    Ensures:
      - Low = 0
      - Medium = 1
      - High = 2
    """
    df_feat = engineer_features(df_clean)

    # Drop leakage columns
    drop_cols = ["record_id", "patient_id", "appointment_date", "no_show", "readmitted_30_days"]
    df_model = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])

    # Target Mapping: Low=0, Medium=1, High=2
    y_raw = df_model["disease_risk_level"]
    y_encoded = pd.Series(y_raw.map(TARGET_MAP).astype(int), name="disease_risk_level")
    X_raw = df_model.drop(columns=["disease_risk_level"])

    # Fit LabelEncoders for all categorical columns
    encoders = {}
    X_encoded = X_raw.copy()
    for col in X_encoded.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col] = le

    # Select K Best Features
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_encoded, y_encoded)
    scores = pd.DataFrame({"feature": X_encoded.columns, "score": selector.scores_}).sort_values("score", ascending=False)
    selected_features = scores["feature"].head(k).tolist()

    X_selected = X_encoded[selected_features]

    # Standard Scaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_selected), columns=selected_features)

    pipeline_artifacts = {
        "encoders": encoders,
        "selected_features": selected_features,
        "scaler": scaler,
        "target_map": TARGET_MAP,
        "inv_target_map": INV_TARGET_MAP,
        "target_classes": TARGET_CLASSES,
        "feature_scores": scores
    }

    return X_scaled, y_encoded, pipeline_artifacts


def transform_single_patient(raw_dict: dict, pipeline_artifacts: dict) -> pd.DataFrame:
    """
    Take a raw dictionary of patient attributes from UI/API, apply feature engineering,
    encode categorical features, align columns with selected_features, and scale.
    """
    encoders = pipeline_artifacts["encoders"]
    selected_features = pipeline_artifacts["selected_features"]
    scaler = pipeline_artifacts["scaler"]

    # Base features from raw dictionary
    p = dict(raw_dict)

    # Compute engineered features
    age = float(p.get("age", 45))
    bmi = float(p.get("bmi", 25.0))
    s_bp = float(p.get("systolic_bp", 120))
    d_bp = float(p.get("diastolic_bp", 80))
    prev_adm = int(p.get("previous_admissions", 0))
    prev_app = int(p.get("previous_appointments", 0))
    miss_app = int(p.get("missed_previous_appointments", 0))
    lab_cnt = int(p.get("lab_tests_count", 0))
    tx_cnt = int(p.get("treatments_count", 0))

    p["age_group"] = classify_age_group(age)
    p["bmi_category"] = classify_bmi_category(bmi)
    p["bp_category"] = classify_bp(s_bp, d_bp)
    p["missed_appointment_rate"] = (miss_app / prev_app) if prev_app > 0 else 0.0
    p["is_chronic_patient"] = int(prev_adm >= 2)
    p["care_intensity"] = lab_cnt + tx_cnt

    # Create 1-row DataFrame
    df = pd.DataFrame([p])

    # Apply encoders
    for col, le in encoders.items():
        if col in df.columns:
            val = str(df[col].iloc[0])
            if val in le.classes_:
                df[col] = le.transform([val])[0]
            else:
                df[col] = 0

    # Ensure all selected features exist
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0.0

    df_selected = df[selected_features].astype(float)

    # Scale
    scaled_array = scaler.transform(df_selected)
    return pd.DataFrame(scaled_array, columns=selected_features)
