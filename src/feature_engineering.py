"""
CCS3440 Artificial Intelligence Coursework | Group 02
Option C: Disease Risk Classification - SmartCare Hospital
Module: Feature Engineering & Preprocessing Pipeline
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

# Explicit target mapping: 0 = Low, 1 = Medium, 2 = High
# Ensures deterministic class assignment avoiding alphabetical inversion
TARGET_MAP = {"Low": 0, "Medium": 1, "High": 2}
INV_TARGET_MAP = {0: "Low", 1: "Medium", 2: "High"}
TARGET_CLASSES = ["Low", "Medium", "High"]

# Nominal categorical columns (One-Hot Encoded to eliminate artificial ordinal bias)
NOMINAL_CATEGORICAL_COLS = [
    "gender", "blood_group", "department", "diagnosis",
    "appointment_status", "room_type", "payment_status", "payment_method",
    "age_group", "bmi_category", "bp_category"
]

NUMERIC_COLS = [
    "age", "waiting_days", "previous_appointments", "missed_previous_appointments",
    "admitted", "length_of_stay_days", "previous_admissions", "systolic_bp",
    "diastolic_bp", "blood_sugar_mg_dl", "cholesterol_mg_dl", "bmi",
    "lab_tests_count", "treatments_count", "consultation_fee_lkr",
    "room_charge_lkr", "lab_charge_lkr", "medicine_charge_lkr", "total_bill_lkr",
    "missed_appointment_rate", "is_chronic_patient", "care_intensity"
]


def classify_bp(systolic: float, diastolic: float) -> str:
    """Classify blood pressure into clinical staging."""
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif systolic < 130 and diastolic < 80:
        return "Elevated"
    else:
        return "Hypertension"


def classify_age_group(age: float) -> str:
    """Classify age into clinical age brackets."""
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
    """Classify BMI according to WHO standard guidelines."""
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
    Construct domain-specific clinical, operational, and behavioral features.
    """
    df_feat = df.copy()

    # 1. Clinical age group
    if "age" in df_feat.columns:
        df_feat["age_group"] = pd.cut(
            df_feat["age"], bins=[0, 17, 35, 50, 65, 120],
            labels=["Under 18", "18-35", "36-50", "51-65", "65+"]
        ).astype(str)

    # 2. BMI WHO category
    if "bmi" in df_feat.columns:
        df_feat["bmi_category"] = pd.cut(
            df_feat["bmi"], bins=[0, 18.5, 25, 30, 100],
            labels=["Underweight", "Normal", "Overweight", "Obese"]
        ).astype(str)

    # 3. Blood Pressure staging
    if "systolic_bp" in df_feat.columns and "diastolic_bp" in df_feat.columns:
        df_feat["bp_category"] = df_feat.apply(
            lambda r: classify_bp(r["systolic_bp"], r["diastolic_bp"]), axis=1
        )

    # 4. Behavioral reliability: Missed-appointment rate
    if "previous_appointments" in df_feat.columns and "missed_previous_appointments" in df_feat.columns:
        df_feat["missed_appointment_rate"] = np.where(
            df_feat["previous_appointments"] > 0,
            df_feat["missed_previous_appointments"] / df_feat["previous_appointments"],
            0.0
        )

    # 5. Clinical chronicity flag
    if "previous_admissions" in df_feat.columns:
        df_feat["is_chronic_patient"] = (df_feat["previous_admissions"] >= 2).astype(int)

    # 6. Operational care intensity
    if "lab_tests_count" in df_feat.columns and "treatments_count" in df_feat.columns:
        df_feat["care_intensity"] = df_feat["lab_tests_count"] + df_feat["treatments_count"]

    # 7. Appointment month / quarter if date is present
    if "appointment_date" in df_feat.columns:
        try:
            dates = pd.to_datetime(df_feat["appointment_date"])
            df_feat["appointment_month"] = dates.dt.month
        except Exception:
            pass

    return df_feat


def fit_and_transform_pipeline(X_train_raw: pd.DataFrame, y_train: pd.Series,
                               X_test_raw: pd.DataFrame = None, k: int = 15):
    """
    Fit feature engineering, One-Hot Encoding, Feature Selection (ANOVA F-score),
    and Standard Scaling strictly on the Training set (X_train_raw, y_train) to prevent
    data leakage. Then transform X_test_raw using the fitted components.
    """
    # 1. Engineer features on train
    X_train_feat = engineer_features(X_train_raw)

    # Drop leakage / identifier columns
    drop_cols = ["record_id", "patient_id", "appointment_date", "no_show", "readmitted_30_days", "disease_risk_level"]
    X_train_clean = X_train_feat.drop(columns=[c for c in drop_cols if c in X_train_feat.columns])

    # Identify categorical and numeric columns present
    cat_cols = [c for c in NOMINAL_CATEGORICAL_COLS if c in X_train_clean.columns]
    num_cols = [c for c in NUMERIC_COLS if c in X_train_clean.columns]

    # OneHotEncoder for nominal variables
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(X_train_clean[cat_cols])
    ohe_feature_names = ohe.get_feature_names_out(cat_cols).tolist()

    train_ohe_df = pd.DataFrame(
        ohe.transform(X_train_clean[cat_cols]),
        columns=ohe_feature_names,
        index=X_train_clean.index
    )
    train_num_df = X_train_clean[num_cols].copy()

    # Combine all numeric + OHE features
    X_train_combined = pd.concat([train_num_df, train_ohe_df], axis=1)

    # 2. Feature Selection (Top K Best Features by ANOVA F-value on Train)
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train_combined, y_train)
    scores = pd.DataFrame({
        "feature": X_train_combined.columns,
        "score": selector.scores_
    }).sort_values("score", ascending=False)
    selected_features = scores["feature"].head(k).tolist()

    X_train_selected = X_train_combined[selected_features]

    # 3. Standard Scaler fitted ONLY on training features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_selected),
        columns=selected_features,
        index=X_train_selected.index
    )

    pipeline_artifacts = {
        "ohe": ohe,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "ohe_feature_names": ohe_feature_names,
        "all_feature_names": X_train_combined.columns.tolist(),
        "selector": selector,
        "selected_features": selected_features,
        "scaler": scaler,
        "target_map": TARGET_MAP,
        "inv_target_map": INV_TARGET_MAP,
        "target_classes": TARGET_CLASSES,
        "feature_scores": scores
    }

    # Transform test set if provided
    X_test_scaled = None
    if X_test_raw is not None:
        X_test_feat = engineer_features(X_test_raw)
        X_test_clean = X_test_feat.drop(columns=[c for c in drop_cols if c in X_test_feat.columns])

        test_ohe_df = pd.DataFrame(
            ohe.transform(X_test_clean[cat_cols]),
            columns=ohe_feature_names,
            index=X_test_clean.index
        )
        test_num_df = X_test_clean[num_cols].copy()
        X_test_combined = pd.concat([test_num_df, test_ohe_df], axis=1)
        X_test_selected = X_test_combined[selected_features]
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_selected),
            columns=selected_features,
            index=X_test_selected.index
        )

    return X_train_scaled, X_test_scaled, pipeline_artifacts


def transform_single_patient(raw_dict: dict, pipeline_artifacts: dict) -> pd.DataFrame:
    """
    Take a raw dictionary of patient attributes from UI/API, apply domain feature engineering,
    One-Hot Encode nominal variables, select top K features, and scale using the training artifacts.
    """
    ohe = pipeline_artifacts["ohe"]
    cat_cols = pipeline_artifacts["cat_cols"]
    num_cols = pipeline_artifacts["num_cols"]
    ohe_feature_names = pipeline_artifacts["ohe_feature_names"]
    selected_features = pipeline_artifacts["selected_features"]
    scaler = pipeline_artifacts["scaler"]

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

    df = pd.DataFrame([p])

    # OHE transform
    df_cat = df[[c for c in cat_cols if c in df.columns]]
    ohe_df = pd.DataFrame(ohe.transform(df_cat), columns=ohe_feature_names)

    # Numeric df
    num_data = {}
    for col in num_cols:
        num_data[col] = float(df[col].iloc[0]) if col in df.columns else 0.0
    num_df = pd.DataFrame([num_data])

    combined = pd.concat([num_df, ohe_df], axis=1)

    for col in selected_features:
        if col not in combined.columns:
            combined[col] = 0.0

    selected_df = combined[selected_features].astype(float)
    scaled_array = scaler.transform(selected_df)
    return pd.DataFrame(scaled_array, columns=selected_features)
