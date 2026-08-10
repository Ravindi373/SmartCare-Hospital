"""
CCS3440 - Task 03: Feature Engineering (Option C - Disease Risk Classification)
Creates derived features, encodes categoricals, and produces the final
model-ready feature matrix X and target y.

Usage:
    from feature_engineering import build_features
    X, y, feature_columns, encoding_maps = build_features(df_clean)
"""

import numpy as np
import pandas as pd

BMI_ORDER = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
AGE_ORDER = {"0-18": 0, "19-35": 1, "36-50": 2, "51-65": 3, "65+": 4}
RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2}

NOMINAL_COLS = [
    "gender", "blood_group", "department", "diagnosis",
    "appointment_status", "room_type", "payment_status", "payment_method",
]

NUMERIC_COLS = [
    "age", "waiting_days", "previous_appointments", "missed_previous_appointments",
    "admitted", "length_of_stay_days", "previous_admissions", "systolic_bp", "diastolic_bp",
    "blood_sugar_mg_dl", "cholesterol_mg_dl", "bmi", "lab_tests_count", "treatments_count",
    "consultation_fee_lkr", "room_charge_lkr", "lab_charge_lkr", "medicine_charge_lkr",
    "total_bill_lkr", "prior_utilization", "care_intensity", "missed_appointment_rate",
    "risk_flag_count",
]

# Unlike Option B, 'admitted' is kept as a feature here (it is not constant
# across the full population and may carry genuine severity information).
# no_show and readmitted_30_days are the target variables for Options A and B
# respectively — not legitimate predictors for this Option C task.
DROP_COLS = ["record_id", "patient_id", "appointment_date", "no_show", "readmitted_30_days"]

TARGET = "disease_risk_level"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically meaningful derived features."""
    fe = df.copy()

    fe["bmi_category"] = pd.cut(
        fe["bmi"], bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"])

    fe["age_group"] = pd.cut(
        fe["age"], bins=[0, 18, 35, 50, 65, 120],
        labels=["0-18", "19-35", "36-50", "51-65", "65+"])

    fe["high_bp_flag"] = ((fe["systolic_bp"] >= 140) | (fe["diastolic_bp"] >= 90)).astype(int)
    fe["high_sugar_flag"] = (fe["blood_sugar_mg_dl"] >= 126).astype(int)
    fe["high_chol_flag"] = (fe["cholesterol_mg_dl"] >= 240).astype(int)

    fe["prior_utilization"] = fe["previous_admissions"] + fe["previous_appointments"]
    fe["care_intensity"] = fe["lab_tests_count"] + fe["treatments_count"]
    fe["missed_appointment_rate"] = (
        fe["missed_previous_appointments"] / fe["previous_appointments"].replace(0, np.nan)
    ).fillna(0)

    # Composite clinical risk score — sums the three individual vital-sign flags
    fe["risk_flag_count"] = fe["high_bp_flag"] + fe["high_sugar_flag"] + fe["high_chol_flag"]

    return fe


def encode_features(fe: pd.DataFrame):
    """
    Drop non-predictive/alternate-target columns, ordinal-encode
    bmi_category/age_group, one-hot encode nominal columns, and label-encode
    the target preserving its clinical ordering. Returns (X, y, feature_columns).
    """
    model_df = fe.drop(columns=DROP_COLS)
    y_labels = model_df[TARGET]
    X = model_df.drop(columns=[TARGET])

    # Ordinal encode — cast to int (not left as pandas 'category' dtype,
    # which tree models can treat specially and break inference on new data)
    X["bmi_category"] = X["bmi_category"].map(BMI_ORDER).astype(int)
    X["age_group"] = X["age_group"].map(AGE_ORDER).astype(int)

    X = pd.get_dummies(X, columns=NOMINAL_COLS, drop_first=True)

    y = y_labels.map(RISK_ORDER)

    return X, y, list(X.columns)


def build_features(df_clean: pd.DataFrame):
    """Full Task 03 feature engineering pipeline."""
    fe = add_engineered_features(df_clean)
    X, y, feature_columns = encode_features(fe)
    encoding_maps = {
        "bmi_order": BMI_ORDER,
        "age_order": AGE_ORDER,
        "risk_order": RISK_ORDER,
        "numeric_cols": [c for c in NUMERIC_COLS if c in X.columns],
    }
    return X, y, feature_columns, encoding_maps


if __name__ == "__main__":
    from preprocessing import load_and_clean_data

    df_clean = load_and_clean_data("../data/raw/smartcare_ai_dataset_1000.csv")
    X, y, feature_columns, encoding_maps = build_features(df_clean)
    print(f"Feature matrix: {X.shape}")
    print(f"Target balance:\n{y.value_counts(normalize=True).round(3)}")
