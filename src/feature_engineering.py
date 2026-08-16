"""
Feature Engineering Module for SmartCare Hospital AI Project
Provides clinical feature construction, one-hot categorical encoding,
feature selection, standard scaling, and single-patient transformation.
Enforces strict train/test leakage isolation and explicit target mapping.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

BASE_DIR = Path(__file__).resolve().parent.parent

# Explicit target mapping: 0 = Low, 1 = Medium, 2 = High
TARGET_MAP = {"Low": 0, "Medium": 1, "High": 2}
INV_TARGET_MAP = {0: "Low", 1: "Medium", 2: "High"}
TARGET_CLASSES = ["Low", "Medium", "High"]

DROP_COLS = ["record_id", "patient_id", "appointment_date", "no_show", "readmitted_30_days"]


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
    Row-wise operations — completely free of cross-sample data leakage.
    """
    df_feat = df.copy()

    # Age group
    if "age" in df_feat.columns:
        df_feat["age_group"] = pd.cut(
            df_feat["age"], bins=[-1, 17, 35, 50, 65, 120],
            labels=["Under 18", "18-35", "36-50", "51-65", "65+"]
        ).astype(str)

    # BMI category
    if "bmi" in df_feat.columns:
        df_feat["bmi_category"] = pd.cut(
            df_feat["bmi"], bins=[-1, 18.5, 25, 30, 100],
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


def fit_feature_pipeline(df_train: pd.DataFrame, df_test: pd.DataFrame = None, k: int = 15):
    """
    Execute feature engineering, OneHotEncoding, feature selection, and scaling strictly on df_train.
    Transform df_test (if provided) using fitted artifacts only.
    Returns:
      (X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, pipeline_artifacts)
      or (X_train_scaled, y_train_encoded, pipeline_artifacts) if df_test is None.
    """
    # 1. Engineer features
    train_feat = engineer_features(df_train)
    if df_test is not None:
        test_feat = engineer_features(df_test)

    # 2. Drop leakage columns & target separation
    train_clean = train_feat.drop(columns=[c for c in DROP_COLS if c in train_feat.columns])
    y_train = train_clean["disease_risk_level"].map(TARGET_MAP).astype(int)
    X_train_raw = train_clean.drop(columns=["disease_risk_level"])

    if df_test is not None:
        test_clean = test_feat.drop(columns=[c for c in DROP_COLS if c in test_feat.columns])
        y_test = test_clean["disease_risk_level"].map(TARGET_MAP).astype(int)
        X_test_raw = test_clean.drop(columns=["disease_risk_level"])

    # 3. Categorical vs Numerical split
    cat_cols = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()

    # 4. One-Hot Encode nominal variables (FIT strictly on train)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(X_train_raw[cat_cols])
    ohe_feature_names = ohe.get_feature_names_out(cat_cols).tolist()

    X_train_cat = pd.DataFrame(
        ohe.transform(X_train_raw[cat_cols]),
        columns=ohe_feature_names,
        index=X_train_raw.index
    )
    X_train_encoded = pd.concat([X_train_raw[num_cols], X_train_cat], axis=1)

    # 5. Feature Selection via ANOVA F-score (FIT strictly on train)
    k_actual = min(k, X_train_encoded.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k_actual)
    selector.fit(X_train_encoded, y_train)

    scores = pd.DataFrame({
        "feature": X_train_encoded.columns,
        "score": selector.scores_
    }).sort_values("score", ascending=False).reset_index(drop=True)

    selected_features = X_train_encoded.columns[selector.get_support()].tolist()
    X_train_selected = X_train_encoded[selected_features]

    # 6. Standard Scaling (FIT strictly on train)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_selected),
        columns=selected_features,
        index=X_train_raw.index
    )

    pipeline_artifacts = {
        "ohe": ohe,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "ohe_feature_names": ohe_feature_names,
        "selector": selector,
        "selected_features": selected_features,
        "scaler": scaler,
        "target_map": TARGET_MAP,
        "inv_target_map": INV_TARGET_MAP,
        "target_classes": TARGET_CLASSES,
        "feature_scores": scores
    }

    if df_test is not None:
        # Transform test set strictly using fitted artifacts
        X_test_cat = pd.DataFrame(
            ohe.transform(X_test_raw[cat_cols]),
            columns=ohe_feature_names,
            index=X_test_raw.index
        )
        X_test_encoded = pd.concat([X_test_raw[num_cols], X_test_cat], axis=1)
        X_test_selected = X_test_encoded[selected_features]
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_selected),
            columns=selected_features,
            index=X_test_raw.index
        )
        return X_train_scaled, y_train, X_test_scaled, y_test, pipeline_artifacts

    return X_train_scaled, y_train, pipeline_artifacts


def transform_single_patient(raw_dict: dict, pipeline_artifacts: dict) -> pd.DataFrame:
    """
    Take a raw dictionary of patient attributes from UI/API, apply feature engineering,
    one-hot encode categorical features, select K best features, and scale using saved pipeline artifacts.
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

    # Ensure all numerical cols exist
    for col in num_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Ensure all categorical cols exist
    for col in cat_cols:
        if col not in df.columns:
            df[col] = "Unknown"

    df_num = df[num_cols].astype(float)
    df_cat = pd.DataFrame(
        ohe.transform(df[cat_cols].astype(str)),
        columns=ohe_feature_names,
        index=df.index
    )
    df_encoded = pd.concat([df_num, df_cat], axis=1)

    # Ensure all selected features exist
    for col in selected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0.0

    df_selected = df_encoded[selected_features].astype(float)
    scaled_array = scaler.transform(df_selected)
    return pd.DataFrame(scaled_array, columns=selected_features)
