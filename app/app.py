"""
SmartCare Hospital — Disease Risk Level Classification System (Option C)
Streamlit Prototype Application for Clinical Decision Support
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SmartCare Disease Risk Classifier",
    page_icon="🩺",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "pipeline_bundle.joblib"
if not BUNDLE_PATH.exists():
    BUNDLE_PATH = BASE_DIR.parent / "models" / "pipeline_bundle.joblib"

LABEL_NAMES = ["Low", "Medium", "High"]


def classify_bp(s: float, d: float) -> str:
    if s < 120 and d < 80:
        return "Normal"
    elif s < 130 and d < 80:
        return "Elevated"
    else:
        return "Stage 1 Hypertension"


def classify_age_group(age: float) -> str:
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
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal"
    elif bmi < 30.0:
        return "Overweight"
    else:
        return "Obese"


@st.cache_resource
def load_artifacts():
    if BUNDLE_PATH.exists():
        bundle = joblib.load(BUNDLE_PATH)
        return bundle
    else:
        # Fallback to direct model and scaler if bundle not yet created
        model = joblib.load(BASE_DIR / "disease_risk_model.pkl")
        scaler = joblib.load(BASE_DIR / "feature_scaler.pkl")
        return {"best_model": model, "scaler": scaler, "selected_features": getattr(scaler, "feature_names_in_", None)}


bundle = load_artifacts()

st.title("🩺 SmartCare Hospital")
st.subheader("AI-Powered Patient Disease Risk Classifier")
st.caption("CCS3440 Artificial Intelligence Coursework — Option C: Multi-Class Risk Stratification")
st.divider()

st.markdown("### 📋 Patient Intake & Clinical Form")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=45, help="Patient age in years")
        gender = st.selectbox("Gender", ["Male", "Female"])
        blood_group = st.selectbox("Blood Group", ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"])
        department = st.selectbox(
            "Department",
            ["Cardiology", "General Medicine", "Laboratory Services", "Neurology", "Orthopedics", "Pediatrics", "Radiology"]
        )
        diagnosis = st.selectbox(
            "Primary Diagnosis",
            ["Asthma", "Back Pain", "Chest Pain", "Diabetes", "Fever", "Fracture", "Hypertension", "Kidney Infection", "Migraine", "Pneumonia"]
        )
        appointment_status = st.selectbox("Appointment Status", ["Completed", "Scheduled", "No-Show", "Cancelled"])
        admitted = st.selectbox("Admitted?", ["No", "Yes"])
        room_type = st.selectbox("Room Type", ["Not Admitted", "General Ward", "Private Room", "ICU"])

    with col2:
        payment_status = st.selectbox("Payment Status", ["Paid", "Partially Paid", "Unpaid"])
        payment_method = st.selectbox("Payment Method", ["Card", "Cash", "Insurance", "Online"])
        waiting_days = st.number_input("Waiting Days", 0, 60, 4)
        previous_appointments = st.number_input("Previous Appointments", 0, 30, 3)
        missed_previous_appointments = st.number_input("Missed Appointments", 0, 30, 0)
        length_of_stay_days = st.number_input("Length of Stay (days)", 0, 60, 0)
        previous_admissions = st.number_input("Previous Admissions", 0, 20, 1)

    st.markdown("#### 🫀 Physiological Vitals")
    col3, col4 = st.columns(2)
    with col3:
        systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", 70, 220, 125)
        diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", 40, 140, 82)
        blood_sugar = st.number_input("Blood Sugar (mg/dL)", 50, 400, 110)
    with col4:
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 190)
        bmi = st.number_input("Body Mass Index (BMI)", 10.0, 60.0, 26.0, step=0.1)

    st.markdown("#### 💳 Operations & Billing")
    col5, col6 = st.columns(2)
    with col5:
        lab_tests_count = st.number_input("Lab Tests Count", 0, 20, 2)
        treatments_count = st.number_input("Treatments Count", 0, 20, 2)
    with col6:
        consultation_fee = st.number_input("Consultation Fee (LKR)", 0, 20000, 2000)
        room_charge = st.number_input("Room Charge (LKR)", 0, 200000, 0)
        lab_charge = st.number_input("Lab Charge (LKR)", 0, 100000, 3000)
        medicine_charge = st.number_input("Medicine Charge (LKR)", 0, 100000, 4000)

    submitted = st.form_submit_button("🔍 Classify Disease Risk Level", use_container_width=True)

# ---------------------------------------------------------------
# Inference Execution
# ---------------------------------------------------------------
if submitted:
    # 1. Construct raw dictionary
    patient_dict = {
        "age": float(age),
        "gender": gender,
        "blood_group": blood_group,
        "department": department,
        "diagnosis": diagnosis,
        "appointment_status": appointment_status,
        "admitted": 1 if admitted == "Yes" else 0,
        "room_type": room_type,
        "payment_status": payment_status,
        "payment_method": payment_method,
        "waiting_days": float(waiting_days),
        "previous_appointments": float(previous_appointments),
        "missed_previous_appointments": float(missed_previous_appointments),
        "length_of_stay_days": float(length_of_stay_days),
        "previous_admissions": float(previous_admissions),
        "systolic_bp": float(systolic_bp),
        "diastolic_bp": float(diastolic_bp),
        "blood_sugar_mg_dl": float(blood_sugar),
        "cholesterol_mg_dl": float(cholesterol),
        "bmi": float(bmi),
        "lab_tests_count": float(lab_tests_count),
        "treatments_count": float(treatments_count),
        "consultation_fee_lkr": float(consultation_fee),
        "room_charge_lkr": float(room_charge),
        "lab_charge_lkr": float(lab_charge),
        "medicine_charge_lkr": float(medicine_charge),
    }

    # 2. Compute engineered features
    bp_cat = classify_bp(systolic_bp, diastolic_bp)
    age_grp = classify_age_group(age)
    bmi_cat = classify_bmi_category(bmi)
    is_chronic = int(previous_admissions >= 2)
    miss_rate = (missed_previous_appointments / previous_appointments) if previous_appointments > 0 else 0.0

    patient_dict["bp_category"] = bp_cat
    patient_dict["age_group"] = age_grp
    patient_dict["bmi_category"] = bmi_cat
    patient_dict["is_chronic_patient"] = is_chronic
    patient_dict["missed_appointment_rate"] = miss_rate
    patient_dict["care_intensity"] = lab_tests_count + treatments_count

    # 3. Transform inputs
    model = bundle.get("best_model", bundle.get("model"))
    scaler = bundle.get("scaler")
    ohe = bundle.get("ohe")
    selected_features = bundle.get("selected_features", [])

    if ohe is not None and scaler is not None and len(selected_features) > 0:
        # OneHotEncoder Pipeline Transformation
        cat_cols = bundle.get("cat_cols", [])
        num_cols = bundle.get("num_cols", [])
        ohe_feature_names = bundle.get("ohe_feature_names", [])

        df_input = pd.DataFrame([patient_dict])
        for c in num_cols:
            if c not in df_input.columns:
                df_input[c] = 0.0
        for c in cat_cols:
            if c not in df_input.columns:
                df_input[c] = "Unknown"

        df_num = df_input[num_cols].astype(float)
        df_cat = pd.DataFrame(
            ohe.transform(df_input[cat_cols].astype(str)),
            columns=ohe_feature_names,
            index=df_input.index
        )
        df_encoded = pd.concat([df_num, df_cat], axis=1)

        for col in selected_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0.0

        df_selected = df_encoded[selected_features].astype(float)
        scaled_input = pd.DataFrame(scaler.transform(df_selected), columns=selected_features)
    elif "encoders" in bundle:
        # Legacy LabelEncoder fallback
        encoders = bundle.get("encoders", {})
        df_input = pd.DataFrame([patient_dict])
        for col, le in encoders.items():
            if col in df_input.columns:
                val = str(df_input[col].iloc[0])
                if val in le.classes_:
                    df_input[col] = le.transform([val])[0]
                else:
                    df_input[col] = 0
        for col in selected_features:
            if col not in df_input.columns:
                df_input[col] = 0.0
        df_selected = df_input[selected_features].astype(float)
        scaled_input = pd.DataFrame(scaler.transform(df_selected), columns=selected_features)
    else:
        # Fallback for 5-feature scaler
        fallback_features = ["blood_sugar_mg_dl", "cholesterol_mg_dl", "age", "bmi", "previous_admissions"]
        df_selected = pd.DataFrame([{f: patient_dict[f] for f in fallback_features}])
        scaled_input = pd.DataFrame(scaler.transform(df_selected), columns=fallback_features)

    # 4. Predict
    pred_idx = int(model.predict(scaled_input)[0])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(scaled_input)[0]
    else:
        probs = np.array([1.0 if i == pred_idx else 0.0 for i in range(3)])

    pred_label = LABEL_NAMES[pred_idx]
    confidence = probs[pred_idx]

    # 5. Render Output
    st.divider()
    st.markdown("### 📊 Prediction Results & Clinical Stratification")

    if pred_label == "Low":
        st.success(f"🟢 **Predicted Risk Level: LOW RISK** (Confidence: {confidence:.1%})")
        st.info("💡 **Clinical Recommendation:** Routine monitoring and preventive lifestyle counseling.")
    elif pred_label == "Medium":
        st.warning(f"🟠 **Predicted Risk Level: MEDIUM RISK** (Confidence: {confidence:.1%})")
        st.info("⚠️ **Clinical Recommendation:** Schedule closer follow-up review and targeted diagnostic panels.")
    else:
        st.error(f"🔴 **Predicted Risk Level: HIGH RISK** (Confidence: {confidence:.1%})")
        st.info("🚨 **Clinical Recommendation:** Immediate clinician evaluation, inpatient care or aggressive intervention.")

    # Clinical Alert Indicators
    st.markdown("#### 🩺 Clinical Vitals Summary")
    alert_cols = st.columns(4)
    with alert_cols[0]:
        st.metric("BP Staging", bp_cat, delta="Normal" if bp_cat == "Normal" else "Elevated/High", delta_color="inverse")
    with alert_cols[1]:
        st.metric("Blood Sugar", f"{blood_sugar:.0f} mg/dL", delta="Normal" if blood_sugar < 126 else "Diabetic", delta_color="inverse")
    with alert_cols[2]:
        st.metric("Cholesterol", f"{cholesterol:.0f} mg/dL", delta="Normal" if cholesterol < 200 else "High", delta_color="inverse")
    with alert_cols[3]:
        st.metric("Chronic Flag", "Yes (>=2 adm)" if is_chronic else "No", delta=None)

    # Probability Chart
    st.markdown("#### 📈 Class Probability Breakdown")
    prob_df = pd.DataFrame({"Risk Level": LABEL_NAMES, "Probability": probs}).set_index("Risk Level")
    st.bar_chart(prob_df)

    with st.expander("🔍 View Scaled Model Feature Vector"):
        st.dataframe(scaled_input.T.rename(columns={0: "Standardized Value"}).round(4))

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This system is an academic machine learning prototype for CCS3440 coursework. "
    "It is designed for clinical decision support evaluation and should not replace qualified medical diagnosis."
)
