"""
CCS3440 Artificial Intelligence Coursework | Group 02
SmartCare Hospital — Disease Risk Level Classification System (Option C)
Deployment Demonstration & Clinical Decision Support Interface
"""

from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Add src to sys.path if needed for shared utilities
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from feature_engineering import transform_single_patient, classify_bp, classify_age_group, classify_bmi_category
except ImportError:
    pass

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


@st.cache_resource
def load_artifacts():
    if BUNDLE_PATH.exists():
        bundle = joblib.load(BUNDLE_PATH)
        return bundle
    else:
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

if submitted:
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

    # Transform patient inputs using the full pipeline
    model = bundle.get("best_model", bundle.get("model"))
    if "ohe" in bundle:
        X_input = transform_single_patient(patient_dict, bundle)
    else:
        fallback_features = ["blood_sugar_mg_dl", "cholesterol_mg_dl", "age", "bmi", "systolic_bp"]
        scaler = bundle.get("scaler")
        df_selected = pd.DataFrame([{f: patient_dict[f] for f in fallback_features}])
        X_input = pd.DataFrame(scaler.transform(df_selected), columns=fallback_features)

    # Predict
    pred_idx = int(model.predict(X_input)[0])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
    else:
        probs = np.array([1.0 if i == pred_idx else 0.0 for i in range(3)])

    pred_label = LABEL_NAMES[pred_idx]
    confidence = probs[pred_idx]

    # Clinical categories for summary display
    bp_cat = "Normal" if (systolic_bp < 120 and diastolic_bp < 80) else ("Elevated" if systolic_bp < 130 and diastolic_bp < 80 else "Hypertension")
    is_chronic = int(previous_admissions >= 2)

    # Render Output
    st.divider()
    st.markdown("### 📊 Prediction Results & Clinical Stratification")

    if pred_label == "Low":
        st.success(f"🟢 **Predicted Risk Level: LOW RISK** (Confidence: {confidence:.1%})")
        st.info("💡 **Clinical Recommendation:** Routine annual monitoring and lifestyle wellness counseling.")
    elif pred_label == "Medium":
        st.warning(f"🟠 **Predicted Risk Level: MEDIUM RISK** (Confidence: {confidence:.1%})")
        st.info("⚠️ **Clinical Recommendation:** Schedule 3-month follow-up review and targeted diagnostic metabolic panels.")
    else:
        st.error(f"🔴 **Predicted Risk Level: HIGH RISK** (Confidence: {confidence:.1%})")
        st.info("🚨 **Clinical Recommendation:** Immediate clinician consultation, continuous telemetry or inpatient management.")

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

    with st.expander("🔍 View Standardized Feature Vector"):
        st.dataframe(X_input.T.rename(columns={0: "Standardized Value"}).round(4))

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This system is an academic machine learning deployment demonstration for CCS3440 coursework. "
    "It is designed for clinical decision support evaluation and should not replace qualified medical diagnosis."
)
