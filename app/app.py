"""
CCS3440 - Task 08: AI Prototype Development
SmartCare Hospital - Disease Risk Level Classifier (Option C)
Run with: streamlit run app_optionC.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Disease Risk Classifier", page_icon="🩺", layout="centered")

# ---------------------------------------------------------------
# Load trained model + preprocessing artifacts (from Task 05-07)
# ---------------------------------------------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
<<<<<<< HEAD
    model = joblib.load(BASE_DIR / "disease_risk_model.pkl")
    scaler = joblib.load(BASE_DIR / "feature_scaler.pkl")
=======
    model = joblib.load("disease_risk_model.pkl")
    scaler = joblib.load("feature_scaler.pkl")
>>>>>>> 937e2c246a2f3b642a8054e578a868a23829fc39
    return model, scaler

model, scaler = load_artifacts()
label_names = ["Low", "Medium", "High"]

st.title("🩺 SmartCare Hospital")
st.subheader("Disease Risk Level Classifier")
st.caption("CCS3440 Artificial Intelligence Coursework — Option C Prototype (Logistic Regression + SHAP)")
st.divider()

st.markdown("### Patient & Clinical Details")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 120, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        blood_group = st.selectbox("Blood Group", ["A+","A-","AB+","AB-","B+","B-","O+","O-"])
        department = st.selectbox("Department", ["Cardiology","General Medicine","Laboratory Services",
                                                    "Neurology","Orthopedics","Pediatrics","Radiology"])
        diagnosis = st.selectbox("Diagnosis", ["Asthma","Back Pain","Chest Pain","Diabetes","Fever",
                                                 "Fracture","Hypertension","Kidney Infection","Migraine","Pneumonia"])
        appointment_status = st.selectbox("Appointment Status", ["Completed","No-Show","Scheduled","Cancelled"])
        admitted = st.selectbox("Admitted?", ["No", "Yes"])
        room_type = st.selectbox("Room Type", ["Not Admitted","General Ward","Private Room","ICU"])

    with col2:
        payment_status = st.selectbox("Payment Status", ["Paid","Partially Paid","Unpaid"])
        payment_method = st.selectbox("Payment Method", ["Card","Cash","Insurance","Online"])
        waiting_days = st.number_input("Waiting Days", 0, 60, 5)
        previous_appointments = st.number_input("Previous Appointments", 0, 30, 3)
        missed_previous_appointments = st.number_input("Missed Previous Appointments", 0, 30, 0)
        length_of_stay_days = st.number_input("Length of Stay (days)", 0, 60, 0)
        previous_admissions = st.number_input("Previous Admissions", 0, 20, 0)

    st.markdown("#### Vitals")
    col3, col4 = st.columns(2)
    with col3:
        systolic_bp = st.number_input("Systolic BP", 70, 220, 125)
        diastolic_bp = st.number_input("Diastolic BP", 40, 140, 80)
        blood_sugar = st.number_input("Blood Sugar (mg/dL)", 50, 400, 110)
    with col4:
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 190)
        bmi = st.number_input("BMI", 10.0, 60.0, 26.0, step=0.1)

    st.markdown("#### Treatment & Billing")
    col5, col6 = st.columns(2)
    with col5:
        lab_tests_count = st.number_input("Lab Tests Count", 0, 20, 2)
        treatments_count = st.number_input("Treatments Count", 0, 20, 2)
    with col6:
        consultation_fee = st.number_input("Consultation Fee (LKR)", 0, 20000, 2000)
        room_charge = st.number_input("Room Charge (LKR)", 0, 200000, 0)
        lab_charge = st.number_input("Lab Charge (LKR)", 0, 100000, 3000)
        medicine_charge = st.number_input("Medicine Charge (LKR)", 0, 100000, 4000)

    submitted = st.form_submit_button("🔍 Classify Disease Risk", use_container_width=True)

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
if submitted:
    # Prepare inputs for the 5 features required by the model
    raw = {
        "blood_sugar_mg_dl": float(blood_sugar),
        "cholesterol_mg_dl": float(cholesterol),
        "age": float(age),
        "bmi": float(bmi),
        "previous_admissions": float(previous_admissions)
    }

    input_df = pd.DataFrame([raw])
    
    # Scale input features
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

    pred_idx = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]
    pred_label = label_names[pred_idx]

    st.divider()
    st.markdown("### Prediction Result")

    color_map = {"Low": "success", "Medium": "warning", "High": "error"}
    icon_map = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}
    getattr(st, color_map[pred_label])(
        f"{icon_map[pred_label]} **Predicted Disease Risk Level: {pred_label}** "
        f"(confidence: {probs[pred_idx]:.1%})"
    )

    st.markdown("#### Class Probabilities")
    prob_df = pd.DataFrame({"Risk Level": label_names, "Probability": probs}).set_index("Risk Level")
    st.bar_chart(prob_df)

    st.markdown("""
    **Top clinical drivers of risk classification**:
    Blood sugar, cholesterol, age, BMI, and previous admissions are used by this model
    to classify the patient's risk category.
    """)

    with st.expander("Show scaled model input vector"):
        st.dataframe(input_scaled.T.rename(columns={0: "value"}))

st.divider()
st.caption("Prototype for academic purposes — SmartCare Hospital AI Dataset (synthetic). "
           "Not a real clinical decision-support tool.")
