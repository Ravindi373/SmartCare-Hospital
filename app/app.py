import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Disease Risk Classifier", page_icon="🩺", layout="centered")

# ---------------------------------------------------------------
# Load trained model + preprocessing artifacts (from Task 05-07)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model_lr_optionC.pkl")
    scaler = joblib.load("scaler_optionC.pkl")
    feature_columns = joblib.load("feature_columns_optionC.pkl")
    encoding_maps = joblib.load("encoding_maps_optionC.pkl")
    return model, scaler, feature_columns, encoding_maps

model, scaler, feature_columns, encoding_maps = load_artifacts()
bmi_order = encoding_maps["bmi_order"]
age_order = encoding_maps["age_order"]
risk_order = encoding_maps["risk_order"]
numeric_cols = encoding_maps["numeric_cols"]
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
    total_bill = consultation_fee + room_charge + lab_charge + medicine_charge
    admitted_flag = 1 if admitted == "Yes" else 0

    if bmi < 18.5: bmi_cat = "Underweight"
    elif bmi < 25: bmi_cat = "Normal"
    elif bmi < 30: bmi_cat = "Overweight"
    else: bmi_cat = "Obese"

    if age <= 18: age_grp = "0-18"
    elif age <= 35: age_grp = "19-35"
    elif age <= 50: age_grp = "36-50"
    elif age <= 65: age_grp = "51-65"
    else: age_grp = "65+"

    high_bp_flag = int(systolic_bp >= 140 or diastolic_bp >= 90)
    high_sugar_flag = int(blood_sugar >= 126)
    high_chol_flag = int(cholesterol >= 240)
    risk_flag_count = high_bp_flag + high_sugar_flag + high_chol_flag
    prior_utilization = previous_admissions + previous_appointments
    care_intensity = lab_tests_count + treatments_count
    missed_rate = (missed_previous_appointments / previous_appointments) if previous_appointments > 0 else 0.0

    raw = {
        "age": age, "waiting_days": waiting_days, "previous_appointments": previous_appointments,
        "missed_previous_appointments": missed_previous_appointments, "admitted": admitted_flag,
        "length_of_stay_days": length_of_stay_days, "previous_admissions": previous_admissions,
        "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
        "blood_sugar_mg_dl": blood_sugar, "cholesterol_mg_dl": cholesterol, "bmi": bmi,
        "lab_tests_count": lab_tests_count, "treatments_count": treatments_count,
        "consultation_fee_lkr": consultation_fee, "room_charge_lkr": room_charge,
        "lab_charge_lkr": lab_charge, "medicine_charge_lkr": medicine_charge,
        "total_bill_lkr": total_bill,
        "bmi_category": bmi_order[bmi_cat], "age_group": age_order[age_grp],
        "high_bp_flag": high_bp_flag, "high_sugar_flag": high_sugar_flag,
        "high_chol_flag": high_chol_flag, "risk_flag_count": risk_flag_count,
        "prior_utilization": prior_utilization, "care_intensity": care_intensity,
        "missed_appointment_rate": missed_rate,
        "gender": gender, "blood_group": blood_group, "department": department,
        "diagnosis": diagnosis, "appointment_status": appointment_status,
        "room_type": room_type, "payment_status": payment_status, "payment_method": payment_method,
    }

    input_df = pd.DataFrame([raw])
    nominal_cols = ['gender','blood_group','department','diagnosis','appointment_status',
                     'room_type','payment_status','payment_method']
    input_encoded = pd.get_dummies(input_df, columns=nominal_cols)
    input_final = input_encoded.reindex(columns=feature_columns, fill_value=0).astype(float)

    # Logistic Regression was trained on SCALED numeric features — must scale here too
    input_final[numeric_cols] = scaler.transform(input_final[numeric_cols])

    pred_idx = model.predict(input_final)[0]
    probs = model.predict_proba(input_final)[0]
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
    **Top clinical drivers of risk classification** (from Task 07 SHAP analysis):
    blood sugar, cholesterol, age, BMI, and systolic BP tend to weigh most heavily
    in this model.
    """)

    with st.expander("Show raw model input vector"):
        st.dataframe(input_final.T.rename(columns={0: "value"}))

st.divider()
st.caption("Prototype for academic purposes — SmartCare Hospital AI Dataset (synthetic). "
           "Not a real clinical decision-support tool.")
