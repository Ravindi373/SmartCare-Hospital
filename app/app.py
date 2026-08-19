"""
SmartCare Hospital — Disease Risk Level Classification System
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

# Page Configuration
st.set_page_config(
    page_title="SmartCare AI | Clinical Risk Stratification",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Modern Styling
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Top Hero Header */
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        color: white;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        margin-top: 6px;
        margin-bottom: 12px;
        font-weight: 400;
    }

    .status-badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 12px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #e2e8f0;
    }

    .status-pill-green {
        background: rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.4);
        color: #34d399;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #34d399;
        box-shadow: 0 0 8px #34d399;
    }

    /* Section Cards */
    .section-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 18px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .section-card:hover {
        border-color: rgba(45, 212, 191, 0.3);
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Prediction Result Cards */
    .result-card-low {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.8) 0%, rgba(6, 95, 70, 0.5) 100%);
        border: 1px solid rgba(52, 211, 153, 0.5);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.2);
    }

    .result-card-med {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.8) 0%, rgba(146, 64, 14, 0.5) 100%);
        border: 1px solid rgba(251, 191, 36, 0.5);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 0 25px rgba(245, 158, 11, 0.2);
    }

    .result-card-high {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.8) 0%, rgba(153, 27, 27, 0.5) 100%);
        border: 1px solid rgba(248, 113, 113, 0.5);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.25);
    }

    .result-risk-tag {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        margin-bottom: 8px;
    }

    .tag-low { background: #065f46; color: #6ee7b7; }
    .tag-med { background: #78350f; color: #fde68a; }
    .tag-high { background: #7f1d1d; color: #fca5a5; }

    .risk-headline {
        font-size: 2rem;
        font-weight: 800;
        margin: 4px 0 10px 0;
        color: white;
    }

    /* Metric Vitals Badges */
    .vital-badge {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
    }

    .vital-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .vital-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 4px 0;
    }

    .vital-status {
        font-size: 0.78rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
    }

    .status-normal { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .status-warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    .status-danger { background: rgba(239, 68, 68, 0.2); color: #f87171; }

    /* Action Checklist */
    .action-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 10px 12px;
        background: rgba(15, 23, 42, 0.5);
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 3px solid #38bdf8;
    }

    /* Submit Button Glow */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0d9488 0%, #0284c7 100%);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        box-shadow: 0 4px 15px rgba(13, 148, 136, 0.35);
        transition: all 0.25s ease;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(13, 148, 136, 0.55);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "pipeline_bundle.joblib"
if not BUNDLE_PATH.exists():
    BUNDLE_PATH = BASE_DIR.parent / "models" / "pipeline_bundle.joblib"

LABEL_NAMES = ["Low", "Medium", "High"]


@st.cache_resource
def load_artifacts():
    bundle = None
    if BUNDLE_PATH.exists():
        try:
            bundle = joblib.load(BUNDLE_PATH)
            return bundle
        except Exception:
            bundle = None

    try:
        model = joblib.load(BASE_DIR / "best_model.pkl")
    except Exception:
        try:
            model = joblib.load(BASE_DIR / "disease_risk_model.pkl")
        except Exception:
            model = joblib.load(BASE_DIR.parent / "models" / "disease_risk_model.pkl")

    try:
        scaler = joblib.load(BASE_DIR / "feature_scaler.pkl")
    except Exception:
        scaler = joblib.load(BASE_DIR.parent / "models" / "feature_scaler.pkl")

    return {
        "best_model": model,
        # Derived from the actual loaded model object (e.g. "LogisticRegression"),
        # not a hardcoded guess — this is what was silently wrong before, since
        # the UI displayed a fixed "SVM" fallback regardless of which model
        # was really loaded and used for prediction.
        "best_model_name": type(model).__name__,
        "scaler": scaler,
        "selected_features": getattr(scaler, "feature_names_in_", None),
        "prototype_5_features": ["blood_sugar_mg_dl", "cholesterol_mg_dl", "age", "bmi", "systolic_bp"]
    }


bundle = load_artifacts()

# -------------------------------------------------------------
# Sidebar: Patient Presets & Diagnostic Configuration
# -------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
            <div style="background: linear-gradient(135deg, #0d9488 0%, #0284c7 100%); width: 52px; height: 52px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 26px; box-shadow: 0 4px 15px rgba(13, 148, 136, 0.45); border: 1px solid rgba(255,255,255,0.15);">
                🏥
            </div>
            <div>
                <div style="font-weight: 800; font-size: 1.15rem; color: #f8fafc; line-height: 1.2;">SmartCare AI</div>
                <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 500;">Clinical Intelligence v2.4</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.markdown("#### ⚡ Quick Patient Presets")
    st.caption("Load verified test profiles to inspect risk stratification:")

    preset = st.radio(
        "Select Profile:",
        [
            "Custom Patient Intake",
            "🟢 Healthy Young Outpatient (Low Risk)",
            "🟠 Borderline Metabolic Adult (Medium Risk)",
            "🔴 Severe Cardiac/Diabetic Inpatient (High Risk)"
        ],
        index=0
    )

    # Preset values dictionary
    if preset == "🟢 Healthy Young Outpatient (Low Risk)":
        p_age, p_gender, p_bg, p_dept, p_diag = 24, "Female", "O+", "General Medicine", "Fever"
        p_status, p_adm, p_room, p_pay_s, p_pay_m = "Completed", "No", "Not Admitted", "Paid", "Cash"
        p_wait, p_app, p_miss, p_los, p_prev_adm = 1, 1, 0, 0, 0
        p_sbp, p_dbp, p_bs, p_chol, p_bmi = 112, 74, 88.0, 162.0, 21.4
        p_labs, p_tx, p_cfee, p_rfee, p_lfee, p_mfee = 1, 1, 1500, 0, 1200, 1800
    elif preset == "🟠 Borderline Metabolic Adult (Medium Risk)":
        p_age, p_gender, p_bg, p_dept, p_diag = 52, "Male", "A+", "Cardiology", "Hypertension"
        p_status, p_adm, p_room, p_pay_s, p_pay_m = "Completed", "No", "Not Admitted", "Paid", "Card"
        p_wait, p_app, p_miss, p_los, p_prev_adm = 4, 3, 1, 0, 1
        p_sbp, p_dbp, p_bs, p_chol, p_bmi = 138, 88, 134.0, 218.0, 28.6
        p_labs, p_tx, p_cfee, p_rfee, p_lfee, p_mfee = 2, 2, 2500, 0, 3500, 4200
    elif preset == "🔴 Severe Cardiac/Diabetic Inpatient (High Risk)":
        p_age, p_gender, p_bg, p_dept, p_diag = 71, "Male", "B+", "Cardiology", "Diabetes"
        p_status, p_adm, p_room, p_pay_s, p_pay_m = "Completed", "Yes", "ICU", "Partially Paid", "Insurance"
        p_wait, p_app, p_miss, p_los, p_prev_adm = 0, 6, 2, 5, 3
        p_sbp, p_dbp, p_bs, p_chol, p_bmi = 168, 102, 210.0, 285.0, 34.2
        p_labs, p_tx, p_cfee, p_rfee, p_lfee, p_mfee = 5, 4, 3500, 45000, 18000, 22000
    else:
        # Default Custom Intake
        p_age, p_gender, p_bg, p_dept, p_diag = 45, "Male", "O+", "Cardiology", "Chest Pain"
        p_status, p_adm, p_room, p_pay_s, p_pay_m = "Completed", "No", "Not Admitted", "Paid", "Card"
        p_wait, p_app, p_miss, p_los, p_prev_adm = 3, 2, 0, 0, 1
        p_sbp, p_dbp, p_bs, p_chol, p_bmi = 125, 82, 110.0, 190.0, 26.0
        p_labs, p_tx, p_cfee, p_rfee, p_lfee, p_mfee = 2, 2, 2000, 0, 3000, 4000
# -------------------------------------------------------------
# Top Hero Header Banner
# -------------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">
            <span>🩺</span> SmartCare Health Engine
        </div>
        <div class="hero-subtitle">
            AI-Powered Multi-Class Disease Risk Stratification & Clinical Decision Support
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------
# Main Patient Clinical Intake Form
# -------------------------------------------------------------
with st.form("patient_clinical_form"):
    st.markdown("### 📋 Patient Clinical Intake & Diagnostics Form")
    
    tabs = st.tabs([
        "👤 Demographics & Medical History",
        "🫀 Physiological Biomarkers & Vitals",
        "🏥 Hospital Operations & Financials"
    ])

    # Tab 1: Demographics & Admissions
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (Years)", min_value=0, max_value=120, value=int(p_age), help="Patient chronological age")
            gender_options = ["Male", "Female"]
            gender = st.selectbox("Biological Sex", gender_options, index=gender_options.index(p_gender))
            bg_options = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
            blood_group = st.selectbox("ABO Blood Group", bg_options, index=bg_options.index(p_bg))
            dept_options = ["Cardiology", "General Medicine", "Laboratory Services", "Neurology", "Orthopedics", "Pediatrics", "Radiology"]
            department = st.selectbox("Consulting Department", dept_options, index=dept_options.index(p_dept))

        with col2:
            diag_options = ["Asthma", "Back Pain", "Chest Pain", "Diabetes", "Fever", "Fracture", "Hypertension", "Kidney Infection", "Migraine", "Pneumonia"]
            diagnosis = st.selectbox("Primary Clinical Diagnosis", diag_options, index=diag_options.index(p_diag))
            status_options = ["Completed", "Scheduled", "No-Show", "Cancelled"]
            appointment_status = st.selectbox("Appointment Status", status_options, index=status_options.index(p_status))
            adm_options = ["No", "Yes"]
            admitted = st.selectbox("Inpatient Admission Required?", adm_options, index=adm_options.index(p_adm))
            room_options = ["Not Admitted", "General Ward", "Private Room", "ICU"]
            room_type = st.selectbox("Room / Ward Assignment", room_options, index=room_options.index(p_room))

    # Tab 2: Physiological Biomarkers
    with tabs[1]:
        col3, col4 = st.columns(2)
        with col3:
            systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=70, max_value=240, value=int(p_sbp), help="AHA guideline cutoff: 130 mmHg")
            diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=150, value=int(p_dbp), help="AHA guideline cutoff: 80 mmHg")
            blood_sugar = st.number_input("Fasting / Random Blood Sugar (mg/dL)", min_value=40.0, max_value=500.0, value=float(p_bs), step=1.0, help="ADA diabetic cutoff: >126 mg/dL")

        with col4:
            cholesterol = st.number_input("Serum Total Cholesterol (mg/dL)", min_value=80.0, max_value=500.0, value=float(p_chol), step=1.0, help="Desirable cutoff: <200 mg/dL")
            bmi = st.number_input("Body Mass Index (BMI kg/m²)", min_value=10.0, max_value=65.0, value=float(p_bmi), step=0.1, help="Overweight >= 25, Obese >= 30")
            previous_admissions = st.number_input("Prior Hospital Admissions (Past 12 Months)", min_value=0, max_value=25, value=int(p_prev_adm), help=">=2 indicates chronic status")

    # Tab 3: Operations & Financials
    with tabs[2]:
        col5, col6 = st.columns(2)
        with col5:
            waiting_days = st.number_input("Waiting Days for Appointment", min_value=0, max_value=90, value=int(p_wait))
            previous_appointments = st.number_input("Lifetime Completed Appointments", min_value=0, max_value=50, value=int(p_app))
            missed_previous_appointments = st.number_input("Historical Missed Appointments (No-Shows)", min_value=0, max_value=30, value=int(p_miss))
            length_of_stay_days = st.number_input("Length of Inpatient Stay (Days)", min_value=0, max_value=90, value=int(p_los))
            lab_tests_count = st.number_input("Diagnostic Lab Tests Ordered", min_value=0, max_value=30, value=int(p_labs))
            treatments_count = st.number_input("Clinical Interventions & Procedures", min_value=0, max_value=30, value=int(p_tx))

        with col6:
            pay_s_options = ["Paid", "Partially Paid", "Unpaid"]
            payment_status = st.selectbox("Billing Payment Status", pay_s_options, index=pay_s_options.index(p_pay_s))
            pay_m_options = ["Card", "Cash", "Insurance", "Online"]
            payment_method = st.selectbox("Primary Payment Method", pay_m_options, index=pay_m_options.index(p_pay_m))
            consultation_fee = st.number_input("Consultation Fee (LKR)", min_value=0, max_value=50000, value=int(p_cfee), step=500)
            room_charge = st.number_input("Room Accommodation Charges (LKR)", min_value=0, max_value=500000, value=int(p_rfee), step=1000)
            lab_charge = st.number_input("Laboratory Diagnostics Fee (LKR)", min_value=0, max_value=300000, value=int(p_lfee), step=500)
            medicine_charge = st.number_input("Pharmaceutical Charges (LKR)", min_value=0, max_value=300000, value=int(p_mfee), step=500)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Execute Clinical Risk Assessment", use_container_width=True)

# -------------------------------------------------------------
# Prediction & Inference Execution
# -------------------------------------------------------------
if submitted or preset != "Custom Patient Intake":
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
        scaler = bundle.get("scaler")
        fallback_features = bundle.get("prototype_5_features") or getattr(scaler, "feature_names_in_", None)
        if fallback_features is None:
            fallback_features = ["blood_sugar_mg_dl", "cholesterol_mg_dl", "age", "bmi", "systolic_bp"]
        else:
            fallback_features = list(fallback_features)
        df_selected = pd.DataFrame([{f: float(patient_dict.get(f, 0.0)) for f in fallback_features}])
        X_input = pd.DataFrame(scaler.transform(df_selected), columns=fallback_features)

    # Predict Class and Probabilities
    pred_idx = int(model.predict(X_input)[0])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
    else:
        probs = np.array([1.0 if i == pred_idx else 0.0 for i in range(3)])

    pred_label = LABEL_NAMES[pred_idx]
    confidence = probs[pred_idx]

    # Clinical categories for summary display
    bp_cat = "Normal" if (systolic_bp < 120 and diastolic_bp < 80) else ("Elevated" if systolic_bp < 130 and diastolic_bp < 80 else "Hypertension")
    bs_cat = "Normal" if blood_sugar < 100 else ("Pre-Diabetes" if blood_sugar <= 125 else "Diabetic Range")
    chol_cat = "Desirable" if cholesterol < 200 else ("Borderline High" if cholesterol < 240 else "High Risk")
    bmi_cat = "Normal" if bmi < 25 else ("Overweight" if bmi < 30 else "Obese")
    is_chronic = int(previous_admissions >= 2)

    # -------------------------------------------------------------
    # Render High-Impact Risk Stratification Results
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 📊 Clinical Risk Stratification Report")

    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        if pred_label == "Low":
            st.markdown(
                f"""
                <div class="result-card-low">
                    <span class="result-risk-tag tag-low">🟢 Stratification Level 0</span>
                    <div class="risk-headline">LOW DISEASE RISK</div>
                    <p style="color: #a7f3d0; font-size: 1.05rem; margin-bottom: 16px;">
                        Model Confidence: <strong>{confidence:.1%}</strong> · Patient exhibits stable biomarker profiles.
                    </p>
                    <div class="action-item">
                        <span>✅</span>
                        <div><strong>Care Pathway:</strong> Routine annual preventive wellness screening and outpatient lifestyle maintenance.</div>
                    </div>
                    <div class="action-item">
                        <span>📅</span>
                        <div><strong>Follow-Up Window:</strong> 12 Months for general health review.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif pred_label == "Medium":
            st.markdown(
                f"""
                <div class="result-card-med">
                    <span class="result-risk-tag tag-med">🟠 Stratification Level 1</span>
                    <div class="risk-headline">MEDIUM DISEASE RISK</div>
                    <p style="color: #fde68a; font-size: 1.05rem; margin-bottom: 16px;">
                        Model Confidence: <strong>{confidence:.1%}</strong> · Elevated cardiovascular/metabolic parameters detected.
                    </p>
                    <div class="action-item">
                        <span>⚠️</span>
                        <div><strong>Care Pathway:</strong> Schedule comprehensive metabolic panel, HbA1c testing, and dietary intervention.</div>
                    </div>
                    <div class="action-item">
                        <span>📅</span>
                        <div><strong>Follow-Up Window:</strong> 3 Months clinical outpatient consultation.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-card-high">
                    <span class="result-risk-tag tag-high">🔴 Stratification Level 2 — CRITICAL</span>
                    <div class="risk-headline">HIGH DISEASE RISK</div>
                    <p style="color: #fca5a5; font-size: 1.05rem; margin-bottom: 16px;">
                        Model Confidence: <strong>{confidence:.1%}</strong> · Multiple compound high-risk clinical vitals present.
                    </p>
                    <div class="action-item">
                        <span>🚨</span>
                        <div><strong>Care Pathway:</strong> Immediate senior clinician evaluation, continuous telemetry or urgent admission review.</div>
                    </div>
                    <div class="action-item">
                        <span>📅</span>
                        <div><strong>Follow-Up Window:</strong> Immediate / 24–48 Hours acute surveillance.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with res_col2:
        st.markdown("#### 📈 Multi-Class Probability Distribution")
        prob_df = pd.DataFrame({
            "Risk Tier": ["Low Risk (Tier 0)", "Medium Risk (Tier 1)", "High Risk (Tier 2)"],
            "Probability": [probs[0], probs[1], probs[2]]
        })

        for idx, row in prob_df.iterrows():
            tier = row["Risk Tier"]
            p_val = row["Probability"]
            color = "#10b981" if "Low" in tier else ("#f59e0b" if "Medium" in tier else "#ef4444")
            st.markdown(
                f"""
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.88rem; font-weight: 600; margin-bottom: 4px;">
                        <span>{tier}</span>
                        <span style="color: {color};">{p_val:.1%}</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); height: 10px; border-radius: 5px; overflow: hidden;">
                        <div style="background: {color}; width: {p_val*100:.1f}%; height: 100%; border-radius: 5px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # -------------------------------------------------------------
    # Clinical Biomarker Cards Grid
    # -------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🫀 Physiological Biomarkers & Clinical Staging")
    
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)

    with b_col1:
        st_class = "status-normal" if bp_cat == "Normal" else ("status-warning" if bp_cat == "Elevated" else "status-danger")
        st.markdown(
            f"""
            <div class="vital-badge">
                <div class="vital-title">Blood Pressure</div>
                <div class="vital-value">{systolic_bp:.0f}/{diastolic_bp:.0f} <span style="font-size:0.75rem; color:#94a3b8;">mmHg</span></div>
                <span class="vital-status {st_class}">{bp_cat}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b_col2:
        st_class = "status-normal" if bs_cat == "Normal" else ("status-warning" if bs_cat == "Pre-Diabetes" else "status-danger")
        st.markdown(
            f"""
            <div class="vital-badge">
                <div class="vital-title">Blood Glucose</div>
                <div class="vital-value">{blood_sugar:.0f} <span style="font-size:0.75rem; color:#94a3b8;">mg/dL</span></div>
                <span class="vital-status {st_class}">{bs_cat}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b_col3:
        st_class = "status-normal" if chol_cat == "Desirable" else ("status-warning" if chol_cat == "Borderline High" else "status-danger")
        st.markdown(
            f"""
            <div class="vital-badge">
                <div class="vital-title">Total Cholesterol</div>
                <div class="vital-value">{cholesterol:.0f} <span style="font-size:0.75rem; color:#94a3b8;">mg/dL</span></div>
                <span class="vital-status {st_class}">{chol_cat}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b_col4:
        st_class = "status-normal" if bmi_cat == "Normal" else ("status-warning" if bmi_cat == "Overweight" else "status-danger")
        st.markdown(
            f"""
            <div class="vital-badge">
                <div class="vital-title">Body Mass Index</div>
                <div class="vital-value">{bmi:.1f} <span style="font-size:0.75rem; color:#94a3b8;">kg/m²</span></div>
                <span class="vital-status {st_class}">{bmi_cat}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------------------
    # Standardized Feature Vector & Export Summary
    # -------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    exp_col1, exp_col2 = st.columns([1, 1])

    with exp_col1:
        with st.expander("🔍 View Processed Feature Vector (Model Input)"):
            st.caption("Fitted Z-Score Standardized Features passed to classifier:")
            st.dataframe(X_input.T.rename(columns={0: "Standardized Value"}).round(4), use_container_width=True)

    with exp_col2:
        report_summary = (
            f"SMARTCARE HOSPITAL — CLINICAL RISK STRATIFICATION REPORT\n"
            f"----------------------------------------------------\n"
            f"Assessed Risk Tier: {pred_label.upper()} RISK (Confidence: {confidence:.1%})\n"
            f"Class Probabilities: Low={probs[0]:.1%}, Medium={probs[1]:.1%}, High={probs[2]:.1%}\n\n"
            f"Patient Summary:\n"
            f"  - Age / Sex: {age} yrs / {gender}\n"
            f"  - Department: {department} | Diagnosis: {diagnosis}\n"
            f"  - Blood Pressure: {systolic_bp:.0f}/{diastolic_bp:.0f} mmHg ({bp_cat})\n"
            f"  - Blood Sugar: {blood_sugar:.0f} mg/dL ({bs_cat})\n"
            f"  - Cholesterol: {cholesterol:.0f} mg/dL ({chol_cat})\n"
            f"  - BMI: {bmi:.1f} kg/m² ({bmi_cat})\n"
            f"  - Prior Admissions: {previous_admissions}\n"
            f"----------------------------------------------------\n"
            f"Model Engine: {bundle.get('best_model_name', 'Logistic Regression') or (type(model).__name__ if model else 'Logistic Regression')}\n"
        )
        st.download_button(
            label="📥 Download Clinical Risk Summary (.txt)",
            data=report_summary,
            file_name=f"SmartCare_Risk_Report_{diagnosis}_{pred_label}.txt",
            mime="text/plain",
            use_container_width=True
        )

# -------------------------------------------------------------
# Footer Disclaimer
# -------------------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ **Clinical Disclaimer:** This application provides machine learning-assisted clinical risk stratification for clinical decision support evaluation. "
    "Predictions should always be verified by certified medical personnel before formulating diagnostic or treatment plans."
)
