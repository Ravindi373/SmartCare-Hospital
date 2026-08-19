# 🏥 SmartCare Hospital — Disease Risk Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-blue)
![SHAP](https://img.shields.io/badge/SHAP-0.45%2B-purple)
![Pandas](https://img.shields.io/badge/Pandas-2.2%2B-darkblue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **CCS3440 Artificial Intelligence Final Coursework | SLTC | Group 02**

An end-to-end, leakage-free machine learning pipeline that stratifies SmartCare Hospital patients into **Low**, **Medium**, or **High** disease risk levels using physiological biomarkers, clinical diagnoses, hospital operations, and financial records — built with a strict train-only-fitting protocol, six benchmarked models, true SHAP explainability, and a deployment-ready lightweight prototype.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Leakage-Free Pipeline](#-leakage-free-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Models & Results](#-models--results)
- [Explainable AI (SHAP)](#-explainable-ai-shap)
- [Deployment Prototype](#-deployment-prototype)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Limitations](#-limitations)
- [Team](#-team)

---

## 🎯 Problem Statement

**Option C — Multi-Class Disease Risk Classification**

| Item | Specification |
|---|---|
| **Target Variable** | `disease_risk_level` |
| **Problem Type** | Multi-class classification (3 classes) |
| **Class Distribution** | Low: 13.1% (N=131) · Medium: 46.9% (N=469) · High: 40.0% (N=400) |
| **Test Set** | Low: N=26 · Medium: N=94 · High: N=80 (Total N=200, 20% stratified) |

Early identification of disease risk supports preventive care, helps triage clinical staff and resources, and assists capacity planning — as a decision-support layer, not a replacement for clinical judgement.

---

## 📊 Dataset

| Property | Value |
|---|---|
| **File** | `smartcare_ai_dataset_1000.csv` |
| **Records** | 1,000 patient admissions/visits |
| **Raw Features** | 33 attributes (demographics, clinical vitals, operations, billing) |
| **Missing Data** | Only `room_type` (90.6% complete — mix of structural and genuine gaps) |
| **Duplicates** | 0 (verified on full rows, `patient_id`, `record_id`) |

**Feature categories:** Demographics (age, gender, blood group) · Physiological vitals (BP, blood sugar, cholesterol, BMI) · Hospital operations (department, diagnosis, admission, length of stay) · Financials (consultation/room/lab/medicine charges, payment status)

---

## 🔄 Leakage-Free Pipeline

1. **Split-first protocol** — stratified 80/20 train/test split performed *before* any transformation.
2. **Train-only fitting** — imputation, encoding, SelectKBest (K=15), and StandardScaler fitted strictly on `X_train` (N=800), applied via `.transform()` to `X_test` (N=200).
3. **Leakage columns dropped** — `record_id`, `patient_id`, `appointment_date`, `no_show`, `readmitted_30_days` explicitly removed.

```
Raw (N=1000) → Stratified 80/20 Split ─┬─ Train (N=800) → Fit(Impute+OHE+SelectKBest+Scale) → Train Models
                                        └─ Test  (N=200) → Transform (fitted pipeline) → Held-out Eval
```

Target mapping is deterministic: **Low = 0, Medium = 1, High = 2** (avoids scikit-learn's alphabetical `LabelEncoder` mismatch).

---

## 🛠 Feature Engineering

Seven engineered features raised the column count from 33 → 40:

- `age_group`, `bmi_category`, `bp_category` — clinically ordered bands (ordinal-encoded)
- `missed_appointment_rate`, `is_chronic_patient`, `care_intensity`, `appointment_month`

Nominal fields (`gender`, `blood_group`, `department`, `diagnosis`, `room_type`, `payment_status`, `payment_method`) use **One-Hot Encoding** — replacing `LabelEncoder` to remove artificial ordinal bias.

**Top ANOVA F-scores (SelectKBest, K=15):** blood pressure class (154.82), age (142.25), age category (55.73), BMI (51.21), hypertension flag (19.95).

---

## 🤖 Models & Results

All models tuned via `GridSearchCV` with 5-fold stratified CV, optimizing **macro-F1**.

### Test-Set Performance (N=200)

| Rank | Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (OvR) |
|---|---|---|---|---|---|---|
| 🥇 | **Logistic Regression** | **0.985** | 0.9721 | 0.9894 | **0.9802** | **0.9998** |
| 🥇 | **SVM (Linear)** | **0.985** | 0.9721 | 0.9894 | **0.9802** | **0.9998** |
| 🥉 | Neural Network (bonus MLP) | 0.970 | 0.9697 | 0.9700 | 0.9698 | — |
| 4 | XGBoost | 0.870 | 0.8979 | 0.8262 | 0.8534 | 0.9638 |
| 5 | Random Forest | 0.840 | 0.8795 | 0.8104 | 0.8362 | 0.9503 |
| 6 | Decision Tree | 0.700 | 0.6857 | 0.7142 | 0.6918 | 0.8357 |

**Best model:** Logistic Regression (saved as `best_model.pkl`) — linear boundaries fit the near-linear clinical relationships better than ensemble methods.

**Full classification report (Logistic Regression, test set):**

```
              precision   recall   f1-score   support
Low               0.99      1.00      0.99        80
Medium            0.93      1.00      0.96        26
High              1.00      0.97      0.98        94

accuracy                              0.98       200
macro avg         0.97      0.99      0.98       200
weighted avg       0.99      0.98      0.99       200
```

### Class-Weighting Ablation

Cost-sensitive weighting boosted minority-class (Low) recall in several models — e.g. **+15.4%** for Logistic Regression & SVM, **+19.2%** for Decision Tree — but had negligible effect on LR/SVM's already-strong macro-F1, showing usefulness depends on the model rather than being universal.

### Target-Leakage Investigation

Because 98.5% accuracy is unusually high, four diagnostics were run (shallow decision tree, linear regression fit, prediction-confidence distribution, manual review of misclassifications). Findings: simple round-number thresholds separate classes well and predictions are highly confident, suggesting the dataset likely follows a rule-based generation process — a caution flagged for any real-world clinical use.

---

## 🧠 Explainable AI (SHAP)

- **Logistic Regression → LinearExplainer**, **Random Forest → TreeExplainer** (cross-model validation)
- **Top global drivers:** blood sugar, cholesterol, age, BMI, systolic BP — consistent across both explainers, with **8 of the top 10 features shared** between LR and RF
- **High-risk drivers:** elevated blood sugar (>126 mg/dL) and systolic hypertension (>140 mmHg) push predictions toward High risk
- Waterfall plots provide per-patient local explanations

---

## 📦 Deployment Prototype

A lightweight **5-feature** model (`age`, `systolic_bp`, `cholesterol_mg_dl`, `blood_sugar_mg_dl`, `bmi`) — selected directly from the SHAP results as a single source of truth — trades some accuracy for simplicity:

| Model | Features | Accuracy | Macro-F1 |
|---|---|---|---|
| Full model | 15 | 98.5% | 0.980 |
| Prototype | 5 | 90.5% | 0.90 |

Deployed as an interactive **Streamlit** clinical decision-support app (`app/app.py`), live on Streamlit Community Cloud, using `disease_risk_model.pkl` + `feature_scaler.pkl`.

**🌐 Deployed Prototype:** [smartcare-hospital-group2.streamlit.app](https://smartcare-hospital-group2.streamlit.app/)

```bash
streamlit run app/app.py
```

---

## 📁 Project Structure

```
SmartCare-Hospital/
├── data/        # Raw + processed splits, data dictionary
├── src/         # Task02–Task08 pipeline scripts
├── models/      # pipeline_bundle, best_model, prototype artefacts
├── app/         # Streamlit clinical decision support app
├── reports/     # Technical report, SHAP plots, evaluation CSVs
└── Notebook/    # Fully executed coursework notebook
```

## 🚀 Getting Started

```bash
git clone https://github.com/Ravindi373/SmartCare-Hospital.git
cd SmartCare-Hospital
pip install -r requirements.txt
streamlit run app/app.py
```

---

## ⚠️ Limitations

- Possible deterministic/rule-based label construction in the synthetic dataset
- Small sample size (1,000 records) — may not generalize across hospitals
- Medium-risk class harder to separate due to overlap with Low/High
- 5-feature prototype trades ~8 accuracy points for usability
- Not validated for fairness across patient subgroups; intended as clinical decision *support*, not an autonomous diagnostic tool

---

## 👥 Team

**Group 02 — SLTC | CCS3440 Artificial Intelligence**

Ravindi Ayodhya · Sithumi Jayarathna · Thimeth Chathnuka · Malith Shehan · Ashan Gamage

> ⚠️ Developed for academic evaluation only. Not intended as an autonomous medical diagnostic tool.

## License

MIT
