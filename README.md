# 🏥 SmartCare Hospital — Disease Risk Classification

> **CCS3440 Artificial Intelligence Coursework | SLTC | Group 02**

An end-to-end Machine Learning pipeline that classifies hospital patients into **Low**, **Medium**, or **High** disease risk levels using clinical, demographic, and operational data from SmartCare Hospital.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Pipeline Overview & Data Leakage Prevention](#-pipeline-overview--data-leakage-prevention)
- [Models & Results](#-models--results)
- [Class-Weighting Ablation Study](#-class-weighting-ablation-study)
- [Explainable AI (SHAP)](#-explainable-ai-shap)
- [Prototype & Deployment](#-prototype--deployment)
- [Getting Started](#-getting-started)
- [Team](#-team)
- [License](#-license)

---

## 🎯 Problem Statement

**Option C — Disease Risk Classification (Multi-Class)**

Early identification of disease risk can improve preventive healthcare interventions. This project develops AI models to classify patients into three risk categories based on clinical measurements, hospital operations data, and financial records.

| Item | Detail |
|------|--------|
| **Target Variable** | `disease_risk_level` |
| **Problem Type** | Multi-Class Classification |
| **Classes & Test Support** | **Low** (N=26) · **Medium** (N=94) · **High** (N=80) |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **File** | `smartcare_ai_dataset_1000.csv` |
| **Total Records** | 1,000 |
| **Features** | Patient demographics, clinical vitals, hospital operations, financial data |
| **Class Ratio** | Medium (46.9%) · High (40.0%) · Low (13.1%) |

---

## 📁 Project Structure

```
SmartCare-Hospital/
├── README.md
├── requirements.txt
├── Dockerfile
├── data/
│   ├── raw/
│   │   └── smartcare_ai_dataset_1000.csv
│   ├── processed/
│   │   ├── smartcare_cleaned.csv
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   └── data_dictionary.csv
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── Task02_Dataset_Understanding.py
│   ├── Task03_Data_Preprocessing_and_Feature_Engineering.py
│   ├── Task04_Exploratory_Data_Analysis.py
│   ├── Task05_Model_Development.py
│   ├── Task06_Model_Evaluation.py
│   ├── Task07_Explainable_AI_Analysis.py
│   └── Task08_AI_Prototype_Development.py
├── app/
│   ├── app.py
│   ├── requirements.txt
│   └── pipeline_bundle.joblib
├── models/
│   ├── best_model.pkl
│   ├── all_tuned_models.pkl
│   ├── pipeline_bundle.joblib
│   └── feature_artifacts.joblib
└── reports/
    ├── Task_09_Final_Technical_Report.md
    ├── task05_model_comparison_results.csv
    ├── task06_model_comparison_table.csv
    ├── task06_per_class_metrics.csv
    ├── class_weighting_ablation_study.csv
    ├── task08_prototype_comparison.csv
    ├── shap_feature_importance.csv
    ├── confusion_matrices_all_models.png
    ├── eval_roc_curves_best_model.png
    ├── shap_summary_multiclass.png
    ├── shap_high_risk_importance.png
    └── shap_waterfall_patient_example.png
```

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ |
| **Environment** | Jupyter Notebook / VS Code |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-Learn, XGBoost, TensorFlow |
| **Explainable AI** | SHAP (SHapley Additive exPlanations) |
| **Prototype** | Streamlit |

---

## 🔄 Pipeline Overview & Data Leakage Prevention

```
Raw Data → Conditional Imputation → Duplicate Removal → [STRATIFIED TRAIN/TEST SPLIT (80/20)]
                                                               │
                       ┌───────────────────────────────────────┴───────────────────────────────────────┐
                       ▼                                                                               ▼
              Fit ONLY on Train (N=800)                                                     Transform ONLY on Test (N=200)
    • OneHotEncoder (Nominals: gender, dept, etc.)                                  • OneHotEncoder transform
    • SelectKBest ANOVA F-score (Top 15 Features)                                   • SelectKBest transform
    • StandardScaler                                                                • StandardScaler transform
```

| Task | Description | Status |
|------|-------------|--------|
| **Task 01** | Problem Definition & Literature Review | ✅ Complete |
| **Task 02** | Dataset Understanding | ✅ Complete |
| **Task 03** | Data Preprocessing & Leak-Free Feature Pipeline | ✅ Complete |
| **Task 04** | Exploratory Data Analysis (EDA) | ✅ Complete |
| **Task 05** | ML Model Development & Ablation Study | ✅ Complete |
| **Task 06** | Multi-Class Model Evaluation & Benchmarking | ✅ Complete |
| **Task 07** | Explainable AI (SHAP Analysis) | ✅ Complete |
| **Task 08** | AI Prototype & 5-Feature Generalization Benchmark | ✅ Complete |
| **Task 09** | Technical Report Deliverable | ✅ Complete |

---

## 🤖 Models & Results

Five classification models were trained using **Stratified 5-Fold Cross-Validation** (macro-F1 scoring) on the leak-free training set (`N=800`) and evaluated on the held-out test set (`N=200`).

### Held-Out Test Set Benchmark (N=200)

| Rank | Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (OvR macro) |
|------|-------|----------|-------------------|----------------|------------|---------------------|
| 🥇 1 | **Logistic Regression** | **0.990** | **0.9762** | **0.9929** | **0.9841** | **0.9999** |
| 🥈 2 | **SVM (Linear Kernel)** | 0.990 | 0.9762 | 0.9929 | 0.9841 | 0.9998 |
| 🥉 3 | **XGBoost** | 0.875 | 0.8858 | 0.8223 | 0.8467 | 0.9660 |
| 4 | **Random Forest** | 0.820 | 0.8407 | 0.8049 | 0.8190 | 0.9500 |
| 5 | **Decision Tree** | 0.715 | 0.7060 | 0.6995 | 0.7017 | 0.7686 |

### Per-Class Performance (Best Model — Logistic Regression)

| Class Name | Target ID | Support | Precision | Recall | F1 Score |
|------------|-----------|---------|-----------|--------|----------|
| **Low Risk** | 0 | 26 | 0.9286 | 1.0000 | 0.9630 |
| **Medium Risk** | 1 | 94 | 1.0000 | 0.9787 | 0.9892 |
| **High Risk** | 2 | 80 | 1.0000 | 1.0000 | 1.0000 |

---

## ⚖️ Class-Weighting Ablation Study

To evaluate the impact of class imbalance handling (`class_weight='balanced'`), an ablation study was conducted comparing weighted models against unweighted baselines on the held-out test set:

| Model Family | Unweighted Acc | Unweighted Macro-F1 | Unweighted Low-F1 | Weighted Acc | Weighted Macro-F1 | Weighted Low-F1 |
|--------------|----------------|---------------------|-------------------|--------------|-------------------|-----------------|
| **Logistic Regression** | 0.985 | 0.9751 | 0.9412 | **0.990** | **0.9841** | **0.9630** |
| **SVM** | 0.900 | 0.8751 | 0.7917 | **0.990** | **0.9841** | **0.9630** |
| **Random Forest** | 0.835 | 0.8052 | 0.7143 | **0.820** | **0.8190** | **0.8163** |
| **Decision Tree** | 0.770 | 0.7757 | 0.7925 | 0.715 | 0.7017 | 0.6538 |
| **XGBoost** | 0.875 | 0.8546 | 0.7907 | 0.875 | 0.8467 | 0.7556 |

*Key Finding: Class weighting significantly improves minority class (`Low Risk`) recall and F1-score for linear model families (LR and SVM).*

---

## 💡 Explainable AI (SHAP)

Feature importance and patient-level explanations were computed using SHAP values on the saved best pipeline model:

- **Top Clinical Feature Drivers**:
  1. `blood_sugar_mg_dl` (Mean |SHAP| = 4.28)
  2. `cholesterol_mg_dl` (Mean |SHAP| = 4.00)
  3. `age` (Mean |SHAP| = 3.66)
  4. `bmi` (Mean |SHAP| = 3.47)
  5. `previous_admissions` (Mean |SHAP| = 2.34)
  6. `systolic_bp` (Mean |SHAP| = 1.89)

Visualizations generated:
- `reports/shap_summary_multiclass.png`: Multi-class beeswarm/summary plot.
- `reports/shap_high_risk_importance.png`: Feature drivers for High-Risk patients.
- `reports/shap_waterfall_patient_example.png`: Individual patient waterfall explanation.

---

## 🖥 Prototype & Deployment

A **Streamlit** interactive web application allows clinicians to input patient attributes and view real-time risk predictions with class probabilities and vitals alerts.

### 5-Feature Prototype Generalization Evaluation
Before deployment, a simplified 5-feature prototype model (`age`, `blood_sugar`, `cholesterol`, `bmi`, `previous_admissions`) was evaluated on held-out test data:

| Model System | Features Used | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|--------------|---------------|----------|-------------------|----------------|------------|
| **Full Pipeline Model** | 15 | **0.990** | **0.9762** | **0.9929** | **0.9841** |
| **5-Feature Prototype** | 5 | **0.900** | **0.8783** | **0.9080** | **0.8910** |

---

## 🚀 Getting Started

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Ravindi373/SmartCare-Hospital.git
cd SmartCare-Hospital

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the leak-free ML pipeline in order
python src/Task03_Data_Preprocessing_and_Feature_Engineering.py
python src/Task05_Model_Development.py
python src/Task06_Model_Evaluation.py
python src/Task07_Explainable_AI_Analysis.py
python src/Task08_AI_Prototype_Development.py

# 4. Launch the Streamlit application
streamlit run app/app.py
```

---

## 👥 Team

**Group 02 — SLTC**

| Role | Name |
|------|------|
| **Module** | CCS3440 — Artificial Intelligence |
| **Lecturer in Charge** | Dr. Chameera De Silva |
| **Teaching Assistants** | Mr. Chamod Hewage · Pamod Dilshan |

---

## 📜 License

This project is developed for academic purposes as part of the **CCS3440 — Artificial Intelligence** module at **Sri Lanka Technological Campus (SLTC)**.

> ⚠️ **Disclaimer**: This system is an academic machine learning coursework project trained on a synthetic clinical dataset. It is **not** a certified clinical decision-support tool and should not be used for actual medical diagnosis.
