# 🏥 SmartCare Hospital — Disease Risk Classification

> **CCS3440 Artificial Intelligence Coursework | SLTC | Group 02**

An end-to-end Machine Learning pipeline that classifies hospital patients into **Low**, **Medium**, or **High** disease risk levels using clinical, demographic, and operational data from SmartCare Hospital.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Pipeline Overview](#-pipeline-overview)
- [Models & Results](#-models--results)
- [Prototype](#-prototype)
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
| **Classes** | Low · Medium · High |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **File** | `smartcare_ai_dataset_1000.csv` |
| **Records** | 1,000 |
| **Features** | Patient demographics, clinical vitals, hospital operations, financial data |

<details>
<summary><b>Feature Categories</b></summary>

- **Patient Info** — Age, Gender, Blood Group
- **Clinical** — Diagnosis, Blood Pressure, Blood Sugar, Cholesterol, BMI
- **Operations** — Department, Appointment History, Admissions, Length of Stay, Room Type, Treatments, Lab Tests
- **Financial** — Consultation, Lab, Room & Medicine Charges, Total Bill

</details>

---

## 📁 Project Structure

```
SmartCare-Hospital/
├── README.md
├── data/
│   ├── raw/
│   │   └── smartcare_ai_dataset_1000.csv
│   ├── processed/
│   │   ├── cleaned_data.csv
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   └── data_dictionary.csv
├── src/
│   ├── Task 02 – Dataset Understanding.py
│   ├── Task 03 – Data Preprocessing and Feature Engineering.py
│   ├── Task 04 – Exploratory Data Analysis.py
│   ├── Task05_Model_Development.py
│   └── Task06_Model_Evaluation.py
├── app/
│   ├── app.py
│   └── requirements.txt
```

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.x |
| **Environment** | Jupyter Notebook / Google Colab |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-Learn, XGBoost |
| **Explainable AI** | SHAP |
| **Prototype** | Streamlit |

---

## 🔄 Pipeline Overview

```
Task 02          Task 03                    Task 04          Task 05            Task 06         Task 07        Task 08
Dataset    →   Preprocessing &    →     Exploratory    →   Model        →    Model       →   Explainable  →  Prototype
Understanding    Feature Engineering      Data Analysis     Development       Evaluation       AI (SHAP)       (Streamlit)
```

| Task | Description | Status |
|------|-------------|--------|
| **Task 01** | Problem Definition & Literature Review | 📄 Report |
| **Task 02** | Dataset Understanding | ✅ Complete |
| **Task 03** | Data Preprocessing & Feature Engineering | ✅ Complete |
| **Task 04** | Exploratory Data Analysis | ✅ Complete |
| **Task 05** | ML Model Development (5 models) | ✅ Complete |
| **Task 06** | Model Evaluation | ✅ Complete |
| **Task 07** | Explainable AI (SHAP/LIME) | 🔧 In Progress |
| **Task 08** | AI Prototype (Streamlit) | ✅ Complete |
| **Task 09** | Technical Report | 📄 Report |

---

## 🤖 Models & Results

Five classification models were trained with **GridSearchCV** (Stratified 5-Fold CV, macro-F1 scoring):

| # | Model | Highlights |
|---|-------|-----------|
| 1 | **Logistic Regression** | L2 regularization, balanced class weights |
| 2 | **Decision Tree** | Pruned with max depth, class-weighted |
| 3 | **Random Forest** | Ensemble of decision trees, feature importance |
| 4 | **Support Vector Machine (SVM)** | RBF kernel, class-weighted |
| 5 | **XGBoost** | Gradient boosting, sample-weighted |

### Evaluation Metrics

All models are evaluated using:
- ✅ Accuracy
- ✅ Precision (macro)
- ✅ Recall (macro)
- ✅ F1 Score (macro)
- ✅ ROC-AUC (One-vs-Rest, macro)
- ✅ Confusion Matrix
- ✅ Per-class Classification Report

---

## 🖥 Prototype

A **Streamlit** web application that accepts patient details and predicts the disease risk level in real-time.

### Features
- 🩺 Patient & clinical data input form
- 🔍 Real-time disease risk prediction (Low / Medium / High)
- 📊 Class probability bar chart
- 🧠 SHAP-based feature importance insights

### Run Locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/sankalpams/SmartCare-Hospital.git
cd SmartCare-Hospital

# 2. Install dependencies
pip install -r app/requirements.txt

# 3. Run the notebooks/scripts in order
# Open in Jupyter or run:
python src/Task05_Model_Development.py
python src/Task06_Model_Evaluation.py

# 4. Launch the prototype
cd app
streamlit run app.py
```

---

## 👥 Team

**Group 02 — SLTC**

| Role | Name |
|------|------|
| **Lecturer in Charge** | Dr. Chameera De Silva |
| **Teaching Assistants** | Mr. Chamod Hewage · Pamod Dilshan |

---

## 📜 License

This project is developed for academic purposes as part of the **CCS3440 — Artificial Intelligence** module at **Sri Lanka Technological Campus (SLTC)**.

> ⚠️ **Disclaimer**: This is a coursework project using a synthetic dataset. It is **not** a real clinical decision-support tool and should not be used for actual medical decisions.