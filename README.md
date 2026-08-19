# 🏥 SmartCare Hospital — Disease Risk Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-337AB7.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.45%2B-8A2BE2.svg)](https://shap.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Problem Statement

### **Multi-Class Disease Risk Classification**

Early clinical identification of disease risk levels enables timely medical interventions, optimizing inpatient bed allocation and improving preventive outpatient care.

| Item | Specification |
|------|---------------|
| **Target Variable** | `disease_risk_level` |
| **Problem Type** | Multi-Class Classification (3 classes) |
| **Classes & Support** | **Low (Class 0):** 13.1% ($N=131$) · **Medium (Class 1):** 46.9% ($N=469$) · **High (Class 2):** 40.0% ($N=400$) |
| **Held-Out Test Support** | **Low:** $N=26$ · **Medium:** $N=94$ · **High:** $N=80$ (Total $N=200$, 20% stratified test set) |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **File** | `smartcare_ai_dataset_1000.csv` |
| **Records** | 1,000 patient admissions/visits |
| **Features** | 32 initial raw attributes (clinical vitals, operations, billing, demographics) |

### Feature Categories
- **Demographics:** Age, Gender, Blood Group
- **Physiological Vitals:** Systolic BP, Diastolic BP, Blood Sugar (mg/dL), Cholesterol (mg/dL), BMI
- **Hospital Operations:** Department, Primary Diagnosis, Admission Status, Room Type, Length of Stay, Appointment History
- **Billing & Financials:** Consultation, Room, Lab, and Medicine Charges, Total Bill (LKR), Payment Status/Method

---

## 🔄 Leakage-Free Pipeline Architecture

To guarantee strict adherence to machine learning standards and prevent data leakage:
1. **Split-First Protocol:** Stratified 80/20 train/test split is performed **prior** to any data transformation.
2. **Train-Only Fitting:** Imputation, One-Hot Encoding, ANOVA F-Score feature selection ($K=15$), and standard scaling are fitted **strictly on `X_train`** ($N=800$) and applied via `.transform()` to `X_test` ($N=200$).
3. **Identifier & Outcome Leakage Dropping:** `record_id`, `patient_id`, `appointment_date`, `no_show`, and `readmitted_30_days` are explicitly removed.

```
Raw Data (N=1000) ──> Stratified 80/20 Split ──┬─> Train (N=800) ──> Fit (OHE + SelectKBest + Scaler) ──> Train Model
                                               └─> Test  (N=200) ──> Transform with Fitted Pipeline  ──> Held-Out Eval
```

---

## 🏷 Categorical Encoding & Target Mapping

- **Nominal Categorical Encoding:** Replaced `LabelEncoder` with `OneHotEncoder(handle_unknown='ignore')` across all nominal variables (`gender`, `blood_group`, `department`, `diagnosis`, `appointment_status`, `room_type`, `payment_status`, `payment_method`) to eliminate artificial ordinal bias.
- **Target Mapping:** Deterministically mapped as:
  $$\text{Low} = 0 \quad (\text{Support } N=26), \quad \text{Medium} = 1 \quad (\text{Support } N=94), \quad \text{High} = 2 \quad (\text{Support } N=80)$$

---

## 🤖 Models & Benchmarking

All five model families were tuned via `GridSearchCV` with **Stratified 5-Fold Cross-Validation** on training data (optimizing Macro-F1) and evaluated on the identical held-out test set ($N=200$):

| Model Family | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 | ROC-AUC (OvR macro) | Primary Clinical Selection |
|--------------|:--------:|:-----------------:|:--------------:|:--------:|:-------------------:|:--------------------------:|
| **Logistic Regression (Balanced)** | **0.8150** | **0.7986** | **0.7935** | **0.8050** | **0.9379** | 🥇 **Selected Primary Model** (Best Minority Recall & XAI) |
| **SVM (Linear / RBF)** | 0.8150 | 0.7958 | 0.8150 | 0.8045 | 0.9408 | 🥈 Baseline Benchmark |
| **Random Forest** | 0.7950 | 0.8090 | 0.7538 | 0.7753 | 0.9185 | 🥉 Tree Ensemble |
| **XGBoost** | 0.7650 | 0.7433 | 0.7505 | 0.7467 | 0.9103 | Gradient Boosted Trees |
| **Decision Tree** | 0.6650 | 0.6456 | 0.6022 | 0.6164 | 0.7366 | Interpretable Single Tree |

> **Production Model Choice:** **Logistic Regression** is selected as the primary production engine for clinical deployment due to its superior minority-class recall (76.92%), direct linear probability calibration, high prototype accuracy (85.50%), and transparent SHAP explanations.

---

## ⚖️ Class Weighting Ablation Study

To prove the efficacy of cost-sensitive class balancing on the minority class (`Low Risk`, $N=26$), an empirical ablation study was performed comparing `class_weight='balanced'` against unweighted baselines:

| Model | Weighting Scheme | Overall Accuracy | Macro-F1 | Low-Risk Recall (Minority) | Low-Risk F1 |
|-------|------------------|:----------------:|:--------:|:--------------------------:|:-----------:|
| **Logistic Regression** | Unweighted Baseline | 0.8250 | 0.7947 | 61.54% | 0.6957 |
| **Logistic Regression** | **Balanced (Cost-Sensitive)** | **0.8150** | **0.8050** | **76.92% (+15.4%)** | **0.7692** |
| **Decision Tree** | Unweighted Baseline | 0.6800 | 0.6429 | 42.31% | 0.5366 |
| **Decision Tree** | **Balanced (Cost-Sensitive)** | 0.6600 | 0.6462 | 61.54% (+19.2%) | 0.5926 |
| **Random Forest** | Unweighted Baseline | 0.7750 | 0.7347 | 46.15% | 0.6154 |
| **Random Forest** | **Balanced (Cost-Sensitive)** | 0.7900 | 0.7564 | 53.85% (+7.7%) | 0.6512 |
| **SVM** | Unweighted Baseline | 0.8100 | 0.7749 | 53.85% | 0.6667 |
| **SVM** | **Balanced (Cost-Sensitive)** | 0.8050 | 0.7748 | 69.23% (+15.4%) | 0.6667 |

> **Finding:** Class weighting significantly boosts minority-class recall across all classifiers (+15.4% in Logistic Regression & SVM, +19.2% in Decision Trees) without compromising macro-averaged F1. Logistic Regression achieved the highest minority recall (76.92%).

---

## 🧠 Explainable AI (True SHAP)

True SHAP analysis (`shap.TreeExplainer`) was executed on the ensemble pipeline test dataset:
- **Global Feature Attributions:** Blood sugar ($0.1038$), cholesterol ($0.0887$), age ($0.0736$), BMI ($0.0575$), and systolic BP ($0.0440$) are the top 5 clinical biomarkers driving multi-class risk predictions.
- **High-Risk Drivers:** Elevated blood sugar ($>126$ mg/dL) and systolic hypertension ($>140$ mmHg) are the strongest positive contributors pushing patients into the High-Risk category.
- **Local Explanations:** Individual patient waterfall force plots explain specific case decisions.

---

## 📦 Deployment-Ready Model Artefact & Prototype

The final deliverable consists of a **Deployment-Ready Model Artefact bundle** (`pipeline_bundle.joblib`, `disease_risk_model.pkl`, `feature_scaler.pkl`) along with an interactive Streamlit Clinical Decision Support interface (`app/app.py`).

### 5-Feature Lightweight Prototype Evaluation
Prior to deployment, the streamlined 5-feature model (Logistic Regression) was evaluated on the held-out test set ($N=200$):
- **Features:** `blood_sugar_mg_dl`, `cholesterol_mg_dl`, `age`, `bmi`, `systolic_bp`
- **Test Performance:** Accuracy = **85.50%**, Macro-Precision = **84.04%**, Macro-Recall = **85.39%**, Macro-F1 = **84.67%**

```bash
# Launch the Streamlit demonstration interface
streamlit run app/app.py
```

---

## 📁 Project Structure

```
SmartCare-Hospital/
├── README.md                                                 # Project overview & documentation
├── requirements.txt                                          # Root environment dependencies
├── data/
│   ├── raw/smartcare_ai_dataset_1000.csv                    # Benchmark dataset (1,000 records)
│   ├── processed/                                            # Processed CSV splits (X_train, X_test, y_train, y_test)
│   └── data_dictionary.csv                                   # Feature metadata dictionary
├── src/
│   ├── preprocessing.py                                      # Clean data loader & leakage prevention
│   ├── feature_engineering.py                                # OHE, feature selection & pipeline transformers
│   ├── Task02_Dataset_Understanding.py                       # Exploratory dataset diagnostics
│   ├── Task03_Data_Preprocessing_and_Feature_Engineering.py  # Leakage-free preprocessing pipeline
│   ├── Task04_Exploratory_Data_Analysis.py                   # Clinical statistical visualizations
│   ├── Task05_Model_Development.py                           # 5 model tuning & ablation experiments
│   ├── Task06_Model_Evaluation.py                            # Multi-class metrics, confusion matrices & ROC curves
│   ├── Task07_Explainable_AI_Analysis.py                     # True SHAP XAI calculations
│   └── Task08_AI_Prototype_Development.py                    # 5-feature prototype evaluation & serialization
├── models/
│   ├── pipeline_bundle.joblib                                # Unified production pipeline bundle
│   ├── best_model.pkl                                        # Serialized top model (Logistic Regression)
│   ├── disease_risk_model.pkl                                # Serialized lightweight prototype model
│   └── feature_scaler.pkl                                    # Serialized standard scaler
├── app/
│   ├── app.py                                                # Streamlit clinical decision support interface
│   ├── pipeline_bundle.joblib                                # App pipeline bundle
│   └── requirements.txt                                      # App dependencies
├── reports/
│   ├── Task_09_Comprehensive_Technical_Report.md             # Formal coursework technical report
│   ├── task05_class_weighting_ablation.csv                   # Empirical ablation results table
│   ├── task06_model_comparison_table.csv                     # Synchronized benchmark table
│   ├── task06_per_class_metrics.csv                          # Ground truth per-class metrics
│   ├── task08_prototype_evaluation.csv                       # Prototype generalisation evaluation
│   ├── shap_summary_multiclass.png                           # Multi-class SHAP beeswarm plot
│   ├── shap_high_risk_importance.png                         # High-risk SHAP bar chart
│   ├── shap_waterfall_patient_example.png                    # Patient-level SHAP waterfall explanation
│   ├── eval_roc_curves_best_model.png                        # One-vs-Rest ROC curves
│   └── confusion_matrices_all_models.png                     # Multi-class confusion matrices
└── Notebook/
    └── SmartCare_Hospital.ipynb                              # Fully executed coursework Jupyter Notebook
```

---

## 📚 References (IEEE Format)

[1] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS 2017)*, vol. 30, pp. 4765–4774, Dec. 2017.

[2] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD '16)*, San Francisco, CA, USA, Aug. 2016, pp. 785–794. doi: 10.1145/2939672.2939785.

[3] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research (JMLR)*, vol. 12, pp. 2825–2830, Nov. 2011.

[4] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, Oct. 2001. doi: 10.1023/A:1010933404324.

[5] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, Sep. 1995. doi: 10.1007/BF00994018.

[6] J. H. Ward, "Hierarchical grouping to optimize an objective function," *Journal of the American Statistical Association*, vol. 58, no. 301, pp. 236–244, Mar. 1963. doi: 10.1080/01621459.1963.10500845.

---

## 🤝 Contribution

| Task | Name |
|------|------|
| Task 01–04, 09 | Ravindi Ayodya |
| Task 05–06 | Malith Shehan |
| Task 07 | Thimeth Chathnuka |
| Task 08 | Sithumi Jayarathna |
