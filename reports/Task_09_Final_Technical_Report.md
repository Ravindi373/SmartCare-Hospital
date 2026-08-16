# 🏥 CCS3440 Artificial Intelligence — Coursework Project Technical Report

> **Sri Lanka Technological Campus (SLTC)**  
> **Module Code:** CCS3440 — Artificial Intelligence  
> **Option C:** Multi-Class Disease Risk Classification (SmartCare Hospital)  
> **Group:** Group 02  
> **Date:** August 2026  
> **Document Status:** Final Submission (All Review Corrections Applied)

---

## Executive Summary

This report documents the design, rigorous methodology, implementation, evaluation, and explainability of an end-to-end Machine Learning pipeline for **Multi-Class Disease Risk Stratification** (`Low`, `Medium`, `High`) at SmartCare Hospital. 

Addressing all academic review feedback, this revised implementation enforces **strict train/test data leakage isolation** by executing the train/test split prior to feature encoding, selection, and scaling. Nominal categorical variables are encoded using **One-Hot Encoding** (`OneHotEncoder`) to prevent spurious ordinal relationships. Class labels are explicitly mapped (`Low = 0`, `Medium = 1`, `High = 2`), aligning class supports on the held-out test set (`Low = 26`, `Medium = 94`, `High = 80`).

Evaluating five tuned classical models (Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, and XGBoost) alongside a Neural Network architecture and an ablation study on class weighting, **Logistic Regression and SVM** achieved top held-out test performance (**Accuracy = 99.0%**, **Macro-F1 = 0.9841**, **ROC-AUC = 0.9999**). Model interpretability was established using true **SHAP (SHapley Additive exPlanations)** values derived from the best saved pipeline. A 5-feature prototype model was evaluated on held-out test data (**Accuracy = 90.0%**, **Macro-F1 = 0.8910**) and deployed as a Streamlit web application model artifact.

---

## 1. Problem Definition & Clinical Context

Early identification of patient disease risk enables hospital systems to allocate resources efficiently, optimize outpatient triage, and implement targeted preventive healthcare interventions. In hospital operations, misclassifying a high-risk patient as low-risk can lead to delayed critical treatment, while misclassifying a low-risk patient as high-risk leads to unnecessary bed occupancy and inflated billing costs.

### 1.1 Machine Learning Formulation
The problem is formulated as a supervised **Multi-Class Classification** task:

$$\hat{y} = f(\mathbf{x}) \in \{\text{Low (0)}, \text{Medium (1)}, \text{High (2)}\}$$

where $\mathbf{x}$ represents a patient vector containing physiological vitals, clinical diagnoses, operational history, and financial metrics.

### 1.2 Evaluation Strategy & Primary Metric
Because clinical risk stratification requires balanced sensitivity across all risk categories (especially avoiding false negatives for High and Low risk patients), **Macro-Averaged F1 Score ($\text{F1}_{\text{macro}}$)** is designated as the primary optimization metric, supported by Accuracy, Precision (macro), Recall (macro), and One-vs-Rest ROC-AUC.

---

## 2. Dataset Understanding & Exploratory Analysis

The dataset consists of $N = 1,000$ patient records collected from SmartCare Hospital, spanning 33 raw attributes across clinical, demographic, operational, and financial domain categories.

### 2.1 Target Variable Distribution
The raw target column `disease_risk_level` exhibits the following class distribution:

| Class Name | Target ID | Frequency | Proportion (%) | Expected Test Support (20%) |
|------------|-----------|-----------|----------------|-----------------------------|
| **Low Risk** | 0 | 131 | 13.1% | 26 |
| **Medium Risk** | 1 | 469 | 46.9% | 94 |
| **High Risk** | 2 | 400 | 40.0% | 80 |
| **Total** | — | **1,000** | **100.0%** | **200** |

*Correction Note:* In previous draft iterations, string label encoding sorted classes alphabetically (`High=0, Low=1, Medium=2`), leading to inverted support displays (`Low=80, Medium=26, High=94`). In this final version, target encoding is explicitly fixed as `Low=0, Medium=1, High=2`, matching expected clinical distribution.

---

## 3. Data Preprocessing & Leakage Elimination

To ensure academic and clinical validity, all data preprocessing operations were refactored to eliminate cross-sample data leakage.

```
Raw Dataset (N=1000)
       │
       ▼
Conditional Imputation & Duplicate Removal (Section 3.2)
       │
       ▼
[STRATIFIED TRAIN / TEST SPLIT (80/20)]
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
Training Set (N=800)                     Test Set (N=200)
       │                                         │
       ├─► Fit OneHotEncoder                    ├─► Transform via fitted OHE
       ├─► Fit SelectKBest (K=15)               ├─► Transform via fitted Selector
       └─► Fit StandardScaler                   └─► Transform via fitted Scaler
```

### 3.1 Data Cleaning & Imputation (Section 3.1 & 3.2)
- **Room Type Imputation**: Patients with `admitted == 0` are assigned `"Not Admitted"`. Patients with `admitted == 1` but missing `room_type` are imputed using the mode of admitted patients (`"General Ward"`).
- **Duplicate Detection**: Exact duplicate patient rows were verified and removed prior to splitting.
- **Identifier Removal**: Identifiers and potential target surrogates (`record_id`, `patient_id`, `appointment_date`, `no_show`, `readmitted_30_days`) were removed.

### 3.2 Stratified Train/Test Split (Section 3.3)
The dataset was split into **80% Training ($N=800$)** and **20% Held-Out Testing ($N=200$)** immediately after duplicate detection, stratified by `disease_risk_level`.

### 3.3 Categorical Encoding via OneHotEncoder (Section 3.7)
Label encoding was replaced with `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for nominal variables (`gender`, `blood_group`, `department`, `diagnosis`, `payment_method`, `room_type`, `payment_status`). This prevents introducing artificial numeric ordering into nominal categories.

### 3.4 Feature Selection & Scaling (Section 3.8 & 3.9)
- **Feature Selection**: `SelectKBest(score_func=f_classif, k=15)` was fitted **strictly on $X_{\text{train}}$**.
- **Standard Scaling**: `StandardScaler` was fitted **strictly on selected $X_{\text{train}}$** and applied to $X_{\text{test}}$ via `.transform()`.

---

## 4. Feature Engineering & Clinical Indicators

Six domain-specific clinical and operational features were engineered row-wise without data leakage:

1. **`bp_category`**: Clinical blood pressure staging based on systolic and diastolic BP (Normal, Elevated, Stage 1 Hypertension).
2. **`age_group`**: WHO clinical age stratification (Under 18, 18-35, 36-50, 51-65, 65+).
3. **`bmi_category`**: Standard BMI categorization (Underweight, Normal, Overweight, Obese).
4. **`missed_appointment_rate`**: Ratio of missed previous appointments to total previous appointments.
5. **`is_chronic_patient`**: Binary indicator ($\ge 2$ previous admissions).
6. **`care_intensity`**: Sum of lab tests count and treatment procedures count.

---

## 5. Machine Learning Model Development & Tuning

Five classical model families were trained using **Stratified 5-Fold Cross-Validation** on $X_{\text{train}}$ ($N=800$) with `f1_macro` optimization via `GridSearchCV`.

### 5.1 Hyperparameter Search Space & Best Parameters

| Model | Hyperparameter Grid | Selected Optimal Parameters |
|-------|---------------------|-----------------------------|
| **Logistic Regression** | `C`: [0.01, 0.1, 1.0, 10.0], `class_weight`: ['balanced'] | `C=10.0`, `solver='lbfgs'`, `class_weight='balanced'` |
| **Decision Tree** | `max_depth`: [3, 5, 7, 10, None], `criterion`: ['gini', 'entropy'] | `max_depth=None`, `criterion='entropy'`, `min_samples_leaf=1` |
| **Random Forest** | `n_estimators`: [100, 200], `min_samples_leaf`: [1, 3, 5] | `n_estimators=200`, `max_depth=None`, `min_samples_leaf=3` |
| **SVM** | `C`: [0.1, 1.0, 10.0], `kernel`: ['rbf', 'linear'] | `C=10.0`, `kernel='linear'`, `class_weight='balanced'` |
| **XGBoost** | `n_estimators`: [100, 200], `learning_rate`: [0.05, 0.1, 0.2] | `n_estimators=200`, `learning_rate=0.2`, `max_depth=3` |

---

## 6. Multi-Class Model Evaluation & Benchmarking

All trained models were evaluated on the **held-out test set ($N=200$)**.

### 6.1 Benchmark Performance Table

| Rank | Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (OvR macro) |
|------|-------|----------|-------------------|----------------|------------|---------------------|
| 🥇 1 | **Logistic Regression** | **0.9900** | **0.9762** | **0.9929** | **0.9841** | **0.9999** |
| 🥈 2 | **SVM (Linear Kernel)** | **0.9900** | **0.9762** | **0.9929** | **0.9841** | **0.9998** |
| 🥉 3 | **XGBoost** | 0.8750 | 0.8858 | 0.8223 | 0.8467 | 0.9660 |
| 4 | **Random Forest** | 0.8200 | 0.8407 | 0.8049 | 0.8190 | 0.9500 |
| 5 | **Decision Tree** | 0.7150 | 0.7060 | 0.6995 | 0.7017 | 0.7686 |

### 6.2 Verified Per-Class Performance Table (Best Model — Logistic Regression)

| Risk Class | Target ID | Test Support | Precision | Recall | F1 Score |
|------------|-----------|--------------|-----------|--------|----------|
| **Low Risk** | 0 | 26 | 0.9286 | 1.0000 | 0.9630 |
| **Medium Risk** | 1 | 94 | 1.0000 | 0.9787 | 0.9892 |
| **High Risk** | 2 | 80 | 1.0000 | 1.0000 | 1.0000 |
| **Macro Average** | — | 200 | 0.9762 | 0.9929 | 0.9841 |

### 6.3 Neural Network (Deep Learning) Extension
A Multi-Layer Perceptron (MLP) Neural Network (Dense 64 -> Dropout 0.3 -> Dense 32 -> Softmax 3) was trained using TensorFlow/Keras on $X_{\text{train}}$. Using consistent macro-averaging, the Neural Network achieved **Accuracy = 0.9750**, **Precision (macro) = 0.9520**, **Recall (macro) = 0.9780**, and **F1 (macro) = 0.9645**, confirming strong generalization alongside linear baseline models.

---

## 7. Class-Weighting Ablation Study

To rigorously test the claim that class weighting addressed dataset imbalance (`Low Risk` = 13.1%), an ablation study was performed by evaluating unweighted baseline models against class-weighted models on the held-out test set:

| Model Family | Unweighted Acc | Unweighted Macro-F1 | Unweighted Low-F1 | Weighted Acc | Weighted Macro-F1 | Weighted Low-F1 |
|--------------|----------------|---------------------|-------------------|--------------|-------------------|-----------------|
| **Logistic Regression** | 0.9850 | 0.9751 | 0.9412 | **0.9900** | **0.9841** | **0.9630** |
| **SVM** | 0.9000 | 0.8751 | 0.7917 | **0.9900** | **0.9841** | **0.9630** |
| **Random Forest** | 0.8350 | 0.8052 | 0.7143 | **0.8200** | **0.8190** | **0.8163** |
| **Decision Tree** | 0.7700 | 0.7757 | 0.7925 | 0.7150 | 0.7017 | 0.6538 |
| **XGBoost** | 0.8750 | 0.8546 | 0.7907 | 0.8750 | 0.8467 | 0.7556 |

*Ablation Analysis:* Class weighting significantly improved minority class (`Low Risk`) recall and F1-score for linear models (Logistic Regression: 0.9412 -> 0.9630; SVM: 0.7917 -> 0.9630), confirming its necessity for balanced clinical safety.

---

## 8. Target Leakage & Synthetic Data Analysis

The top-performing models achieved near-perfect accuracy (99.0%) and ROC-AUC (0.9999). As highlighted in academic review:
1. **Deterministic Target Generation**: The synthetic dataset label (`disease_risk_level`) was synthetically generated using deterministic physiological boundaries (e.g., Blood Sugar $> 200 \text{ mg/dL}$, Cholesterol $> 250 \text{ mg/dL}$, and Age $> 65$ mapped strictly to High Risk).
2. **Linear Separability**: Because the underlying data generator uses hyperplanes based on blood sugar, cholesterol, age, and BMI, linear classification models (Logistic Regression and Linear SVM) perfectly separate the feature space.
3. **Clinical Finding**: While 99.0% accuracy validates that the linear models successfully learned the synthetic generation function, real-world clinical datasets contain stochastic noise, laboratory error, and unobserved confounders. Future deployment requires validation on real clinical cohorts.

---

## 9. Explainable AI (SHAP Analysis)

Model interpretability was implemented using **SHAP (SHapley Additive exPlanations)** on the saved best-performing pipeline model (`Logistic Regression`) using held-out test data $X_{\text{test}}$.

```
Rank  Clinical Feature                      Mean |SHAP| Value
 1    blood_sugar_mg_dl                    4.2849  ████████████████████
 2    cholesterol_mg_dl                    4.0013  ██████████████████
 3    age                                  3.6564  ███████████████
 4    bmi                                  3.4693  ██████████████
 5    previous_admissions                  2.3403  █████████
 6    systolic_bp                          1.8857  ███████
```

### 9.1 Global Feature Drivers
- **`blood_sugar_mg_dl`** (Mean $|SHAP| = 4.28$) and **`cholesterol_mg_dl`** (Mean $|SHAP| = 4.00$) were identified as the primary global drivers of disease risk classification, followed by **`age`** ($3.66$), **`bmi`** ($3.47$), and **`previous_admissions`** ($2.34$).
- Visualizations (`shap_summary_multiclass.png`, `shap_high_risk_importance.png`, `shap_waterfall_patient_example.png`) were exported to `reports/` to provide global and patient-level explanations.

---

## 10. AI Prototype & Generalization Evaluation

To satisfy Task 08 requirements, a 5-feature prototype model (`age`, `blood_sugar_mg_dl`, `cholesterol_mg_dl`, `bmi`, `previous_admissions`) was trained on $X_{\text{train}}$ and evaluated on held-out $X_{\text{test}}$ prior to deployment.

### 10.1 Prototype vs Full Pipeline Performance

| Model System | Feature Count | Held-Out Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|--------------|---------------|-------------------|-------------------|----------------|------------|
| **Full 15-Feature Pipeline** | 15 | **0.9900** | **0.9762** | **0.9929** | **0.9841** |
| **5-Feature Prototype Model** | 5 | **0.9000** | **0.8783** | **0.9080** | **0.8910** |

*Deployment Artifact:* The primary model artifact and OneHotEncoder transformation pipeline were bundled into `models/pipeline_bundle.joblib` and integrated into an interactive **Streamlit prototype application** (`app/app.py`).

---

## 11. Conclusion & Recommendations

### 11.1 Conclusion
This project successfully developed, evaluated, and explained a machine learning pipeline for multi-class disease risk stratification. By fixing data leakage, utilizing `OneHotEncoder`, rectifying target class mappings, conducting class-weighting ablation, and applying SHAP explainability, the pipeline achieves robust performance while maintaining complete academic integrity.

### 11.2 Key Recommendations
1. **Deploy Full 15-Feature Pipeline**: The 15-feature pipeline outperforms the 5-feature prototype (F1-macro: 0.9841 vs 0.8910) and should be preferred in clinical settings.
2. **Clinical Cohort Validation**: Before real-world deployment, validate the pipeline on real hospital EHR data containing clinical noise.

---

## 12. References

1. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765–4774. https://doi.org/10.48550/arXiv.1705.07874
2. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785
4. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357. https://doi.org/10.1613/jair.953
5. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
6. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297. https://doi.org/10.1007/BF00994018
