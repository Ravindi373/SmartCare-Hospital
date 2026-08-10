# ======================================================================
# # Task 05 – Machine Learning Model Development
# 
# This task trains and compares **five** classification models (exceeding the minimum
# requirement of four) on the prepared train/test data from Task 03
# (`smartcare_train_prepared.csv`, `smartcare_test_prepared.csv`), and delivers:
# 
# 1. **Model Training** — five models spanning linear, tree, ensemble, kernel-based, and
#    gradient-boosting approaches, chosen to reflect the algorithm shortlist identified in the
#    Task 01 literature review.
# 2. **Hyperparameter Selection** — grid search with stratified cross-validation for every model,
#    using macro-F1 as the tuning metric because of the class imbalance established in Task 04.
# 3. **Comparative Analysis** — a side-by-side comparison of all five tuned models on the held-out
#    test set, with confusion matrices and a discussion of the best-performing model.
# ======================================================================

# XGBoost is not part of the default Colab image in all environments — install if missing
try:
    import xgboost
except ImportError:
    import sys
# [Jupyter shell]     !{sys.executable} -m pip install xgboost -q
    import xgboost

print("xgboost version:", xgboost.__version__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              classification_report, confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier

pd.set_option('display.max_columns', 50)
sns.set_style('whitegrid')
RANDOM_STATE = 42
CLASS_NAMES = ['Low', 'Medium', 'High']   # target encoding: 0=Low, 1=Medium, 2=High (from Task 03)

train_df = pd.read_csv('smartcare_train_prepared.csv')
test_df  = pd.read_csv('smartcare_test_prepared.csv')

X_train, y_train = train_df.drop(columns=['disease_risk_level']), train_df['disease_risk_level']
X_test,  y_test  = test_df.drop(columns=['disease_risk_level']),  test_df['disease_risk_level']

print("Train:", X_train.shape, " Test:", X_test.shape)
print("\nTrain class distribution:\n", y_train.value_counts(normalize=True).sort_index().round(3))
print("\nTest class distribution:\n",  y_test.value_counts(normalize=True).sort_index().round(3))

# ======================================================================
# **Note on class imbalance:** as established in Task 04, the target is imbalanced (Medium ≈
# 47%, High ≈ 40%, Low ≈ 13%). Every model below is trained with class-balancing enabled
# (`class_weight='balanced'` where the estimator supports it directly, or balanced `sample_weight`
# for XGBoost, which has no built-in `class_weight` parameter for multi-class objectives), so that
# the minority Low-risk class is not systematically ignored. All hyperparameter tuning below also
# uses **macro-F1** rather than accuracy as the scoring metric, for the same reason.
# ======================================================================

# ======================================================================
# ## 5.1 Model Training & 5.2 Hyperparameter Selection
# ======================================================================

# ======================================================================
# Five models are trained, chosen to cover the main algorithm families identified as
# effective for structured healthcare risk prediction in the Task 01 literature review
# (Mavrogiorgou et al., 2022; Suma et al., 2024; Trigka, Dritsas & Mylonas, 2023):
# 
# | Model | Family | Why included |
# |---|---|---|
# | Logistic Regression | Linear | Fast, interpretable baseline; strong performer for multi-class risk in Trigka et al. (2023) |
# | Decision Tree | Single tree | Interpretable, non-linear; forms the basis for Random Forest below |
# | Random Forest | Bagging ensemble | Best performer for 3-class risk-level prediction in Suma et al. (2024) |
# | Support Vector Machine (RBF) | Kernel-based | Effective for smaller, well-scaled tabular datasets like this one |
# | XGBoost | Gradient-boosting ensemble | Represents the ensemble/boosting family shown to dominate structured healthcare prediction in Banerjee & Paçal (2025) |
# 
# For each model we run a `GridSearchCV` with 5-fold **stratified** cross-validation (stratification
# preserves the same class proportions in every fold, which matters given the imbalance) and
# `scoring='f1_macro'`, then evaluate the best estimator found.
# ======================================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}       # will hold fitted best estimators
cv_summary = []     # will hold tuning summary rows

def tune_and_report(name, estimator, param_grid, X, y, fit_params=None):
    start = time.time()
    grid = GridSearchCV(estimator, param_grid, scoring='f1_macro', cv=cv, n_jobs=-1, refit=True)
    grid.fit(X, y, **(fit_params or {}))
    elapsed = time.time() - start
    best_idx = grid.best_index_
    best_std = grid.cv_results_['std_test_score'][best_idx]
    print(f"{name:22s} | best CV macro-F1 = {grid.best_score_:.4f} ± {best_std:.4f} | best params = {grid.best_params_} | {elapsed:.1f}s")
    results[name] = grid.best_estimator_
    cv_summary.append({
        'Model': name,
        'Best CV Macro-F1': grid.best_score_,
        'CV Std': best_std,
        'Best Params': grid.best_params_
    })
    return grid.best_estimator_

# ======================================================================
# ### Model 1 — Logistic Regression
# ======================================================================

lr_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'max_iter': [2000],
    'class_weight': ['balanced'],
}
best_lr = tune_and_report('Logistic Regression', LogisticRegression(random_state=RANDOM_STATE), lr_grid, X_train, y_train)

# ======================================================================
# **Justification of grid:** `C` (inverse regularisation strength) is the main hyperparameter
# for Logistic Regression; a wide log-spaced range (0.01–10) is searched to balance underfitting
# (small C) against overfitting (large C) on a moderately sized (800-row) training set.
# `class_weight='balanced'` is fixed rather than searched, since Task 04 established the imbalance
# is a structural property of the data that should always be corrected for, not something to
# tune away.
# ======================================================================

# ======================================================================
# ### Model 2 — Decision Tree
# ======================================================================

dt_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_leaf': [1, 5, 10],
    'min_samples_split': [2, 10],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced'],
}
best_dt = tune_and_report('Decision Tree', DecisionTreeClassifier(random_state=RANDOM_STATE), dt_grid, X_train, y_train)

# ======================================================================
# **Justification of grid:** `max_depth` and `min_samples_leaf` jointly control overfitting —
# a single unconstrained tree can memorise the 800-row training set perfectly (0% training error)
# while generalising poorly, so depths from a shallow 3 up to unrestricted (`None`) are compared
# under cross-validation to find the best bias–variance trade-off. Both split criteria are
# included since neither is guaranteed to dominate on a given dataset.
# ======================================================================

# ======================================================================
# ### Model 3 — Random Forest
# ======================================================================

rf_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 3, 5],
    'class_weight': ['balanced'],
}
best_rf = tune_and_report('Random Forest', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), rf_grid, X_train, y_train)

# ======================================================================
# **Justification of grid:** `n_estimators` (number of trees) is searched up to 300, beyond
# which returns typically diminish for a dataset this size; `max_depth` and `min_samples_leaf`
# control individual tree complexity in the same way as for the single Decision Tree above. This
# model is prioritised for tuning depth given Suma et al. (2024) found Random Forest to be the best
# of three models for a very similar 3-class risk-level task.
# ======================================================================

# ======================================================================
# ### Model 4 — Support Vector Machine
# ======================================================================

svm_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto'],
    'class_weight': ['balanced'],
    'probability': [True],
}
best_svm = tune_and_report('SVM', SVC(random_state=RANDOM_STATE), svm_grid, X_train, y_train)

# ======================================================================
# **Justification of grid:** both `rbf` (non-linear) and `linear` kernels are compared, since
# it is not known in advance whether the class boundaries are linearly separable in this feature
# space. `C` and `gamma` are the standard SVM regularisation/kernel-width parameters; `probability=True`
# is set so the fitted model can later produce probability estimates if needed for Task 07/08.
# Because all features were already standardised in Task 03, SVM (a distance-based method) can be
# applied fairly without additional scaling here.
# ======================================================================

# ======================================================================
# ### Model 5 — XGBoost
# ======================================================================

xgb_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
best_xgb = tune_and_report(
    'XGBoost',
    XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss',
                  random_state=RANDOM_STATE, n_jobs=-1),
    xgb_grid, X_train, y_train,
    fit_params={'sample_weight': sample_weights}
)

# ======================================================================
# **Justification of grid:** `n_estimators`, `max_depth`, and `learning_rate` are the three
# hyperparameters that most directly trade off underfitting against overfitting in gradient
# boosting; a low learning rate typically needs more estimators to converge, so both are searched
# jointly rather than independently. XGBoost has no native `class_weight` parameter for multi-class
# objectives, so class balance is instead enforced by passing balanced `sample_weight`s computed
# from the training labels — functionally equivalent to `class_weight='balanced'` in the other
# models.
# ======================================================================

cv_summary_df = pd.DataFrame(cv_summary).sort_values('Best CV Macro-F1', ascending=False)
cv_summary_df[['Model', 'Best CV Macro-F1']].reset_index(drop=True)

# ======================================================================
# **Interpretation:** the cross-validation ranking above reflects each model's ability to
# generalise across folds *before* looking at the held-out test set at all, which keeps the test set
# fully unseen until the final comparison in Section 5.3.
# ======================================================================

# ======================================================================
# ## 5.3 Comparative Analysis
# ======================================================================

# ======================================================================
# All five tuned models are now evaluated once on the held-out test set (used for the first
# time here), reporting accuracy, macro-averaged precision/recall/F1 (macro-averaging weights all
# three classes equally, which matters given the imbalance — see Task 04), and a full
# per-class classification report.
# ======================================================================

def evaluate_model(name, model, X_te, y_te):
    y_pred = model.predict(X_te)
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_te, y_pred),
        'Precision (macro)': precision_score(y_te, y_pred, average='macro'),
        'Recall (macro)': recall_score(y_te, y_pred, average='macro'),
        'F1 (macro)': f1_score(y_te, y_pred, average='macro'),
    }, y_pred

test_results = []
predictions = {}
for name, model in results.items():
    row, y_pred = evaluate_model(name, model, X_test, y_test)
    test_results.append(row)
    predictions[name] = y_pred

test_results_df = pd.DataFrame(test_results).sort_values('F1 (macro)', ascending=False).reset_index(drop=True)
test_results_df.round(4)

fig, ax = plt.subplots(figsize=(10, 5.5))
metrics_to_plot = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)']
plot_df = test_results_df.set_index('Model')[metrics_to_plot]
plot_df.plot(kind='bar', ax=ax, colormap='viridis')
ax.set_ylabel('Score')
ax.set_ylim(0, 1)
ax.set_title('Test-Set Performance Comparison Across 5 Models')
ax.legend(loc='lower right')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('model_comparison_bar.png', dpi=110)
plt.show()

best_model_name = test_results_df.iloc[0]['Model']
print(f"Best model by test macro-F1: {best_model_name}\n")
print(classification_report(y_test, predictions[best_model_name], target_names=CLASS_NAMES))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, (name, model) in enumerate(results.items()):
    cm = confusion_matrix(y_test, predictions[name])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=axes[i], colorbar=False, cmap='Blues')
    axes[i].set_title(name)
axes[-1].axis('off')
plt.suptitle('Confusion Matrices — All 5 Tuned Models (Test Set)', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrices_all_models.png', dpi=110)
plt.show()

# ======================================================================
# **Interpretation:** The confusion matrices reveal a pattern consistent across all five
# models: the majority of misclassifications occur between the **Medium** and **High** classes,
# while the **Low** class — despite being the minority class — is identified comparatively well once
# class-balancing is applied during training. This is expected given the Task 04 finding that
# clinical vitals separate Low-risk patients most cleanly (tightest boxplot ranges, clearest
# scatterplot clustering), while Medium and High share more overlapping clinical profiles.
# ======================================================================

print("Final ranked comparison (test set):")
test_results_df.round(4)

# ======================================================================
# ### Discussion — Comparative Analysis
# 
# The actual results (see ranked table above) tell a clear and somewhat unexpected story relative
# to the Task 01 literature, which is itself a useful, honestly-reported finding:
# 
# - **Best models: Logistic Regression and SVM (linear kernel), in a near-tie.** Logistic Regression
#   achieved the top test macro-F1 (≈0.906, accuracy ≈0.915), with SVM essentially matching it
#   (macro-F1 ≈0.906, accuracy ≈0.915) — and critically, the SVM's own hyperparameter search *chose
#   a linear kernel over RBF* (`kernel='linear'` won with CV macro-F1 ≈0.945, its best score of any
#   model). Two independently-tuned linear models converging on the same top performance is strong
#   evidence that **the boundaries between Low/Medium/High risk in this dataset are close to
#   linearly separable** in the engineered feature space, rather than requiring the non-linear
#   decision boundaries that trees, forests, or an RBF kernel are built to capture.
# - **Tree-based ensembles underperformed the linear models here** — Random Forest (macro-F1 ≈0.76)
#   and XGBoost (≈0.81) both trailed Logistic Regression/SVM by 10–15 points of macro-F1, and the
#   single Decision Tree was weakest overall (≈0.70). This is the opposite ranking from what several
#   Task 01 sources reported for their own datasets (e.g., Suma et al., 2024, found Random Forest
#   best for a similar 3-class risk task). The likely explanation is that `disease_risk_level` in
#   this dataset was generated from clinically meaningful, roughly additive thresholds on the
#   clinical vitals (consistent with the smooth, monotonic gradients seen across every boxplot and
#   stacked-bar chart in Task 04) — a pattern linear models are naturally well-suited to fit exactly,
#   while tree ensembles must approximate with many axis-aligned splits and can overfit noise in the
#   process on a training set of only 800 rows. This is a useful example of why the Task 01
#   literature should inform the initial model shortlist but not be assumed to transfer directly —
#   empirical comparison on the actual dataset, as performed here, remains essential.
# - **Per-class performance of the best model (Logistic Regression):** precision and recall are
#   strong across all three classes (Low: precision 0.95 / recall 0.81; Medium: 0.89 / 0.94; High:
#   0.94 / 0.93). The Low-risk class — the minority class at only 13% of the data — has the lowest
#   recall (0.81), meaning roughly 1 in 5 true Low-risk patients is misclassified as a higher risk
#   category; this is the class most worth monitoring further in Task 06, since under-flagging a
#   genuinely low-risk patient as higher risk is a safer clinical error than the reverse, but it
#   still has resource-allocation costs.
# - **Confusion matrix pattern:** across all five models, most remaining errors sit between the
#   Medium and High classes rather than involving Low, consistent with Task 04's finding that
#   Medium and High share more overlapping clinical profiles than either does with Low.
# 
# **Model carried forward:** Logistic Regression is selected as the best-performing model and is
# carried forward into **Task 06** (Model Evaluation) for deeper metric analysis, and into
# **Task 07** (Explainable AI) — where its selection is additionally convenient, since a linear
# model's coefficients are directly and transparently interpretable, aligning well with the
# explainability goals of that task. SVM (linear) is retained as the second-best result and could
# serve as a robustness check if Task 06 finds any evaluation-set-specific sensitivity in the
# Logistic Regression result.
# ======================================================================

import joblib
best_overall_model = results[best_model_name]
joblib.dump(best_overall_model, 'best_model.pkl')
joblib.dump(results, 'all_tuned_models.pkl')
test_results_df.to_csv('task05_model_comparison_results.csv', index=False)
print(f"Saved best model ({best_model_name}) to best_model.pkl")
print("Saved all 5 tuned models to all_tuned_models.pkl")
print("Saved comparison table to task05_model_comparison_results.csv")
