"""
Task 06 – Model Evaluation
Auto-generated from Notebook/SmartCare_Hospital.ipynb (source of truth).
Regenerate this file if the notebook changes, so src/ and the notebook stay in sync.
"""

# # Task 06 – Model Evaluation

try:
    import xgboost
except ImportError:
    import sys
    # [skipped Jupyter shell/magic command] !{sys.executable} -m pip install xgboost -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              precision_recall_fscore_support, roc_auc_score, roc_curve, auc,
                              confusion_matrix, ConfusionMatrixDisplay, classification_report)
from xgboost import XGBClassifier

pd.set_option('display.max_columns', 50)
sns.set_style('whitegrid')
RANDOM_STATE = 42
CLASS_NAMES = ['Low', 'Medium', 'High']  # 0=Low, 1=Medium, 2=High

# Define folder path (assuming drive is already mounted)
folder = '/content/drive/MyDrive/SmartCare/'

# Load X_train, y_train, X_test, y_test from the correct paths
X_train = pd.read_csv(folder + 'X_train.csv')
y_train = pd.read_csv(folder + 'y_train.csv').squeeze()
X_test = pd.read_csv(folder + 'X_test.csv')
y_test = pd.read_csv(folder + 'y_test.csv').squeeze()

print("Train:", X_train.shape, " Test:", X_test.shape)

import joblib

model_files = {
    'Logistic Regression': 'best_logistic_regression.pkl',
    'Decision Tree': 'best_decision_tree.pkl',
    'Random Forest': 'best_random_forest.pkl',
    'SVM': 'best_svm.pkl',
    'XGBoost': 'best_xgboost.pkl',
}

fitted_models = {}
for name, fname in model_files.items():
    fitted_models[name] = joblib.load(folder + fname)

print("Loaded 5 fitted models from Task 05 (identical objects, not re-fit):")
print(list(fitted_models.keys()))

# ## 6.1 Evaluation Results

# ### Multi-Class Metrics — Accuracy, Precision, Recall, F1 (per-class and macro)

per_class_rows = []
summary_rows = []
predictions = {}
probabilities = {}

for name, model in fitted_models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    predictions[name] = y_pred
    probabilities[name] = y_proba

    # Per-class precision/recall/F1
    prec, rec, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2])
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        per_class_rows.append({
            'Model': name, 'Class': cls_name,
            'Precision': prec[cls_idx], 'Recall': rec[cls_idx], 'F1': f1[cls_idx], 'Support': support[cls_idx]
        })

    # Multi-class ROC-AUC: One-vs-Rest, macro-averaged
    roc_auc_macro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    summary_rows.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision (macro)': precision_score(y_test, y_pred, average='macro'),
        'Recall (macro)': recall_score(y_test, y_pred, average='macro'),
        'F1 (macro)': f1_score(y_test, y_pred, average='macro'),
        'ROC-AUC (OvR macro)': roc_auc_macro,
    })

per_class_df = pd.DataFrame(per_class_rows)
summary_df = pd.DataFrame(summary_rows).sort_values('F1 (macro)', ascending=False).reset_index(drop=True)
summary_df.round(4)

per_class_pivot = per_class_df.pivot(index='Model', columns='Class', values=['Precision', 'Recall', 'F1'])
per_class_pivot = per_class_pivot.reindex(summary_df['Model'])
per_class_pivot.round(3)

# ### Confusion Matrices (Multi-Class Metric)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, name in enumerate(summary_df['Model']):
    cm = confusion_matrix(y_test, predictions[name])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=axes[i], colorbar=False, cmap='Blues')
    axes[i].set_title(f"{name}  (Acc={summary_df.loc[summary_df['Model']==name,'Accuracy'].values[0]:.2f})")
axes[-1].axis('off')
plt.suptitle('Confusion Matrices — All 5 Models, Ranked by Macro-F1', fontsize=14)
plt.tight_layout()
plt.savefig('eval_confusion_matrices.png', dpi=110)
plt.show()

# ### ROC-AUC (One-vs-Rest, Multi-Class Metric)

best_model_name = summary_df.iloc[0]['Model']
best_model = fitted_models[best_model_name]
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = probabilities[best_model_name]

plt.figure(figsize=(7, 6))
colors = ['#4C956C', '#F2A541', '#D64550']
for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc_cls = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{cls_name} (AUC = {roc_auc_cls:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'One-vs-Rest ROC Curves — {best_model_name} (Best Model)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('eval_roc_curves_best_model.png', dpi=110)
plt.show()

# ## 6.2 Model Comparison Table

comparison_table = summary_df.copy()
comparison_table.insert(0, 'Rank', range(1, len(comparison_table) + 1))
comparison_table = comparison_table.set_index('Rank')
comparison_table.round(4)

fig, ax = plt.subplots(figsize=(11, 5.5))
metrics_to_plot = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)', 'ROC-AUC (OvR macro)']
summary_df.set_index('Model')[metrics_to_plot].plot(kind='bar', ax=ax, colormap='viridis')
ax.set_ylabel('Score')
ax.set_ylim(0, 1)
ax.set_title('Model Comparison Table — All Metrics, All Models (Test Set)')
plt.xticks(rotation=20)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('eval_model_comparison_table.png', dpi=110)
plt.show()

# ## 6.3 Best Model Identification and Justification

print(f"Best model by every macro-averaged metric: {best_model_name}\n")
print("Full classification report:")
print(classification_report(y_test, predictions[best_model_name], target_names=CLASS_NAMES))

summary_df.to_csv('task06_model_comparison_table.csv', index=False)
per_class_pivot.to_csv('task06_per_class_metrics.csv')
print("Saved: task06_model_comparison_table.csv, task06_per_class_metrics.csv")

# ## 6.4 Investigating Possible Target Leakage / Deterministic Label Construction
#

diag_df = pd.read_csv(folder + 'smartcare_cleaned.csv')

dominant_features = ['age', 'bmi', 'blood_sugar_mg_dl', 'cholesterol_mg_dl', 'systolic_bp']
X_diag = diag_df[dominant_features]
y_diag_str = diag_df['disease_risk_level']

# Match the same split used everywhere else in the notebook for comparability
from sklearn.model_selection import train_test_split
X_diag_train, X_diag_test, y_diag_train, y_diag_test = train_test_split(
    X_diag, y_diag_str, test_size=0.2, random_state=42, stratify=y_diag_str
)
print("Diagnostic train/test sizes:", X_diag_train.shape, X_diag_test.shape)

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

for depth in [2, 3, 4]:
    stump = DecisionTreeClassifier(max_depth=depth, random_state=42)
    stump.fit(X_diag_train, y_diag_train)
    train_acc = accuracy_score(y_diag_train, stump.predict(X_diag_train))
    test_acc = accuracy_score(y_diag_test, stump.predict(X_diag_test))
    print(f"max_depth={depth}: train acc={train_acc:.3f}, test acc={test_acc:.3f}")

# Print the depth-3 tree's actual rules — clean, round-number thresholds
# (e.g. age <= 60.5, blood_sugar_mg_dl <= 140.5) are a classic fingerprint
# of a synthetic rule-based generator rather than fitted noise.
print("\nDepth-3 tree rules:")
print(export_text(DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_diag_train, y_diag_train),
                   feature_names=dominant_features))

# --- Diagnostic 2: Linear R^2 test --------------------------------------
# Regress the ORDINAL-encoded target (Low=0, Medium=1, High=2) on the 5
# standardized clinical features. A very high R^2 suggests the label is
# close to "compute a weighted score, then bin it into 3 classes."
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

ordinal_map = {'Low': 0, 'Medium': 1, 'High': 2}
y_diag_train_ord = y_diag_train.map(ordinal_map)
y_diag_test_ord = y_diag_test.map(ordinal_map)

diag_scaler = StandardScaler()
X_diag_train_scaled = diag_scaler.fit_transform(X_diag_train)
X_diag_test_scaled = diag_scaler.transform(X_diag_test)

lin_reg = LinearRegression().fit(X_diag_train_scaled, y_diag_train_ord)
r2_train = lin_reg.score(X_diag_train_scaled, y_diag_train_ord)
r2_test = lin_reg.score(X_diag_test_scaled, y_diag_test_ord)
print(f"Linear R^2 (train): {r2_train:.3f}")
print(f"Linear R^2 (test):  {r2_test:.3f}")
print("\nImplied weights (standardized units):")
for feat, coef in zip(dominant_features, lin_reg.coef_):
    print(f"  {feat:20s} {coef:+.3f}")

# --- Diagnostic 3: Confidence-distribution test -------------------------
# Uses the actual tuned best model + real test set from Task 06 (fitted_models,
# X_test, y_test — must have already run that section in this session).
# A histogram dominated by near-0/near-1 max-probabilities, with almost
# nothing in the ambiguous 0.4-0.6 band, is consistent with a deterministic
# or near-deterministic label rather than genuinely noisy clinical risk.
best_lr = fitted_models['Logistic Regression']
max_proba = fitted_models['Logistic Regression'].predict_proba(X_test).max(axis=1)

plt.figure(figsize=(7, 4))
plt.hist(max_proba, bins=30, color='#4C72B0', edgecolor='white')
plt.axvline(0.6, color='red', linestyle='--', label='Ambiguous cutoff (0.6)')
plt.xlabel('Predicted probability of the assigned class')
plt.ylabel('Number of test patients')
plt.title('Confidence Distribution — Logistic Regression on Test Set')
plt.legend()
plt.tight_layout()
plt.show()

n_ambiguous = (max_proba < 0.6).sum()
print(f"Test patients with max class probability < 0.6: {n_ambiguous} / {len(max_proba)} "
      f"({n_ambiguous/len(max_proba)*100:.1f}%)")

# --- Diagnostic 4: Inspect the misclassified cases ----------------------
y_pred_lr = fitted_models['Logistic Regression'].predict(X_test)
misclassified_idx = np.where(y_pred_lr != y_test.values)[0]
print(f"Number of misclassified test cases: {len(misclassified_idx)}")

# Map back to the original (unscaled, unselected) feature values for those
# specific rows so the values are clinically interpretable, using the same
# train/test split applied to diag_df.
misclassified_original_index = X_test.iloc[misclassified_idx].index
diag_df.loc[X_diag_test.index[misclassified_idx], dominant_features + ['disease_risk_level']]

# ## 6.5 Class-Weighting Ablation

# Build unweighted counterparts of every tuned model — identical
# hyperparameters, class_weight removed (or sample_weight omitted for XGBoost)
from sklearn.base import clone

unweighted_models = {}
for name, model in fitted_models.items():
    m = clone(model)
    if 'class_weight' in m.get_params():
        m.set_params(class_weight=None)
    unweighted_models[name] = m

for name, model in unweighted_models.items():
    model.fit(X_train, y_train)   # note: no sample_weight passed here, unlike the weighted XGBoost fit
    unweighted_models[name] = model

print("Fitted 5 unweighted baseline models (same hyperparameters, no class-weighting):")
print(list(unweighted_models.keys()))

# Per-class comparison: weighted vs unweighted, for every model and class
from sklearn.metrics import precision_recall_fscore_support

ablation_rows = []
for name in fitted_models:
    for weighting_label, model_dict in [('Weighted', fitted_models), ('Unweighted', unweighted_models)]:
        y_pred = model_dict[name].predict(X_test)
        prec, rec, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2])
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            ablation_rows.append({
                'Model': name, 'Weighting': weighting_label, 'Class': cls_name,
                'Precision': prec[cls_idx], 'Recall': rec[cls_idx],
                'F1': f1[cls_idx], 'Support': support[cls_idx]
            })

ablation_df = pd.DataFrame(ablation_rows)

# Full pivot: every model x class, weighted vs unweighted, side by side
ablation_pivot = ablation_df.pivot_table(
    index=['Model', 'Class'], columns='Weighting', values=['Precision', 'Recall', 'F1']
).round(3)
ablation_pivot

# Focus specifically on the minority 'Low' class, since that's what
# Section 8.4's claim is actually about
low_comparison = (
    ablation_df[ablation_df['Class'] == 'Low']
    .pivot(index='Model', columns='Weighting', values=['Precision', 'Recall', 'F1'])
    .round(3)
)
print("Minority ('Low') class performance — weighted vs unweighted:")
low_comparison

# Visualise: does removing class weighting cost recall on the minority class?
low_recall = ablation_df[ablation_df['Class'] == 'Low'].pivot(
    index='Model', columns='Weighting', values='Recall'
)

ax = low_recall.plot(kind='bar', figsize=(9, 5), color=['#4C956C', '#D64550'])
ax.set_ylabel('Recall on Low-risk class (test set)')
ax.set_ylim(0, 1.05)
ax.set_title("Minority-Class ('Low') Recall — Weighted vs Unweighted")
ax.legend(title='')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('class_weighting_ablation_low_recall.png', dpi=110)
plt.show()

# Explicit gap, model by model
gap = (low_recall['Weighted'] - low_recall['Unweighted']).round(3)
print("Recall gap (Weighted − Unweighted) on the Low class, by model:")
print(gap)
