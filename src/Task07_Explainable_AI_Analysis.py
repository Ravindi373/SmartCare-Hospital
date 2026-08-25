"""
Task 07 - Explainable AI Analysis (SHAP)
Auto-generated from Notebook/SmartCare_Hospital.ipynb (source of truth).
Regenerate this file if the notebook changes, so src/ and the notebook stay in sync.
"""

# # Task 07 - Explainable AI Analysis (SHAP)

# We use SHAP (SHapley Additive exPlanations) on the Random Forest / XGBoost model to interpret which features drive readmission predictions, and how.

# Setup: install shap if needed, then reload the exact fitted models and
# data splits from Task 06 (self-contained restart cell, same pattern used
# throughout this notebook)
try:
    import shap
except ImportError:
    import sys
# [Jupyter shell]     !{sys.executable} -m pip install shap -q
    import shap

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder = '/content/drive/MyDrive/SmartCare/'

CLASS_NAMES = pd.read_csv(folder + 'target_classes.csv')['class_name'].tolist()
print("Verified class order (index -> label):", list(enumerate(CLASS_NAMES)))

X_train = pd.read_csv(folder + 'X_train.csv')
X_test = pd.read_csv(folder + 'X_test.csv')

best_lr = joblib.load(folder + 'best_logistic_regression.pkl')
best_rf = joblib.load(folder + 'best_random_forest.pkl')

print("Loaded fitted models. Feature set:", list(X_train.columns))

# Compatibility helper — different shap versions return either a list of
# per-class arrays or a single 3D array (n_samples, n_features, n_classes)
def get_class_shap(sv, class_idx):
    vals = sv.values if hasattr(sv, 'values') else sv
    return vals[class_idx] if isinstance(vals, list) else vals[:, :, class_idx]

def mean_abs_all_classes(sv):
    vals = sv.values if hasattr(sv, 'values') else sv
    stacked = np.stack(vals, axis=-1) if isinstance(vals, list) else vals
    return np.abs(stacked).mean(axis=(0, 2))

# ## 7.1 SHAP on the Best Model (Logistic Regression)

explainer_lr = shap.LinearExplainer(best_lr, X_train)

X_test_sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
shap_values_lr = explainer_lr(X_test_sample)

print("SHAP values shape:", shap_values_lr.values.shape)  # (n_samples, n_features, n_classes) or similar

# Overall importance: mean |SHAP value| across samples AND classes
mean_abs_shap_lr = mean_abs_all_classes(shap_values_lr)
shap_importance_lr = pd.DataFrame({
    'Feature': X_test_sample.columns,
    'Mean_|SHAP|': mean_abs_shap_lr
}).sort_values('Mean_|SHAP|', ascending=False)

plt.figure(figsize=(10, 6))
top10_lr = shap_importance_lr.head(10)
plt.barh(top10_lr['Feature'][::-1], top10_lr['Mean_|SHAP|'][::-1], color='#4C72B0')
plt.xlabel('Mean |SHAP value| (across all classes)')
plt.title('SHAP Feature Importance — Logistic Regression (Best Model)')
plt.tight_layout()
plt.savefig('shap_lr_overall_importance.png', dpi=110)
plt.show()

shap_importance_lr.head(10)

# SHAP for the 'High' risk class specifically — index resolved from the
# VERIFIED class mapping, not hardcoded (this is exactly the kind of
# hardcoding that caused the Section 8.2 label-swap bug earlier)
high_idx = CLASS_NAMES.index('High')
shap_high_lr = get_class_shap(shap_values_lr, high_idx)

# Beeswarm summary plot for the High-risk class
shap.summary_plot(shap_high_lr, X_test_sample, show=False)
plt.title('SHAP Summary — Drivers of "High" Risk (Logistic Regression)')
plt.tight_layout()
plt.savefig('shap_lr_high_risk_beeswarm.png', dpi=110)
plt.show()

# Signed mean SHAP value (not absolute) — positive = pushes toward High risk
mean_shap_high_lr = shap_high_lr.mean(axis=0)
high_drivers_lr = pd.DataFrame({
    'Feature': X_test_sample.columns,
    'Mean_SHAP_High': mean_shap_high_lr
}).sort_values('Mean_SHAP_High', ascending=False)

print("Top 5 SHAP drivers of 'High' risk (Logistic Regression):")
print(high_drivers_lr.head(5))

# Persist this list as the single source of truth for "top 5 features" —
# Task 08's prototype reads it back instead of hardcoding a separate list
# that could silently drift from what SHAP actually found.
high_drivers_lr.head(5)[['Feature']].to_csv(folder + 'shap_top5_high_risk_drivers.csv', index=False)
print("Saved top-5 driver list to:", folder + 'shap_top5_high_risk_drivers.csv')

# ## 7.2 Supplementary SHAP on a Tree-Based Model (Random Forest)

explainer_rf = shap.TreeExplainer(best_rf)
shap_values_rf = explainer_rf(X_test_sample)

mean_abs_shap_rf = mean_abs_all_classes(shap_values_rf)
shap_importance_rf = pd.DataFrame({
    'Feature': X_test_sample.columns,
    'Mean_|SHAP|': mean_abs_shap_rf
}).sort_values('Mean_|SHAP|', ascending=False)

plt.figure(figsize=(10, 6))
top10_rf = shap_importance_rf.head(10)
plt.barh(top10_rf['Feature'][::-1], top10_rf['Mean_|SHAP|'][::-1], color='#DD8452')
plt.xlabel('Mean |SHAP value| (across all classes)')
plt.title('SHAP Feature Importance — Random Forest (Supplementary)')
plt.tight_layout()
plt.savefig('shap_rf_overall_importance.png', dpi=110)
plt.show()

shap_importance_rf.head(10)

# ## 7.3 Cross-Method Comparison & Interpretation

top10_lr_features = set(shap_importance_lr.head(10)['Feature'])
top10_rf_features = set(shap_importance_rf.head(10)['Feature'])
overlap = top10_lr_features & top10_rf_features

print(f"Top-10 feature overlap between SHAP-LR and SHAP-RF: {len(overlap)}/10")
print("Shared features:", sorted(overlap))
print("\nSHAP-LR only:", sorted(top10_lr_features - top10_rf_features))
print("SHAP-RF only:", sorted(top10_rf_features - top10_lr_features))

comparison_df = pd.DataFrame({
    'SHAP-LR rank': shap_importance_lr.reset_index(drop=True)['Feature'],
    'SHAP-RF rank': shap_importance_rf.reset_index(drop=True)['Feature'],
}).head(10)
comparison_df.index = comparison_df.index + 1
comparison_df

# **Interpretation and limitations:**
#
# - Strong agreement between SHAP-LR and SHAP-RF top features (and with the
#   Section 5.4 ANOVA ranking) is genuine convergent evidence that
#   `age`, `blood_sugar_mg_dl`, `cholesterol_mg_dl`, `bmi`, and `systolic_bp`
#   drive the model's predictions — three independent methods agreeing is
#   meaningfully stronger than any one of them alone.
# - That said, convergence across methods only tells us these features drive
#   *this model's predictions*. It does not by itself confirm the underlying
#   clinical relationship is real rather than an artefact of how
#   `disease_risk_level` was constructed — see Section 6.4's investigation
#   into possible deterministic label construction. If the target turns out
#   to be a formula on these same features, SHAP will (correctly) identify
#   that formula's inputs as "important," which would be recovering the
#   generating rule rather than confirming a clinical mechanism.
# - Both explainers here explain a 200-row sample of the test set for
#   plotting speed, which is standard SHAP practice and does not change the
#   ranking, only the smoothness of the beeswarm plots.
# - SHAP values are computed on the model's actual training feature space
#   (the 15 selected, scaled, encoded features from Section 3), so they are
#   directly comparable to what the deployed model sees — unlike the earlier
#   coefficient-based analysis, which retrained a separate model on a
#   different (wider, differently-encoded) feature set entirely.
