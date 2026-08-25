"""
Task 08 – Deployment-Ready Model Artefact
Auto-generated from Notebook/SmartCare_Hospital.ipynb (source of truth).
Regenerate this file if the notebook changes, so src/ and the notebook stay in sync.
"""

# # Task 08 – Deployment-Ready Model Artefact

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

folder = '/content/drive/MyDrive/SmartCare/'

# Reuse the SAME saved best-performing pipeline outputs — no raw-data reload,
# no separate scaler fit, no separate target mapping.
X_train = pd.read_csv(folder + 'X_train.csv')
X_test = pd.read_csv(folder + 'X_test.csv')
y_train = pd.read_csv(folder + 'y_train.csv').squeeze()
y_test = pd.read_csv(folder + 'y_test.csv').squeeze()
CLASS_NAMES = pd.read_csv(folder + 'target_classes.csv')['class_name'].tolist()

main_scaler = joblib.load(folder + 'feature_scaler_main.pkl')
selected_features = X_train.columns.tolist()  # same order the scaler was fit on

# Single source of truth for "top 5 features" — computed by SHAP in Section
# 7.1, not a separately hardcoded list that could silently diverge from it
top_features = pd.read_csv(folder + 'shap_top5_high_risk_drivers.csv')['Feature'].tolist()
print("Prototype features (from Section 7.1 SHAP results):", top_features)

missing = [f for f in top_features if f not in selected_features]
assert not missing, f"Expected these to already be in the selected 15-feature set: {missing}"

# X_train/X_test are ALREADY scaled (Section 3.10) — slicing columns from an
# already-fit StandardScaler is mathematically identical to fitting a new
# scaler on just these 5 raw columns from the same training rows, so no
# re-fitting is needed or done here.
X_train_proto = X_train[top_features]
X_test_proto = X_test[top_features]

lr_proto = LogisticRegression(random_state=42, max_iter=1000)
lr_proto.fit(X_train_proto, y_train)   # trained on TRAIN split only, unlike the earlier 100%-data version

proto_test_acc = accuracy_score(y_test, lr_proto.predict(X_test_proto))
print(f"5-feature prototype — held-out test accuracy: {proto_test_acc:.3f}")
print(classification_report(y_test, lr_proto.predict(X_test_proto), target_names=CLASS_NAMES))

# Quantify the accuracy trade-off explicitly — previously flagged in report
# limitations as "likely lower... but this drop has not been quantified"
full_model = joblib.load(folder + 'best_logistic_regression.pkl')
full_test_acc = accuracy_score(y_test, full_model.predict(X_test))

print(f"Full 15-feature best model test accuracy: {full_test_acc:.3f}")
print(f"5-feature prototype test accuracy:        {proto_test_acc:.3f}")
print(f"Accuracy trade-off from simplifying to 5 UI-friendly features: {full_test_acc - proto_test_acc:+.3f}")

# Build the UI-facing scaler by SLICING the main scaler's fitted parameters
# to just these 5 features — not fitting a new one. This guarantees the
# prototype's scaling is byte-for-byte consistent with the canonical
# pipeline's statistics, computed only from the training split.
feature_idx = [selected_features.index(f) for f in top_features]

proto_scaler = StandardScaler()
proto_scaler.mean_ = main_scaler.mean_[feature_idx]
proto_scaler.scale_ = main_scaler.scale_[feature_idx]
proto_scaler.var_ = main_scaler.var_[feature_idx]
proto_scaler.n_features_in_ = len(top_features)
proto_scaler.feature_names_in_ = np.array(top_features)

print("proto_scaler built directly from main_scaler's fitted parameters")
print("Feature order:", top_features)
print("Means:", proto_scaler.mean_)
print("Scales:", proto_scaler.scale_)

joblib.dump(lr_proto, folder + 'disease_risk_model.pkl')
joblib.dump(proto_scaler, folder + 'feature_scaler.pkl')
print("Model and scaler saved as deployment-ready artefacts: "
      "disease_risk_model.pkl, feature_scaler.pkl")
print("NOTE: no UI, API, or application code has been built. These files ")
print("are inputs for a future interface, not a working prototype on their own.")
