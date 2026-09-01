"""
Task 08 – Deployment-Ready Model Artefact
Auto-generated from Notebook/SmartCare_Hospital.ipynb (source of truth).
Regenerate this file if the notebook changes, so src/ and the notebook stay in sync.
"""

# # Task 08 – Deployment-Ready Model Artefact

import pandas as pd  # Data manipulation library
import numpy as np  # Numerical computing library
import joblib  # Save and load model/scaler objects
from sklearn.linear_model import LogisticRegression  # Logistic regression model
from sklearn.preprocessing import StandardScaler  # Feature standardization scaler
from sklearn.metrics import accuracy_score, classification_report  # Model evaluation metrics

folder = '/content/drive/MyDrive/SmartCare/'  # Base directory path

# load preprossed data
# no separate scaler fit, no separate target mapping.
X_train = pd.read_csv(folder + 'X_train.csv')  # Load training feature set [IND]
X_test = pd.read_csv(folder + 'X_test.csv')  # Load testing feature set [IND]
y_train = pd.read_csv(folder + 'y_train.csv').squeeze()  # Load training labels as a Series [DP]
y_test = pd.read_csv(folder + 'y_test.csv').squeeze()  # Load testing labels as a Series [DP]
CLASS_NAMES = pd.read_csv(folder + 'target_classes.csv')['class_name'].tolist()  # Load target class names list [risl catagory ]

# have to load the data standerliser 
main_scaler = joblib.load(folder + 'feature_scaler_main.pkl')  # Load pre-fitted 15-feature scaler.has mean and deviation of the each feature.to stop dominating the large value parametres
selected_features = X_train.columns.tolist()  # Get full 15 feature column names and write then as a standerd pyhon list

# Single source of truth for "top 5 features" — computed by SHAP in Section
# 7.1, not a separately hardcoded list that could silently diverge from it
top_features = pd.read_csv(folder + 'shap_top5_high_risk_drivers.csv')['Feature'].tolist()  # Load top 5 SHAP feature names
print("Prototype features (from Section 7.1 SHAP results):", top_features)  # Display top 5 features

missing = [f for f in top_features if f not in selected_features]  # Identify any missing features
assert not missing, f"Expected these to already be in the selected 15-feature set: {missing}"  # Validate all top features exist
# this is a safty check.loop throuth every top feature ensure existance prevent from key error or indexing problems

# generate 5 colom table 
X_train_proto = X_train[top_features]  # Filter training data to top 5 features
X_test_proto = X_test[top_features]  # Filter testing data to top 5 features

lr_proto = LogisticRegression(random_state=42, max_iter=1000)  # Initialize 5-feature logistic regression empty model
lr_proto.fit(X_train_proto, y_train)  # Train model on 5-feature training set feed info and train to predict
proto_test_acc = accuracy_score(y_test, lr_proto.predict(X_test_proto))  # Calculate prototype test accuracy


print(f"5-feature prototype — held-out test accuracy: {proto_test_acc:.3f}")  # Print prototype test accuracy

print(classification_report(y_test, lr_proto.predict(X_test_proto), target_names=CLASS_NAMES))  # Print classification metrics report
# tests how well the 5-feature model performs on brand-new, unseen data and prints the score card.

# Quantify the accuracy trade-off explicitly — previously flagged in report
# limitations as "likely lower... but this drop has not been quantified"
full_model = joblib.load(folder + 'best_logistic_regression.pkl')  # Load fitted 15-feature full model previously developed pkl file 
full_test_acc = accuracy_score(y_test, full_model.predict(X_test))  # Calculate full model test accuracy

print(f"Full 15-feature best model test accuracy: {full_test_acc:.3f}")  # Print full model accuracy
print(f"5-feature prototype test accuracy:        {proto_test_acc:.3f}")  # Print prototype model accuracy
print(f"Accuracy trade-off from simplifying to 5 UI-friendly features: {full_test_acc - proto_test_acc:+.3f}")  # Print accuracy difference


feature_idx = [selected_features.index(f) for f in top_features]  # Get column indices of top 5 features

proto_scaler = StandardScaler()  # Initialize empty StandardScaler instance

proto_scaler.mean_ = main_scaler.mean_[feature_idx]  # Assign sliced mean values AVERAGE
proto_scaler.scale_ = main_scaler.scale_[feature_idx]  # Assign sliced scale values DEVIATION 
proto_scaler.var_ = main_scaler.var_[feature_idx]  # Assign sliced variance values SQARE OF DEVIATION 
proto_scaler.n_features_in_ = len(top_features)  # Set number of input features (5) SET LENGHT OF DIGITS FOR INPUTS 
proto_scaler.feature_names_in_ = np.array(top_features)  # Set input feature names array

print("proto_scaler built directly from main_scaler's fitted parameters")  # Log scaler creation status
print("Feature order:", top_features)  # Print feature order
print("Means:", proto_scaler.mean_)  # Print feature means
print("Scales:", proto_scaler.scale_)  # Print feature scale factors

joblib.dump(lr_proto, folder + 'disease_risk_model.pkl')  # Save trained prototype model to disk
joblib.dump(proto_scaler, folder + 'feature_scaler.pkl')  # Save sliced scaler to disk
print("Model and scaler saved as deployment-ready artefacts: "  # Print confirmation message
      "disease_risk_model.pkl, feature_scaler.pkl")
print("NOTE: no UI, API, or application code has been built. These files ")  # Print architecture note line 1
print("are inputs for a future interface, not a working prototype on their own.")  # Print architecture note line 2

# what is a picle  file  [ A .pkl file (short for Pickle file) is a file format used in Python to save and store Python objects directly to disk.] 
