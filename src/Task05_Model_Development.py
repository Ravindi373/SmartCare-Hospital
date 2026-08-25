"""
Task 05 – Machine Learning Model Development
Auto-generated from Notebook/SmartCare_Hospital.ipynb (source of truth).
Regenerate this file if the notebook changes, so src/ and the notebook stay in sync.
"""

# # Task 05 – Machine Learning Model Development

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

# Define folder path (assuming drive is already mounted)
folder = '/content/drive/MyDrive/SmartCare/' # Added this line to define the 'folder' variable

X_train = pd.read_csv(folder + 'X_train.csv') # Changed path
y_train = pd.read_csv(folder + 'y_train.csv').squeeze() # Changed path
X_test = pd.read_csv(folder + 'X_test.csv') # Changed path
y_test = pd.read_csv(folder + 'y_test.csv').squeeze() # Changed path

print("Train:", X_train.shape, " Test:", X_test.shape)
print("\nTrain class distribution:\n", y_train.value_counts(normalize=True).sort_index().round(3))
print("\nTest class distribution:\n",  y_test.value_counts(normalize=True).sort_index().round(3))


# ## 5.1 Model Training & 5.2 Hyperparameter Selection

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

# ### Model 1 — Logistic Regression

lr_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'max_iter': [2000],
    'class_weight': ['balanced'],
}
best_lr = tune_and_report('Logistic Regression', LogisticRegression(random_state=RANDOM_STATE), lr_grid, X_train, y_train)

# ### Model 2 — Decision Tree

dt_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_leaf': [1, 5, 10],
    'min_samples_split': [2, 10],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced'],
}
best_dt = tune_and_report('Decision Tree', DecisionTreeClassifier(random_state=RANDOM_STATE), dt_grid, X_train, y_train)

# ### Model 3 — Random Forest

rf_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 3, 5],
    'class_weight': ['balanced'],
}
best_rf = tune_and_report('Random Forest', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), rf_grid, X_train, y_train)


# ### Model 4 — Support Vector Machine

svm_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto'],
    'class_weight': ['balanced'],
    'probability': [True],
}
best_svm = tune_and_report('SVM', SVC(random_state=RANDOM_STATE), svm_grid, X_train, y_train)

# ### Model 5 — XGBoost

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

cv_summary_df = pd.DataFrame(cv_summary).sort_values('Best CV Macro-F1', ascending=False)
cv_summary_df[['Model', 'Best CV Macro-F1']].reset_index(drop=True)

# ## 5.3 Persist Fitted Best-Estimators

# Persist the exact fitted best-estimators from GridSearchCV
import joblib

for name, model in results.items():
    safe_name = name.lower().replace(' ', '_')
    joblib.dump(model, folder + f'best_{safe_name}.pkl')

cv_summary_df.to_csv(folder + 'task05_cv_summary.csv', index=False)
print("Saved 5 fitted best-estimators and CV summary to:", folder)
print("Files:", [f'best_{n.lower().replace(" ", "_")}.pkl' for n in results.keys()])

# ## 5.4 Comparative Analysis

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

print("Final ranked comparison (test set):")
test_results_df.round(4)

import joblib
best_overall_model = results[best_model_name]
joblib.dump(best_overall_model, 'best_model.pkl')
test_results_df.to_csv('task05_model_comparison_results.csv', index=False)
print(f"Saved best model ({best_model_name}) to best_model.pkl")
print("Saved comparison table to task05_model_comparison_results.csv")

# ## Neural Network (Deep Learning) — bonus addition

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd # Import pandas to load the data

# Load X_train and y_train
try:
    X_train = pd.read_csv(folder + 'X_train.csv')
    y_train = pd.read_csv(folder + 'y_train.csv').squeeze() # .squeeze() to convert Series to 1D array if needed
except FileNotFoundError:
    print(f"Error: The file 'X_train.csv' or 'y_train.csv' was not found in the specified path: {folder}.")
    print("Please ensure that you have successfully saved the data in Task 03 and that the 'SmartCare' folder exists in your Google Drive 'MyDrive'.")
    print("If the files exist, verify the exact path and permissions.")
    raise # Re-raise the original error after providing context

tf.random.set_seed(42)
np.random.seed(42)

# Build the neural network
nn_model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')   # 3 output classes: High, Low, Medium
])

nn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # works directly with integer labels (0,1,2), no one-hot needed
    metrics=['accuracy']
)

nn_model.summary()

# Train with early stopping (prevents overfitting on a moderate-sized dataset)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = nn_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

print(f"Training stopped after {len(history.history['loss'])} epochs (early stopping)")

# Training curves — required for demonstrating training behavior

fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Loss over epochs')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1].set_title('Accuracy over epochs')
axes[1].set_xlabel('Epoch')
axes[1].legend()
plt.tight_layout()
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure results and test_results are defined for this cell's execution, in case of kernel restart.
# WARNING: If Task 05 models were not run, these will be empty except for NN.
if 'results' not in globals():
    results = {}
if 'test_results' not in globals():
    test_results = []

# Evaluate on test set — same metrics as other models, for fair comparison

nn_probs = nn_model.predict(X_test)
nn_preds = np.argmax(nn_probs, axis=1)

nn_acc = accuracy_score(y_test, nn_preds)
nn_prec = precision_score(y_test, nn_preds, average='weighted')
nn_rec = recall_score(y_test, nn_preds, average='weighted')
nn_f1 = f1_score(y_test, nn_preds, average='weighted')

print("Neural Network Results")
print(f"Accuracy={nn_acc:.3f}  Precision={nn_prec:.3f}  Recall={nn_rec:.3f}  F1={nn_f1:.3f}")

results['Neural Network'] = nn_model
test_results.append({
    'Model': 'Neural Network',
    'Accuracy': nn_acc,
    'Precision (macro)': nn_prec,
    'Recall (macro)': nn_rec,
    'F1 (macro)': nn_f1,
})

print("results has", len(results), "entries")

# Refresh comparison table with Neural Network included
# The 'test_results' list already contains the metrics for all models, including the Neural Network.
# We just need to convert this list of dictionaries into a DataFrame.
results_df = pd.DataFrame(test_results)
results_df = results_df.sort_values('F1 (macro)', ascending=False).reset_index(drop=True)
results_df

nn_model.save(folder + 'neural_network_model.keras')
print("Neural network saved.")

from google.colab import files
files.download(folder + 'neural_network_model.keras')
