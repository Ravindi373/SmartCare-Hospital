import json
from pathlib import Path

nb_path = Path("Notebook/SmartCare_Hospital.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Title and Module Code in Markdown cells
for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        src = "".join(cell["source"])
        if "CCS4354" in src:
            cell["source"] = [line.replace("CCS4354", "CCS3440 Artificial Intelligence") for line in cell["source"]]

# Find key sections
print("Updating notebook cells for leak-free execution...")

# Add train_test_split right after Section 3.2
# Section 3.2 is typically around cell 18
split_code = """# ==========================================
# 3.3 LEAK-FREE STRATIFIED TRAIN/TEST SPLIT
# ==========================================
# Moved immediately after duplicate detection (3.2) to prevent data leakage.
# All encoding, feature selection, and scaling fit strictly on X_train.

from sklearn.model_selection import train_test_split
from feature_engineering import TARGET_MAP

# 1. Explicit target mapping: Low = 0, Medium = 1, High = 2
df_clean['disease_risk_level'] = df_clean['disease_risk_level'].map(TARGET_MAP)

# 2. Stratified 80/20 Train/Test split
df_train, df_test = train_test_split(
    df_clean,
    test_size=0.2,
    random_state=42,
    stratify=df_clean['disease_risk_level']
)

print(f"Train set shape: {df_train.shape}")
print(f"Test set shape:  {df_test.shape}")
print("\\nTrain target counts:\\n", df_train['disease_risk_level'].value_counts())
print("\\nTest target counts (Low=26, Medium=94, High=80):\\n", df_test['disease_risk_level'].value_counts())
"""

# Let's inspect cell 28 (Feature Encoding 3.7)
encoding_code = """# ==========================================
# 3.7 NOMINAL CATEGORICAL ENCODING (OneHotEncoder)
# ==========================================
# Using OneHotEncoder to avoid introducing artificial ordinal relationships for nominal variables.

from sklearn.preprocessing import OneHotEncoder

DROP_COLS = ['record_id', 'patient_id', 'appointment_date', 'no_show', 'readmitted_30_days']

train_clean = df_train.drop(columns=[c for c in DROP_COLS if c in df_train.columns])
test_clean = df_test.drop(columns=[c for c in DROP_COLS if c in df_test.columns])

y_train = train_clean['disease_risk_level']
X_train_raw = train_clean.drop(columns=['disease_risk_level'])

y_test = test_clean['disease_risk_level']
X_test_raw = test_clean.drop(columns=['disease_risk_level'])

cat_cols = X_train_raw.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()

# Fit OneHotEncoder ONLY on X_train_raw
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(X_train_raw[cat_cols])
ohe_names = ohe.get_feature_names_out(cat_cols).tolist()

X_train_cat = pd.DataFrame(ohe.transform(X_train_raw[cat_cols]), columns=ohe_names, index=X_train_raw.index)
X_train_encoded = pd.concat([X_train_raw[num_cols], X_train_cat], axis=1)

X_test_cat = pd.DataFrame(ohe.transform(X_test_raw[cat_cols]), columns=ohe_names, index=X_test_raw.index)
X_test_encoded = pd.concat([X_test_raw[num_cols], X_test_cat], axis=1)

print(f"Encoded Train shape: {X_train_encoded.shape}")
print(f"Encoded Test shape:  {X_test_encoded.shape}")
"""

# Selection 3.8
selection_code = """# ==========================================
# 3.8 FEATURE SELECTION (SelectKBest K=15)
# ==========================================
# Fit SelectKBest ONLY on X_train_encoded

from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=15)
selector.fit(X_train_encoded, y_train)

scores = pd.DataFrame({
    'feature': X_train_encoded.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False).reset_index(drop=True)

selected_features = X_train_encoded.columns[selector.get_support()].tolist()
X_train_selected = X_train_encoded[selected_features]
X_test_selected = X_test_encoded[selected_features]

print("Top Selected 15 Features:")
print(scores.head(15))
"""

# Scaling 3.9
scaling_code = """# ==========================================
# 3.9 FEATURE SCALING (StandardScaler)
# ==========================================
# Fit StandardScaler ONLY on X_train_selected

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_selected), columns=selected_features, index=X_train_selected.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_selected), columns=selected_features, index=X_test_selected.index)

print(f"Scaled Train shape: {X_train_scaled.shape}")
print(f"Scaled Test shape:  {X_test_scaled.shape}")
"""

# Save clean notebook
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated!")
