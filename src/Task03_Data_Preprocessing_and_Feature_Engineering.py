"""
Task 03 – Data Preprocessing and Feature Engineering
Auto-generated from Notebook/SmartCare_Hospital.ipynb (source of truth).
Regenerate this file if the notebook changes, so src/ and the notebook stay in sync.
"""

# # Task 03 – Data Preprocessing and Feature Engineering

# ## 3.1 Missing value handling

# Missing Values Handling
df.isnull().sum()[df.isnull().sum() > 0]

pd.crosstab(df['admitted'], df['room_type'].isnull())

df_clean = df.copy()

# admitted=0 -> genuinely "no room", meaningful category
df_clean.loc[df_clean['admitted'] == 0, 'room_type'] = \
    df_clean.loc[df_clean['admitted'] == 0, 'room_type'].fillna('Not Admitted')

# admitted=1 but still missing -> real data gap, fill with most common room type
mode_room = df_clean.loc[df_clean['admitted'] == 1, 'room_type'].mode()[0]
df_clean.loc[df_clean['admitted'] == 1, 'room_type'] = \
    df_clean.loc[df_clean['admitted'] == 1, 'room_type'].fillna(mode_room)

print("Remaining missing values:", df_clean['room_type'].isnull().sum())
print(df_clean['room_type'].value_counts())

# ## 3.2 Duplicate record detection

# Duplicate Record Detection
full_row_dupes = df.duplicated().sum()
patient_id_dupes = df['patient_id'].duplicated().sum()
record_id_dupes = df['record_id'].duplicated().sum()

print(f"Fully duplicated rows        : {full_row_dupes}")
print(f"Duplicated patient_id values : {patient_id_dupes}")
print(f"Duplicated record_id values  : {record_id_dupes}")

df_clean = df_clean.drop_duplicates()

# ## 3.3 Outlier identification

# Outlier identificatio (IQR method)
numeric_cols = ['age','bmi','systolic_bp','diastolic_bp','blood_sugar_mg_dl','cholesterol_mg_dl','total_bill_lkr']

for col in numeric_cols:
    Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")

# ## 3.4 Feature Engineering

# Feature Engineering
df_feat = df_clean.copy()

df_feat['appointment_date'] = pd.to_datetime(df_feat['appointment_date'])

# Age group — standard clinical age bands
df_feat['age_group'] = pd.cut(
    df_feat['age'], bins=[0, 17, 35, 50, 65, 120],
    labels=['Under 18', '18-35', '36-50', '51-65', '65+']
)

# BMI category — WHO standard cutoffs
df_feat['bmi_category'] = pd.cut(
    df_feat['bmi'], bins=[0, 18.5, 25, 30, 100],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)

# Blood pressure category — clinical staging (systolic-led)
def bp_category(row):
    s, d = row['systolic_bp'], row['diastolic_bp']
    if s < 120 and d < 80:
        return 'Normal'
    elif s < 130 and d < 80:
        return 'Elevated'
    else:
        return 'Stage 1 Hypertension'
df_feat['bp_category'] = df_feat.apply(bp_category, axis=1)

# Missed-appointment rate (guard against divide-by-zero)
df_feat['missed_appointment_rate'] = np.where(
    df_feat['previous_appointments'] > 0,
    df_feat['missed_previous_appointments'] / df_feat['previous_appointments'],
    0.0
)

# Chronic-patient flag — repeat admissions signal a chronic condition
df_feat['is_chronic_patient'] = (df_feat['previous_admissions'] >= 2).astype(int)

# Care intensity — combined labs + treatments in this visit
df_feat['care_intensity'] = df_feat['lab_tests_count'] + df_feat['treatments_count']

# Appointment seasonality
df_feat['appointment_month'] = df_feat['appointment_date'].dt.month
df_feat['appointment_quarter'] = df_feat['appointment_date'].dt.quarter

# Convert pd.cut Categorical outputs to plain strings (needed before encoding)
df_feat['age_group'] = df_feat['age_group'].astype(str)
df_feat['bmi_category'] = df_feat['bmi_category'].astype(str)

print("Shape after feature engineering:", df_feat.shape)
df_feat.head()

# ## 3.5 Data Cleaning

# Data Cleaning - drop irrelevant / leakage columns
drop_cols = ['record_id', 'patient_id', 'appointment_date', 'no_show', 'readmitted_30_days']
df_model = df_feat.drop(columns=drop_cols)   # built from df_feat (keeps engineered features)

print("Shape after cleaning:", df_model.shape)

# ## 3.6 Save Cleaned Dataset

# Save cleaned dataset (once — human-readable, before encoding)

df_feat.to_csv(folder + 'smartcare_cleaned.csv', index=False)
print("Saved:", folder + 'smartcare_cleaned.csv', df_feat.shape)

from google.colab import files
files.download(folder + 'smartcare_cleaned.csv')

# ## 3.7 Train/Test Split

# TRAIN/TEST SPLIT — now done FIRST, before encoding/selection/scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

target_encoder = LabelEncoder()
y_all = pd.Series(
    target_encoder.fit_transform(df_model['disease_risk_level']),
    index=df_model.index,
    name='disease_risk_level'
)
X_all = df_model.drop(columns=['disease_risk_level'])

print("Target classes (index -> label):", list(enumerate(target_encoder.classes_)))

pd.Series(target_encoder.classes_, name='class_name').to_csv(
    folder + 'target_classes.csv', index_label='encoded_value'
)
print("Saved true class order to:", folder + 'target_classes.csv')

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print("Train size:", X_train_raw.shape, "Test size:", X_test_raw.shape)


# ## 3.8 Feature Encoding

# Feature Encoding — fit on TRAIN ONLY, then apply the same mapping to test
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Ordinal: engineered bands with a genuine natural order
ordinal_cols = ['age_group', 'bmi_category', 'bp_category']
ordinal_categories = [
    ['Under 18', '18-35', '36-50', '51-65', '65+'],
    ['Underweight', 'Normal', 'Overweight', 'Obese'],
    ['Normal', 'Elevated', 'Stage 1 Hypertension'],
]

# Nominal: no inherent order — one-hot encode instead of label/ordinal encode
nominal_cols = ['gender', 'blood_group', 'department', 'diagnosis',
                 'appointment_status', 'room_type', 'payment_status', 'payment_method']

other_cols = [c for c in X_train_raw.columns if c not in ordinal_cols + nominal_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OrdinalEncoder(categories=ordinal_categories,
                                handle_unknown='use_encoded_value', unknown_value=-1),
         ordinal_cols),
        ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         nominal_cols),
        ('num', 'passthrough', other_cols),
    ]
)

preprocessor.fit(X_train_raw)  # <-- train data only

encoded_feature_names = (
    ordinal_cols +
    list(preprocessor.named_transformers_['nom'].get_feature_names_out(nominal_cols)) +
    other_cols
)

X_train_enc = pd.DataFrame(
    preprocessor.transform(X_train_raw),
    columns=encoded_feature_names,
    index=X_train_raw.index
)
X_test_enc = pd.DataFrame(
    preprocessor.transform(X_test_raw),
    columns=encoded_feature_names,
    index=X_test_raw.index
)

print("Encoding complete. Train shape:", X_train_enc.shape, "Test shape:", X_test_enc.shape)
X_train_enc.head()

# ## 3.9 Feature Selection

# Feature Selection — fit on TRAIN ONLY
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=15)
selector.fit(X_train_enc, y_train)  # <-- train data only

scores = pd.DataFrame({
    'feature': X_train_enc.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)
print(scores)

selected_features = scores['feature'].head(15).tolist()
print("\nSelected features (from training data only):", selected_features)

X_train_selected = X_train_enc[selected_features]
X_test_selected = X_test_enc[selected_features]   # same columns, no fitting on test

# ## 3.10 Feature Scaling

# Feature Scaling — fit on TRAIN ONLY, transform both splits with those statistics
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_selected),   # <-- fit on train only
    columns=selected_features,
    index=X_train_selected.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_selected),        # <-- test only ever transformed
    columns=selected_features,
    index=X_test_selected.index
)

# Persist the fitted scaler itself, not just the scaled data. Task 08's
# prototype model reuses this exact scaler's parameters (subset to its 5
# features) instead of fitting a new one on a different pipeline.
import joblib
joblib.dump(scaler, folder + 'feature_scaler_main.pkl')
print("Saved main fitted scaler to:", folder + 'feature_scaler_main.pkl')

X_train_scaled.head()

# ## 3.11 Save Model-Ready Splits

# Save model-ready splits (Task 05) — X_train/X_test are now leakage-free:
# encoding, feature selection, and scaling were all fit on X_train only.
X_train_scaled.to_csv(folder + 'X_train.csv', index=False)
X_test_scaled.to_csv(folder + 'X_test.csv', index=False)
y_train.to_csv(folder + 'y_train.csv', index=False)
y_test.to_csv(folder + 'y_test.csv', index=False)

print("Saved to Drive:", folder)

from google.colab import files
files.download(folder + 'X_train.csv')
files.download(folder + 'X_test.csv')
files.download(folder + 'y_train.csv')
files.download(folder + 'y_test.csv')
