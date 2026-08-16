"""
CCS3440 Artificial Intelligence Coursework | Group 02
Option C: Disease Risk Classification - SmartCare Hospital
Task 04 – Exploratory Data Analysis (EDA)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

csv_path = DATA_DIR / "processed" / "smartcare_cleaned.csv"
if not csv_path.exists():
    from preprocessing import load_and_clean_data
    from feature_engineering import engineer_features
    raw_path = DATA_DIR / "raw" / "smartcare_ai_dataset_1000.csv"
    df_clean = load_and_clean_data(raw_path)
    df_feat = engineer_features(df_clean)
else:
    df_feat = pd.read_csv(csv_path)

if "appointment_date" in df_feat.columns:
    df_feat["appointment_date"] = pd.to_datetime(df_feat["appointment_date"])

print("==================================================")
print("  Task 04: Exploratory Data Analysis (EDA)")
print("==================================================")
print("Dataset Shape:", df_feat.shape)

# 1. Descriptive Statistics of Clinical Features
clinical_cols = ["age", "bmi", "systolic_bp", "diastolic_bp", "blood_sugar_mg_dl", "cholesterol_mg_dl"]
print("\n--- Descriptive Statistics for Clinical Vitals ---")
print(df_feat[clinical_cols].describe().round(1))

# 2. Mean of Clinical Values Grouped by Risk Level
print("\n--- Mean Clinical Values by Disease Risk Level ---")
print(df_feat.groupby("disease_risk_level")[clinical_cols].mean().round(1))

# 3. Class Distribution Chart
plt.figure(figsize=(6, 4))
sns.countplot(x="disease_risk_level", data=df_feat, order=["Low", "Medium", "High"], palette="viridis")
plt.title("Disease Risk Level Distribution")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_class_distribution.png", dpi=120)
plt.close()

# 4. Histograms
df_feat[clinical_cols].hist(figsize=(12, 7), bins=25, color="teal")
plt.suptitle("Clinical Vital Distributions", y=1.02)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_clinical_histograms.png", dpi=120)
plt.close()

# 5. Boxplots Across Risk Levels
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flatten(), clinical_cols):
    sns.boxplot(x="disease_risk_level", y=col, data=df_feat, order=["Low", "Medium", "High"], ax=ax, palette="Set2")
    ax.set_title(f"{col} by Risk Level")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_boxplots_by_risk.png", dpi=120)
plt.close()

# 6. Correlation Heatmap
plt.figure(figsize=(12, 9))
corr = df_feat.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap — All Numeric Features")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_correlation_heatmap.png", dpi=120)
plt.close()

# 7. Stacked Bar: Risk Level Proportion by Age Group
if "age_group" in df_feat.columns:
    plt.figure(figsize=(8, 5))
    age_risk = df_feat.groupby("age_group", observed=True)["disease_risk_level"].value_counts(normalize=True).unstack()
    valid_groups = [g for g in ["Under 18", "18-35", "36-50", "51-65", "65+"] if g in age_risk.index]
    age_risk = age_risk.reindex(valid_groups)
    cols_order = [c for c in ["Low", "Medium", "High"] if c in age_risk.columns]
    age_risk[cols_order].plot(kind="bar", stacked=True, colormap="RdYlGn_r", figsize=(8, 5))
    plt.title("Risk Level Proportion by Age Group")
    plt.ylabel("Proportion")
    plt.legend(title="Risk Level")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "eda_risk_by_age_group.png", dpi=120)
    plt.close()

print(f"\n[SUCCESS] Task 04 completed! EDA plots saved to: {REPORTS_DIR}\n")