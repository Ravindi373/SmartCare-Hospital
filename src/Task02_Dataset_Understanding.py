"""
CCS3440 Artificial Intelligence Coursework | Group 02
Option C: Disease Risk Classification - SmartCare Hospital
Task 02 – Dataset Understanding
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Resolve project base directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

raw_csv = DATA_DIR / "raw" / "smartcare_ai_dataset_1000.csv"
dict_csv = DATA_DIR / "data_dictionary.csv"

# Fallback for Google Colab if needed
if not raw_csv.exists():
    raw_csv = Path("/content/drive/MyDrive/SmartCare/smartcare_ai_dataset_1000.csv")
    dict_csv = Path("/content/drive/MyDrive/SmartCare/smartcare_ai_dataset_data_dictionary.csv")

print(f"Loading data from: {raw_csv}")
df = pd.read_csv(raw_csv)
if dict_csv.exists():
    data_dict = pd.read_csv(dict_csv)
    print("\n--- Data Dictionary Preview ---")
    print(data_dict.head(10))

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Dataset Info & Types ---")
df.info()

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Target Variable Distribution (disease_risk_level) ---")
print(df["disease_risk_level"].value_counts(dropna=False))
print("\nTarget Proportions:")
print((df["disease_risk_level"].value_counts(normalize=True) * 100).round(2))

print("\n--- Missing Values Count ---")
missing = df.isnull().sum()
print(missing[missing > 0] if (missing > 0).any() else "No missing values found.")

if "admitted" in df.columns and "room_type" in df.columns:
    print("\n--- Cross-tabulation: Admitted vs Room Type Missingness ---")
    print(pd.crosstab(df["admitted"], df["room_type"].isnull(), rownames=["admitted"], colnames=["room_type_is_null"]))

print("\n--- Duplicate Check ---")
print("Fully duplicated rows:", df.duplicated().sum())
print("Duplicate patient_id + appointment_date combos:",
      df.duplicated(subset=["patient_id", "appointment_date"]).sum() if "patient_id" in df.columns else 0)

# Comprehensive Data Quality Report
quality_report = pd.DataFrame({
    "dtype": df.dtypes,
    "n_missing": df.isnull().sum(),
    "pct_missing": (df.isnull().sum() / len(df) * 100).round(2),
    "n_unique": df.nunique()
}).sort_values("pct_missing", ascending=False)

print("\n--- Data Quality Report ---")
print(quality_report.head(15))