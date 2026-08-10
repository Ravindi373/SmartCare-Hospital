"""
CCS3440 - Task 03: Data Preprocessing (Option C - Disease Risk Classification)
Loads the raw SmartCare dataset and handles missing values. Unlike Option B,
no population filtering is required here — disease_risk_level applies to
every patient regardless of admission status.

Usage:
    from preprocessing import load_and_clean_data
    df_clean = load_and_clean_data("data/raw/smartcare_ai_dataset_1000.csv")
"""

import pandas as pd


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load the raw SmartCare Hospital dataset."""
    return pd.read_csv(csv_path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing room_type with an explicit 'Not Admitted' category.

    Justification: room_type is only recorded for admitted patients; this is
    the true, meaningful reason for the missingness, so imputing with the
    mode would fabricate a ward assignment for a patient who was never
    admitted.
    """
    df = df.copy()
    df["room_type"] = df["room_type"].fillna("Not Admitted")
    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    """Return a small data-quality summary (duplicates, remaining nulls)."""
    return {
        "n_rows": len(df),
        "duplicates": int(df.duplicated().sum()),
        "missing_values": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
    }


def check_leakage_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanity check for leakage: confirm no single column trivially separates
    the disease_risk_level classes (unlike Option B, where 'admitted'
    leaked into readmitted_30_days). Returns mean vitals per risk class —
    a gradual, monotonic increase across Low/Medium/High is expected and
    indicates genuine multi-feature signal rather than a leaking column.
    """
    return df.groupby("disease_risk_level")[
        ["systolic_bp", "blood_sugar_mg_dl", "cholesterol_mg_dl", "bmi", "age"]
    ].mean().round(1)


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Full Task 03 preprocessing pipeline: load -> impute missing values."""
    df = load_raw_data(csv_path)
    df = handle_missing_values(df)
    return df


if __name__ == "__main__":
    df_clean = load_and_clean_data("../data/raw/smartcare_ai_dataset_1000.csv")
    print(f"Cleaned dataset shape: {df_clean.shape}")
    print(check_data_quality(df_clean))
    print("\nMean vitals by risk class (leakage sanity check):")
    print(check_leakage_signal(df_clean))
    df_clean.to_csv("../data/processed/cleaned_data.csv", index=False)
    print("\nSaved to data/processed/cleaned_data.csv")
