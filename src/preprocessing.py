"""
CCS3440 Artificial Intelligence Coursework | Group 02
Option C: Disease Risk Classification - SmartCare Hospital
Module: Preprocessing & Data Cleaning
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "raw" / "smartcare_ai_dataset_1000.csv"


def load_and_clean_data(csv_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load raw patient dataset, impute missing values conditionally, and remove duplicate rows.
    Dataset contains exactly 1,000 patient records.
    """
    df = pd.read_csv(csv_path)
    df_clean = df.copy()

    # Conditional imputation for room_type
    # admitted=0 -> 'Not Admitted' (meaningful category for outpatients)
    if "admitted" in df_clean.columns and "room_type" in df_clean.columns:
        df_clean.loc[df_clean["admitted"] == 0, "room_type"] = (
            df_clean.loc[df_clean["admitted"] == 0, "room_type"].fillna("Not Admitted")
        )

        # admitted=1 but still missing -> fill with mode room type among admitted patients
        admitted_rooms = df_clean.loc[df_clean["admitted"] == 1, "room_type"].dropna()
        mode_room = admitted_rooms.mode()[0] if not admitted_rooms.empty else "General Ward"
        df_clean.loc[df_clean["admitted"] == 1, "room_type"] = (
            df_clean.loc[df_clean["admitted"] == 1, "room_type"].fillna(mode_room)
        )

    # Remove exact duplicate records if any
    df_clean = df_clean.drop_duplicates()

    return df_clean


def remove_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove identifiers and unrelated prediction targets to prevent target & feature leakage.
    Dropped columns:
      - record_id, patient_id (arbitrary identifiers)
      - appointment_date (timestamp)
      - no_show (unrelated operational target)
      - readmitted_30_days (unrelated clinical outcome target)
    """
    drop_cols = ["record_id", "patient_id", "appointment_date", "no_show", "readmitted_30_days"]
    return df.drop(columns=[col for col in drop_cols if col in df.columns])
