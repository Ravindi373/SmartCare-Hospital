"""
Task 02 – Dataset Understanding
Auto-generated from Notebook/SmartCare_Hospital.ipynb (source of truth).
Regenerate this file if the notebook changes, so src/ and the notebook stay in sync.
"""

# # Task 02 – Dataset Understanding

# ## Mount Drive & Load Data

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# change this path to wherever you put your files in Drive
folder = '/content/drive/MyDrive/SmartCare/'

df = pd.read_csv(folder + 'smartcare_ai_dataset_1000.csv')
data_dict = pd.read_csv(folder + 'smartcare_ai_dataset_data_dictionary.csv')

print(df.shape)
df.head()

# ## 2.1 Dataset Overview

# Basic info
df.info()

# ## 2.2 Attribute Description

# Data dictionary (attribute meanings)
data_dict

# ## 2.3  Identify target variables.

# Target variable for Option C
df['disease_risk_level'].value_counts()

# Missing values check
df.isnull().sum()[df.isnull().sum() > 0]

# Check missing room_type by admitted status
pd.crosstab(df['admitted'], df['room_type'].isnull(), rownames=['admitted'], colnames=['room_type_is_null'])


# Check for duplicate rows
print('Fully duplicated rows:', df.duplicated().sum())
print('Duplicate patient_id + appointment_date combos:',
      df.duplicated(subset=['patient_id','appointment_date']).sum())

# Check data quality
quality_report = pd.DataFrame({
    'dtype': df.dtypes,
    'n_missing': df.isnull().sum(),
    'pct_missing': (df.isnull().sum() / len(df) * 100).round(2),
    'n_unique': df.nunique()
}).sort_values('pct_missing', ascending=False)
quality_report
