"""
Task 04 – Exploratory Data Analysis (EDA)
Auto-generated from Notebook/SmartCare_Hospital.ipynb (source of truth).
Regenerate this file if the notebook changes, so src/ and the notebook stay in sync.
"""

# # Task 04 – Exploratory Data Analysis (EDA)

# ## 4.1 Load Cleaned Dataset

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

folder = '/content/drive/MyDrive/SmartCare/'
df_feat = pd.read_csv(folder + 'smartcare_cleaned.csv')
df_feat['appointment_date'] = pd.to_datetime(df_feat['appointment_date'])

print(df_feat.shape)
df_feat.head()

# ## 4.2 Descriptive Statistics

df_feat[['age','bmi','systolic_bp','diastolic_bp','blood_sugar_mg_dl','cholesterol_mg_dl']].describe().round(1)

# Mean of clinical values grouped by risk level
df_feat.groupby('disease_risk_level')[['age','bmi','systolic_bp','diastolic_bp','blood_sugar_mg_dl','cholesterol_mg_dl']].mean().round(1)

# Risk level distribution across the engineered age_group feature
df_feat.groupby('age_group', observed=True)['disease_risk_level'].value_counts().unstack()

# ## 4.3 Correlation Analysis

# Correlation Analysis
# Checked which clinical variables move together
risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['risk_num'] = df['disease_risk_level'].map(risk_map)

corr_target = df[numeric_cols + ['risk_num']].corr()['risk_num'].drop('risk_num').sort_values(key=abs, ascending=False)
print("Correlation of numeric features with disease_risk_level (ordinal-encoded):")
print(corr_target.round(3))

# ## 4.4 Class Distribution Charts

# Class Distribution Chart
plt.figure(figsize=(6,4))
sns.countplot(x='disease_risk_level', data=df_feat, order=['Low','Medium','High'], palette='viridis')
plt.title('Disease Risk Level Distribution')
plt.ylabel('Number of Patients')
plt.show()

# ## 4.5 Histograms

# Histograms — Distribution Analysis
df_feat[['age','bmi','systolic_bp','blood_sugar_mg_dl','cholesterol_mg_dl']].hist(figsize=(14,8), bins=25)
plt.tight_layout()
plt.show()

# ## 4.6 Boxplots

# Boxplots
fig, axes = plt.subplots(2, 3, figsize=(16,9))
cols = ['age','bmi','systolic_bp','diastolic_bp','blood_sugar_mg_dl','cholesterol_mg_dl']
for ax, col in zip(axes.flatten(), cols):
    sns.boxplot(x='disease_risk_level', y=col, data=df_feat, order=['Low','Medium','High'], ax=ax, palette='Set2')
    ax.set_title(f'{col} by Risk Level')
plt.tight_layout()
plt.show()

# Bonus: BMI by age_group and risk level (uses engineered feature)
plt.figure(figsize=(9,5))
sns.boxplot(x='age_group', y='bmi', hue='disease_risk_level',
            order=['Under 18','18-35','36-50','51-65','65+'],
            hue_order=['Low','Medium','High'], data=df_feat)
plt.title('BMI by Age Group and Risk Level')
plt.show()

# ## 4.7 Scatterplots — Pattern Discovery

# Scatterplots — Pattern Discovery
fig, axes = plt.subplots(1, 2, figsize=(14,5))
sns.scatterplot(x='bmi', y='blood_sugar_mg_dl', hue='disease_risk_level',
                 hue_order=['Low','Medium','High'], data=df_feat, ax=axes[0])
axes[0].set_title('BMI vs Blood Sugar by Risk Level')

sns.scatterplot(x='age', y='cholesterol_mg_dl', hue='disease_risk_level',
                 hue_order=['Low','Medium','High'], data=df_feat, ax=axes[1])
axes[1].set_title('Age vs Cholesterol by Risk Level')
plt.tight_layout()
plt.show()

# ## 4.8 Correlation Heatmap

# Correlation Heatmap
plt.figure(figsize=(14,11))
corr = df_feat.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap — All Numeric Features')
plt.show()

# Bonus — Risk Level Proportion by Age Group (stacked bar)
plt.figure(figsize=(8,5))
age_risk = df_feat.groupby('age_group', observed=True)['disease_risk_level'].value_counts(normalize=True).unstack()
age_risk = age_risk.reindex(['Under 18','18-35','36-50','51-65','65+'])
age_risk[['Low','Medium','High']].plot(kind='bar', stacked=True, colormap='RdYlGn_r', figsize=(8,5))
plt.title('Risk Level Proportion by Age Group')
plt.ylabel('Proportion')
plt.legend(title='Risk Level')
plt.tight_layout()
plt.show()
