# SmartCare Hospital — Disease Risk Classification

CCS3440 Artificial Intelligence Coursework — Option C
Group 04 "WHALES" (Batch 2027A)

## Project Structure

```
smartcare-optionC/
├── data/
│   ├── raw/smartcare_ai_dataset_1000.csv     # original dataset
│   └── processed/cleaned_data.csv            # after Task 03 cleaning
├── notebooks/
│   └── CCS3440_DiseaseRisk_Prediction.ipynb  # Tasks 02-07, fully executed
├── src/
│   ├── preprocessing.py         # Task 03: cleaning, missing values
│   ├── feature_engineering.py   # Task 03: derived features, encoding
│   ├── train_model.py           # Task 05: trains LR / RF / XGBoost
│   ├── evaluate_model.py        # Task 06: metrics, confusion matrices
│   └── predict.py               # Task 08 backend: single-patient prediction
├── models/                      # trained model + preprocessing artifacts (.pkl)
├── app/                         # Task 08: Streamlit prototype (self-contained)
│   ├── app.py
│   └── requirements.txt
├── reports/
│   └── CCS3440_Technical_Report_OptionC.pdf   # Task 09
├── slides/          # add your presentation here
├── video/            # add your demo video here
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the pipeline

```bash
cd src
python preprocessing.py       # -> data/processed/cleaned_data.csv
python feature_engineering.py
python train_model.py         # -> models/*.pkl
python evaluate_model.py      # -> prints metrics table + per-class report
```

## Running the prototype locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Deploying the prototype

Deploy via Streamlit Community Cloud (share.streamlit.io), pointing at `app/app.py`
in this repository.

Live URL: **[add your deployed Streamlit URL here]**

## Pushing this project to GitHub

```bash
# from inside the smartcare-optionC/ folder
git init
git add .
git commit -m "Initial commit: CCS3440 disease risk classification project"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

Repository link: **[add your GitHub repo URL here — this goes in your submission]**

## Key Findings

- **No leakage issue** (unlike Option B/readmission): disease_risk_level applies
  to every patient regardless of admission status, and clinical vitals scale
  gradually and sensibly across Low/Medium/High risk — confirming genuine
  multi-feature predictive signal rather than a single leaking column.
- **Best model: Logistic Regression** — 91.0% accuracy, 0.902 macro F1, and
  (unlike Option B's Random Forest) a genuinely strong, non-degenerate result
  confirmed via the per-class classification report.
- **Top predictive features (SHAP)**: blood sugar, cholesterol, age, and BMI —
  closely matching real-world clinical risk factors.
- **Ethical note**: age ranks 3rd of 62 features in importance — worth
  discussing with clinical stakeholders before deployment (see report Section 9.4).

## Team

| Name | Contribution |
|---|---|
| [add name] | [add contribution] |
| [add name] | [add contribution] |
| [add name] | [add contribution] |
| [add name] | [add contribution] |
