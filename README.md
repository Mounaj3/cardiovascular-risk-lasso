# Framingham CVD Risk — Systolic Blood Pressure Prediction

Predicting systolic blood pressure using Lasso regression
on the Framingham Heart Study dataset.

## About the data

The Framingham Heart Study is a long-term cardiovascular study
started in 1948 in Framingham, Massachusetts.
The dataset contains 4,240 patients with clinical measurements
such as cholesterol, BMI, glucose, heart rate, and smoking habits.

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset

Place the file at: data/raw/framingham.csv

## What this project does

We predict systolic blood pressure (sysBP) as a continuous target
using Lasso regression, which performs automatic feature selection
by shrinking irrelevant coefficients to zero.
The analysis focuses on understanding which clinical factors
drive blood pressure predictions.

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
# Step 1 — Preprocess the data
python preprocessing.py

# Step 2 — Train and evaluate the Lasso model
python model.py --alpha 1.0 --n-folds 5

# Step 3 — Analyze Lasso feature selection
python lasso_analysis.py
```

## Results

| Metric | Value |
|--------|-------|
| RMSE   | 14.46 |
| MAE    | 10.85 |
| R²     | 0.528 |

Lasso zeroed out 7 out of 15 features, keeping only
the most clinically relevant predictors.

## Outputs

After running all scripts, the outputs/ folder contains:
- predictions.png — actual vs predicted sysBP
- residuals.png — residual analysis
- lasso_coefficients.png — kept vs zeroed features
- lasso_regularization_path.png — how features get eliminated
- lasso_alpha_vs_r2.png — optimal regularization strength

## Tech stack

Python · scikit-learn · Lasso · Pandas · Matplotlib
