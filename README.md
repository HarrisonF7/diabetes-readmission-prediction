# Diabetes Readmission Prediction

Machine learning models for predicting 30-day hospital readmission among patients with diabetes.

## Research Question

Among hospitalized patients with diabetes, how accurately can clinical information available at discharge predict hospital readmission within 30 days?

## Dataset

This project uses the Diabetes 130-US Hospitals dataset, containing 101,766 hospital encounters. After excluding encounters with discharge dispositions indicating death or hospice care, 99,343 encounters from 69,990 unique patients remained eligible.

The dataset is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

## Methods

We used 14 clinical and demographic predictors to develop two machine learning models:

- Logistic Regression
- Random Forest

Patients were separated into training and testing sets to prevent patient-level data leakage. Hyperparameter tuning used three-fold patient-grouped cross-validation on the training data.

## Results

The random forest achieved the highest test AUROC of 0.6349, compared with 0.6306 for logistic regression.

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| AUROC | 0.6306 | 0.6349 |
| AUPRC | 0.1938 | 0.1957 |
| Brier Score | 0.0967 | 0.0965 |

The random forest was evaluated at a classification threshold of 0.12. Additional analyses included feature importance, calibration, individual patient predictions, age subgroup performance, error analysis, and sensitivity analysis.

## How to Run

1. Download the Diabetes 130-US Hospitals dataset from the UCI Machine Learning Repository.
2. Extract the dataset and locate `diabetic_data.csv`.
3. Install Python and the required packages:

```bash
pip install marimo pandas numpy matplotlib scikit-learn
```

4. Download the Marimo Python notebook from this repository.
5. Open the notebook using:

```bash
marimo edit "notebook (12).py"
```

6. Upload `diabetic_data.csv` using the file upload control at the beginning of the notebook.
7. Allow the notebook to execute its analysis.

## Project Files

- `notebook (12).py` — Executable Marimo notebook containing the complete analysis.
- `Diabetes_Readmission_Analysis.pdf` — Executed notebook showing code, results, tables, and visualizations.
- `README.md` — Project description and instructions.

## Limitations

The models demonstrated modest predictive performance and should not be used independently for clinical decision-making. The dataset is historical, contains missing information, and may not capture readmissions outside participating hospitals.

## Reproducibility

- Random seed: 42
- Patient-level training/testing split
- Three-fold grouped cross-validation
- Marimo notebook generated with version 0.25.2

The analysis is intended for educational and research purposes.
