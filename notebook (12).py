import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    file = mo.ui.file(filetypes=[".csv"])
    file
    return file, mo


@app.cell
def _(file):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import io

    df = pd.read_csv(io.BytesIO(file.contents()))
    df.shape
    return df, np, pd, plt


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df["readmitted"].value_counts()
    return


@app.cell
def _(df):
    df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)

    df["readmitted_30"].value_counts()
    return


@app.cell
def _(df):
    df["discharge_disposition_id"].value_counts().sort_index()
    return


@app.cell
def _(df):
    discharge_codes = [11, 13, 14, 19, 20, 21]

    df["discharge_disposition_id"].value_counts().reindex(
        discharge_codes, fill_value=0
    )
    return


@app.cell
def _(df):
    excluded_codes = [11, 13, 14, 19, 20, 21]

    df_eligible = df[
        ~df["discharge_disposition_id"].isin(excluded_codes)
    ].copy()

    print("Original encounters:", len(df))
    print("Excluded encounters:", len(df) - len(df_eligible))
    print("Eligible encounters:", len(df_eligible))
    print("Unique patients:", df_eligible["patient_nbr"].nunique())
    return (df_eligible,)


@app.cell
def _(df_eligible):
    print("Readmission counts:")
    print(df_eligible["readmitted_30"].value_counts())

    print("\n30-day readmission rate:")
    print(round(df_eligible["readmitted_30"].mean() * 100, 2), "%")
    return


@app.cell
def _(df_eligible, np):
    missing_counts = (
        df_eligible.replace("?", np.nan)
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_counts[missing_counts > 0]
    return


@app.cell
def _(df_eligible):
    print("Exact duplicate rows:", df_eligible.duplicated().sum())

    print(
        "Duplicate encounter IDs:",
        df_eligible["encounter_id"].duplicated().sum()


    )

    print(
        "Patients with multiple encounters:",
        (df_eligible["patient_nbr"].value_counts() > 1).sum()
    )
    return


@app.cell
def _(df_eligible):
    print("Available columns:")
    for column in df_eligible.columns:
        print(column)
    return


@app.cell
def _(df_eligible):
    predictors = [
        "age",
        "gender",
        "race",
        "admission_type_id",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "insulin",
        "diabetesMed"
    ]

    df_model = df_eligible[
        ["patient_nbr", "readmitted_30"] + predictors
    ].copy()

    print("Encounters:", len(df_model))
    print("Predictors:", len(predictors))
    df_model.head()
    return df_model, predictors


@app.cell
def _(df_model):
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(
            df_model,
            df_model["readmitted_30"],
            groups=df_model["patient_nbr"]
        )
    )

    train_df = df_model.iloc[train_idx].copy()
    test_df = df_model.iloc[test_idx].copy()

    print("Training encounters:", len(train_df))
    print("Testing encounters:", len(test_df))

    print(
        "Patients in both sets:",
        len(
            set(train_df["patient_nbr"]) &
            set(test_df["patient_nbr"])
        )
    )
    return test_df, train_df


@app.cell
def _(test_df, train_df):
    print("Training readmission counts:")
    print(train_df["readmitted_30"].value_counts())

    print("\nTraining readmission rate:")
    print(round(train_df["readmitted_30"].mean() * 100, 2), "%")

    print("\nTesting readmission counts:")
    print(test_df["readmitted_30"].value_counts())

    print("\nTesting readmission rate:")
    print(round(test_df["readmitted_30"].mean() * 100, 2), "%")
    return


@app.cell
def _(train_df):
    baseline_probability = train_df["readmitted_30"].mean()

    print("Baseline readmission probability:")
    print(round(baseline_probability, 4))

    print("\nBaseline readmission percentage:")
    print(round(baseline_probability * 100, 2), "%")
    return (baseline_probability,)


@app.cell
def _(np, predictors, test_df, train_df):
    X_train = train_df[predictors].replace("?", np.nan)
    y_train = train_df["readmitted_30"]

    X_test = test_df[predictors].replace("?", np.nan)
    y_test = test_df["readmitted_30"]

    print("Training predictors:", X_train.shape)
    print("Training outcomes:", y_train.shape)
    print("Testing predictors:", X_test.shape)
    print("Testing outcomes:", y_test.shape)
    return X_test, X_train, y_test, y_train


@app.cell
def _():
    numeric_features = [
         "time_in_hospital",
         "num_lab_procedures",
         "num_procedures",
         "num_medications",
         "number_outpatient",
         "number_emergency",
         "number_inpatient",
         "number_diagnoses"
    ]

    categorical_features = [
        "age",
        "gender",
        "race",
        "admission_type_id",
        "insulin",
        "diabetesMed"
    ]

    print("Numerical predictors:", len(numeric_features))
    print("Categorical predictors:", len(categorical_features))
    print("Total predictors:", len(numeric_features) + len(categorical_features))
    return categorical_features, numeric_features


@app.cell
def _(categorical_features, numeric_features):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    print("Preprocessing pipeline created successfully!")
    return (
        ColumnTransformer,
        Pipeline,
        categorical_pipeline,
        numeric_pipeline,
        preprocessor,
    )


@app.cell
def _(Pipeline, preprocessor):
    from sklearn.linear_model import LogisticRegression

    logistic_model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    print("Logistic regression model created successfully!")
    return (logistic_model,)


@app.cell
def _(X_train, logistic_model, train_df, y_train):
    from sklearn.model_selection import GroupKFold, GridSearchCV

    cv = GroupKFold(n_splits=3)

    logistic_search = GridSearchCV(
        estimator=logistic_model,
        param_grid={
            "classifier__C": [0.1, 1.0, 10.0]
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=1
    )

    logistic_search.fit(
        X_train,
        y_train,


        groups=train_df["patient_nbr"]
    )

    print("Best C:", logistic_search.best_params_)
    print("Best cross-validation AUROC:",
          round(logistic_search.best_score_, 4))
    return GridSearchCV, GroupKFold, logistic_search


@app.cell
def _(logistic_search, pd):
    logistic_cv_results = pd.DataFrame(
        logistic_search.cv_results_
    )[[
        "param_classifier__C",
        "mean_test_score",
        "std_test_score"
    ]]

    logistic_cv_results.columns = [
        "C",
        "Mean AUROC",
        "AUROC Standard Deviation"
    ]

    logistic_cv_results
    return


@app.cell
def _(Pipeline, preprocessor):
    from sklearn.ensemble import RandomForestClassifier

    forest_model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=1
        ))
    ])

    print("Random forest model created successfully!")
    return RandomForestClassifier, forest_model


@app.cell
def _(GridSearchCV, GroupKFold, X_train, forest_model, train_df, y_train):
    forest_search = GridSearchCV(
        estimator=forest_model,
        param_grid={
            "classifier__max_depth": [5, 10],
            "classifier__min_samples_leaf": [10, 30]
        },
        scoring="roc_auc",
        cv=GroupKFold(n_splits=3),
        n_jobs=1
    )

    forest_search.fit(
        X_train,
        y_train,
        groups=train_df["patient_nbr"]
    )

    print("Best settings:", forest_search.best_params_)
    print(
        "Best cross-validation AUROC:",
        round(forest_search.best_score_, 4)
    )
    return (forest_search,)


@app.cell
def _(forest_search, pd):
    forest_cv_results = pd.DataFrame(
        forest_search.cv_results_
    )[[
        "param_classifier__max_depth",
        "param_classifier__min_samples_leaf",
        "mean_test_score",
        "std_test_score"
    ]]

    forest_cv_results.columns = [
        "Max Depth",
        "Min Samples per Leaf",
        "Mean AUROC",
        "AUROC Standard Deviation"
    ]

    forest_cv_results
    return


@app.cell
def _(forest_search, logistic_search, pd):
    model_comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Mean CV AUROC": [
            logistic_search.best_score_,
            forest_search.best_score_
        ],
        "CV Standard Deviation": [
            logistic_search.cv_results_["std_test_score"][
                logistic_search.best_index_
            ],
            forest_search.cv_results_["std_test_score"][
                forest_search.best_index_


             ]
         ]
    })

    model_comparison
    return


@app.cell
def _(GroupKFold, X_train, forest_search, train_df, y_train):
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix, recall_score, precision_score

    cv_probabilities = cross_val_predict(
        forest_search.best_estimator_,
        X_train,
        y_train,
        groups=train_df["patient_nbr"],
        cv=GroupKFold(n_splits=3),
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    threshold = 0.15
    cv_predictions = (cv_probabilities >= threshold).astype(int)

    print("Threshold:", threshold)
    print("Confusion matrix:")
    print(confusion_matrix(y_train, cv_predictions))
    print("Sensitivity:", round(recall_score(y_train, cv_predictions), 4))
    print("Precision:", round(precision_score(y_train, cv_predictions, zero_division=0), 4))
    return confusion_matrix, cv_probabilities, precision_score, recall_score


@app.cell
def _(cv_probabilities, pd, precision_score, recall_score, y_train):
    threshold_results = []

    for t in [0.10, 0.12, 0.15, 0.20]:
        predictions = (cv_probabilities >= t).astype(int)

        threshold_results.append({
            "Threshold": t,
            "Sensitivity": recall_score(y_train, predictions),
            "Precision": precision_score(
                y_train, predictions, zero_division=0
            ),
            "Flagged (%)": predictions.mean() * 100
        })

    pd.DataFrame(threshold_results).round(3)
    return


@app.cell
def _():
    final_threshold = 0.12

    print("Selected model: Random Forest")
    print("Selected threshold:", final_threshold)
    print("Maximum tree depth:", 10)
    print("Minimum samples per leaf:", 30)
    print("Selection based on training cross-validation only.")
    return (final_threshold,)


@app.cell
def _(X_test, final_threshold, forest_search, logistic_search, y_test):
    # Use the best models selected during cross-validation
    best_logistic = logistic_search.best_estimator_
    best_forest = forest_search.best_estimator_

    # Generate predicted probabilities on the held-out test set
    logistic_probabilities = best_logistic.predict_proba(X_test)[:, 1]
    forest_probabilities = best_forest.predict_proba(X_test)[:, 1]

    # Apply our previously selected 12% threshold
    logistic_predictions = (logistic_probabilities >= final_threshold).astype(int)
    forest_predictions = (forest_probabilities >= final_threshold).astype(int)

    print("Test encounters:", len(y_test))
    print("Logistic regression predictions:", len(logistic_predictions))
    print("Random forest predictions:", len(forest_predictions))
    print("Threshold used:", final_threshold)
    return (
        best_forest,
        forest_predictions,
        forest_probabilities,
        logistic_predictions,
        logistic_probabilities,
    )


@app.cell
def _(
    baseline_probability,
    forest_probabilities,
    logistic_probabilities,
    np,
    pd,
    y_test,
):
    from sklearn.metrics import roc_auc_score, average_precision_score

    # Baseline predicts the same probability for every encounter
    baseline_probs = np.full(len(y_test), baseline_probability)



    # Compare all three models
    evaluation_results = pd.DataFrame({
        "Model": ["Baseline", "Logistic Regression", "Random Forest"],
        "AUROC": [
            roc_auc_score(y_test, baseline_probs),
            roc_auc_score(y_test, logistic_probabilities),
            roc_auc_score(y_test, forest_probabilities)
        ],
        "AUPRC": [
            average_precision_score(y_test, baseline_probs),
            average_precision_score(y_test, logistic_probabilities),
            average_precision_score(y_test, forest_probabilities)
        ]
    })

    evaluation_results.round(4)
    return baseline_probs, roc_auc_score


@app.cell
def _(forest_probabilities, logistic_probabilities, plt, y_test):
    from sklearn.metrics import RocCurveDisplay

    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))

    RocCurveDisplay.from_predictions(
        y_test,
        logistic_probabilities,
        name="Logistic Regression",
        ax=ax_roc
    )

    RocCurveDisplay.from_predictions(
        y_test,
        forest_probabilities,
        name="Random Forest",
        ax=ax_roc
    )

    ax_roc.plot([0, 1], [0, 1], "k--", label="Random Guessing")

    ax_roc.set_title("ROC Curve: 30-Day Readmission Prediction")
    ax_roc.legend(loc="lower right")

    fig_roc
    return


@app.cell
def _(forest_probabilities, logistic_probabilities, plt, y_test):
    from sklearn.metrics import PrecisionRecallDisplay

    fig_pr, ax_pr = plt.subplots(figsize=(7, 5))

    PrecisionRecallDisplay.from_predictions(
        y_test,
        logistic_probabilities,
        name="Logistic Regression",
        ax=ax_pr
    )

    PrecisionRecallDisplay.from_predictions(
        y_test,
        forest_probabilities,
        name="Random Forest",
        ax=ax_pr
    )

    ax_pr.axhline(
        y_test.mean(),
        color="black",
        linestyle="--",
        label="Baseline"
    )

    ax_pr.set_title("Precision-Recall Curve: 30-Day Readmission")
    ax_pr.legend(loc="upper right")

    fig_pr
    return


@app.cell
def _(confusion_matrix, forest_predictions, logistic_predictions, pd, y_test):
    classification_results = []

    for model_name, test_preds in [
        ("Logistic Regression", logistic_predictions),
        ("Random Forest", forest_predictions)
    ]:
        tn, fp, fn, tp = confusion_matrix(
            y_test, test_preds
        ).ravel()

        classification_results.append({
            "Model": model_name,
            "True Positives": tp,
            "False Positives": fp,
            "True Negatives": tn,
            "False Negatives": fn,
            "Sensitivity": tp / (tp + fn),
            "Specificity": tn / (tn + fp),
            "Precision (PPV)": tp / (tp + fp) if (tp + fp) > 0 else 0
        })

    classification_table = pd.DataFrame(classification_results)

    classification_table.round(4)
    return


@app.cell
def _(
    baseline_probs,
    forest_probabilities,
    logistic_probabilities,
    pd,
    y_test,
):
    from sklearn.metrics import brier_score_loss

    brier_results = pd.DataFrame({
        "Model": ["Baseline", "Logistic Regression", "Random Forest"],
        "Brier Score": [
            brier_score_loss(y_test, baseline_probs),
            brier_score_loss(y_test, logistic_probabilities),
            brier_score_loss(y_test, forest_probabilities)
        ]
    })

    brier_results.round(4)
    return


@app.cell
def _(forest_probabilities, logistic_probabilities, plt, y_test):
    from sklearn.calibration import CalibrationDisplay

    fig_cal, ax_cal = plt.subplots(figsize=(7, 5))

    CalibrationDisplay.from_predictions(
        y_test,
        logistic_probabilities,
        n_bins=10,
        strategy="quantile",
        name="Logistic Regression",
        ax=ax_cal
    )

    CalibrationDisplay.from_predictions(
        y_test,
        forest_probabilities,
        n_bins=10,
        strategy="quantile",
        name="Random Forest",
        ax=ax_cal
    )

    ax_cal.plot(
        [0, 1], [0, 1],
        "k--",
        label="Perfect Calibration"
    )

    ax_cal.set_title("Calibration Plot: 30-Day Readmission")
    ax_cal.legend()

    fig_cal
    return


@app.cell
def _(X_test, best_forest, pd, y_test):
    from sklearn.inspection import permutation_importance

    importance_result = permutation_importance(
        best_forest,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=3,
        random_state=42,
        n_jobs=1
    )

    importance_table = pd.DataFrame({
        "Predictor": X_test.columns,
        "Importance": importance_result.importances_mean,
        "Standard Deviation": importance_result.importances_std
    }).sort_values("Importance", ascending=False)

    importance_table.round(4)
    return (importance_table,)


@app.cell
def _(importance_table, plt):
    top_importance = importance_table.head(10).sort_values("Importance")

    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))

    ax_imp.barh(
        top_importance["Predictor"],
        top_importance["Importance"],
        xerr=top_importance["Standard Deviation"],
        capsize=3
    )

    ax_imp.set_xlabel("Decrease in AUROC When Predictor Is Shuffled")
    ax_imp.set_title("Random Forest: Top 10 Predictor Importances")

    fig_imp.tight_layout()
    fig_imp
    return


@app.cell
def _(X_test, forest_probabilities, np, y_test):


    # Find the highest-risk and lowest-risk test encounters
    high_risk_idx = np.argmax(forest_probabilities)
    low_risk_idx = np.argmin(forest_probabilities)

    # Display their clinical information and predictions
    case_indices = [high_risk_idx, low_risk_idx]

    case_table = X_test.iloc[case_indices].copy()

    case_table["Predicted Readmission Probability"] = (
        forest_probabilities[case_indices]
    )

    case_table["Actual Readmission"] = (
        y_test.iloc[case_indices].values
    )

    case_table.index = ["Highest Predicted Risk", "Lowest Predicted Risk"]

    case_table.T
    return (case_indices,)


@app.cell
def _(X_test, best_forest, case_indices, pd):
    case_explanation = X_test.iloc[case_indices].copy()

    original_risks = best_forest.predict_proba(
        case_explanation
    )[:, 1]

    # Set previous inpatient visits to zero for both cases
    modified_cases = case_explanation.copy()
    modified_cases["number_inpatient"] = 0

    modified_risks = best_forest.predict_proba(
        modified_cases
    )[:, 1]

    case_explanation_results = pd.DataFrame({
        "Case": ["Highest Predicted Risk", "Lowest Predicted Risk"],
        "Previous Inpatient Visits": case_explanation["number_inpatient"].values,
        "Original Risk (%)": original_risks * 100,
        "Risk With 0 Prior Inpatient Visits (%)": modified_risks * 100,
        "Change (Percentage Points)": (
            modified_risks - original_risks
        ) * 100
    })

    case_explanation_results.round(2)
    return


@app.cell
def _(
    X_test,
    best_forest,
    forest_predictions,
    forest_probabilities,
    np,
    y_test,
):
    # Find an incorrect prediction (false positive)
    false_positive_positions = np.where(
        (forest_predictions == 1) & (y_test.to_numpy() == 0)
    )[0]

    # Select the false positive with the highest predicted risk
    i = false_positive_positions[
        np.argmax(forest_probabilities[false_positive_positions])
    ]

    # Display the actual outcome and predicted risk
    print("INDIVIDUAL INCORRECT PREDICTION")
    print("--------------------------------")
    print("Actual 30-day readmission:", int(y_test.iloc[i]))
    print("Predicted 30-day readmission:", int(forest_predictions[i]))
    print("Predicted probability:", round(forest_probabilities[i] * 100, 2), "%")
    print("Age group:", X_test.iloc[i]["age"])
    print("Previous inpatient visits:", X_test.iloc[i]["number_inpatient"])
    print("Previous emergency visits:", X_test.iloc[i]["number_emergency"])

    # Examine how one predictor affects the model's prediction
    modified_case = X_test.iloc[[i]].copy()
    modified_case["number_inpatient"] = 0

    modified_probability = best_forest.predict_proba(modified_case)[0, 1]

    print(
        "Predicted probability with 0 previous inpatient visits:",
        round(modified_probability * 100, 2),
        "%"
    )
    return


@app.cell
def _(X_test, forest_probabilities, pd, roc_auc_score, y_test):
    age_results = []

    for age_group in sorted(X_test["age"].dropna().unique()):
        mask = X_test["age"] == age_group

        actual = y_test[mask]
        predicted = forest_probabilities[mask.to_numpy()]

        if len(actual) > 0 and actual.nunique() == 2:
            age_results.append({
                "Age Group": age_group,
                "Encounters": len(actual),
                "Readmission Rate (%)": actual.mean() * 100,
                "AUROC": roc_auc_score(actual, predicted)
            })

    age_subgroup_results = pd.DataFrame(age_results)

    age_subgroup_results.round(4)
    return (age_subgroup_results,)


@app.cell
def _(age_subgroup_results, plt):
    fig_age, ax_age = plt.subplots(figsize=(9, 5))

    ax_age.bar(
        age_subgroup_results["Age Group"],
        age_subgroup_results["AUROC"]
    )

    ax_age.axhline(
        y=0.5,
        color="black",
        linestyle="--",
        label="Random Guessing"
    )

    ax_age.set_title("Random Forest AUROC by Age Group")
    ax_age.set_xlabel("Age Group")
    ax_age.set_ylabel("AUROC")
    ax_age.set_ylim(0, 1)

    ax_age.tick_params(axis="x", rotation=45)
    ax_age.legend()

    fig_age.tight_layout()
    fig_age
    return


@app.cell
def _(X_test, forest_probabilities, pd, roc_auc_score, y_test):
    # Restrict the test data to encounters aged 40 and older


    older_age_groups = [
        "[40-50)", "[50-60)", "[60-70)",
        "[70-80)", "[80-90)", "[90-100)"
    ]

    older_mask = X_test["age"].isin(older_age_groups)

    older_actual = y_test[older_mask]
    older_probabilities = forest_probabilities[older_mask.to_numpy()]

    older_results = pd.DataFrame({
        "Population": ["All Test Encounters", "Age 40 and Older"],
        "Encounters": [len(y_test), len(older_actual)],
        "Readmission Rate (%)": [
            y_test.mean() * 100,
            older_actual.mean() * 100
        ],
        "AUROC": [
            roc_auc_score(y_test, forest_probabilities),
            roc_auc_score(older_actual, older_probabilities)
        ]
    })

    older_results.round(4)
    return


@app.cell
def _(forest_probabilities, np, pd, y_test):
    error_results = pd.DataFrame({
        "Actual Readmission": y_test.values,
        "Predicted Probability": forest_probabilities
    })

    error_results["Predicted Readmission"] = (
        error_results["Predicted Probability"] >= 0.12
    ).astype(int)

    error_results["Prediction Type"] = np.select(
        [
            (error_results["Actual Readmission"] == 1) &
            (error_results["Predicted Readmission"] == 1),

             (error_results["Actual Readmission"] == 0) &
             (error_results["Predicted Readmission"] == 0),

             (error_results["Actual Readmission"] == 0) &
             (error_results["Predicted Readmission"] == 1),

             (error_results["Actual Readmission"] == 1) &
             (error_results["Predicted Readmission"] == 0)
        ],
        [
             "True Positive",
             "True Negative",
             "False Positive",
             "False Negative"
        ],
        default="Unknown"
    )

    error_results["Prediction Type"].value_counts()
    return (error_results,)


@app.cell
def _(error_results, y_test):
    error_summary = (
        error_results["Prediction Type"]
        .value_counts()
        .rename_axis("Prediction Type")
        .reset_index(name="Encounters")
    )

    error_summary["Percentage of Test Encounters"] = (
        error_summary["Encounters"] / len(y_test) * 100
    )

    error_summary.round(2)
    return


@app.cell
def _(predictors):
    # Sensitivity analysis: remove previous inpatient visits

    reduced_predictors = [
        p for p in predictors if p != "number_inpatient"
    ]

    print("Original predictors:", len(predictors))
    print("Reduced predictors:", len(reduced_predictors))
    print("Removed predictor: number_inpatient")
    return (reduced_predictors,)


@app.cell
def _(
    ColumnTransformer,
    Pipeline,
    RandomForestClassifier,
    categorical_features,
    categorical_pipeline,
    numeric_features,
    numeric_pipeline,
):
    # Create preprocessing without number_inpatient

    reduced_numeric_features = [


        p for p in numeric_features if p != "number_inpatient"
    ]

    reduced_preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, reduced_numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    # Use the same random forest settings as the original model
    reduced_forest = Pipeline([
        ("preprocessing", reduced_preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=30,
            random_state=42,
            n_jobs=1
        ))
    ])

    print("Reduced random forest created successfully!")
    return (reduced_forest,)


@app.cell
def _(
    GroupKFold,
    X_train,
    reduced_forest,
    reduced_predictors,
    train_df,
    y_train,
):
    # Compare the reduced model using patient-level cross-validation

    from sklearn.model_selection import cross_val_score

    reduced_cv_scores = cross_val_score(
        reduced_forest,
        X_train[reduced_predictors],
        y_train,
        groups=train_df["patient_nbr"],
        cv=GroupKFold(n_splits=3),
        scoring="roc_auc",
        n_jobs=1
    )

    print("Reduced model mean CV AUROC:",
          round(reduced_cv_scores.mean(), 4))

    print("Reduced model CV standard deviation:",
          round(reduced_cv_scores.std(), 4))
    return (reduced_cv_scores,)


@app.cell
def _(reduced_cv_scores):
    # Sensitivity analysis comparison

    original_auroc = 0.6402
    reduced_auroc = reduced_cv_scores.mean()

    print("SENSITIVITY ANALYSIS")
    print("------------------------------")
    print(f"Original model CV AUROC: {original_auroc:.4f}")
    print(f"Reduced model CV AUROC: {reduced_auroc:.4f}")
    print(f"AUROC difference:        {original_auroc - reduced_auroc:.4f}")

    print(
        "\nRemoving number_inpatient reduced model discrimination, "
        "suggesting that previous inpatient utilization is an "
        "important predictor of 30-day readmission."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final Project Summary

    ### Research Question

    Among hospitalized patients with diabetes, how accurately can clinical information available at discharge predict hospital readmission within 30 days?

    ### Dataset and Methods

    We analyzed the Diabetes 130-US Hospitals dataset, containing 101,766 hospital encounters. After excluding encounters with discharge dispositions indicating death or hospice care, 99,343 encounters from 69,990 unique patients remained eligible. We used 14 clinical and demographic predictors available at discharge to develop logistic regression and random forest models. Patients were separated between training and testing to prevent information leakage. Hyperparameter tuning was performed using cross-validation on the training data.

    ### Main Findings

    The random forest slightly outperformed logistic regression on the held-out test set, achieving an AUROC of 0.6349 compared with 0.6306. Both models performed better than the baseline AUROC of 0.5000. The random forest achieved an AUPRC of 0.1957 and a Brier score of 0.0965. At the selected 12% classification threshold, it identified approximately 50% of recorded 30-day readmissions, although many encounters were incorrectly flagged.

    Previous inpatient hospitalizations were the most influential predictor. Age subgroup analysis suggested differences in model performance across age groups. An analysis restricted to patients aged 40 and older produced a test AUROC of 0.6240. An additional sensitivity analysis using patient-level cross-validation on the training data showed that removing previous inpatient hospitalizations (`number_inpatient`) reduced the random forest's AUROC from 0.6402 to 0.5953, highlighting the importance of prior hospital utilization in predicting 30-day readmission. An individual prediction analysis identified a false positive with a predicted 30-day readmission risk of 38.26%, despite no recorded readmission within 30 days. Setting previous inpatient visits to zero reduced the model's predicted risk to 17.05%, illustrating the influence of prior hospital utilization without establishing causation.

    ### Clinical Interpretation and Limitations

    The models demonstrated modest predictive ability, suggesting that routinely collected clinical information contains some useful information about readmission risk. However, the relatively low sensitivity and positive predictive value limit their clinical usefulness. The dataset is historical, contains substantial missing information in several variables, and may not capture readmissions outside participating hospitals. The results should therefore be interpreted as an exploratory prediction study.




    ### Conclusion

    Machine learning models can identify some patterns associated with 30-day hospital readmission among patients with diabetes, but the predictive performance achieved in this project is insufficient to support independent clinical decision-making. Further validation, improved clinical data, and additional research would be necessary before considering clinical implementation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
