"""
=======================================================================================
Python script to demonstrate neural network fine tuning and hyperparameter optimisation
=======================================================================================

This program demonstrates hyperparameter optimisation and neural network fine tuning using a synthetic Adult income dataset.

- Synthetic dataset generation
- Data preprocessing
- Feature encoding
- Feature scaling
- Baseline model training
- Hyperparameter optimisation using Optuna
- Fine tuning
- Model evaluation
- Model persistence

Requirements:
    pandas numpy, scikit-learn matplotlib seaborn optuna joblib
"""

# -----------------------------------------------------------------------
# 0. Import required modules
# -----------------------------------------------------------------------
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Constants
# -----------------------------------------------------------------------------
MODEL_FILE = Path("files/adult_income_model.plk")
RANDOM_STATE: int = 42
N_SAMPLES = 5000


# -----------------------------------------------------------------------------
# 2. Program flow functions
# -----------------------------------------------------------------------------
# function to display section headings in the program
def print_heading(
    title: str,
) -> None:
    """
    Print a formatted heading.

    Parameters
    ----------
    title:
        Heading title.
    """

    print()

    print("=" * 60)
    print(title)
    print("=" * 60)


# function to generate the synthetic dataset
def generate_dataset(samples: int = N_SAMPLES) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    age = rng.integers(low=18, high=80, size=samples)
    education = rng.choice(
        [
            "High School",
            "Diploma",
            "Bachelors",
            "Masters",
            "Doctorate",
        ],
        samples,
        p=[0.35, 0.20, 0.25, 0.15, 0.05],
    )

    occupation = rng.choice(
        [
            "Admin",
            "Sales",
            "Professional",
            "Technical",
            "Service",
            "Management",
        ],
        samples,
    )

    marital_status = rng.choice(
        [
            "Single",
            "Married",
            "Divorced",
        ],
        samples,
        p=[0.40, 0.45, 0.15],
    )
    hours_per_week = rng.integers(low=20, high=70, size=samples)
    capital_gain = rng.integers(low=0, high=20000, size=samples)
    years_experience = np.maximum(
        age - 18 + rng.normal(0, 4, samples),
        0,
    ).astype(int)

    education_score = {
        "High School": 0,
        "Diploma": 1,
        "Bachelors": 2,
        "Masters": 3,
        "Doctorate": 4,
    }

    occupation_score = {
        "Admin": 0,
        "Sales": 1,
        "Service": 0,
        "Technical": 2,
        "Professional": 3,
        "Management": 4,
    }

    salary_score = (
        age * 0.25
        + hours_per_week * 0.40
        + years_experience * 0.50
        + capital_gain * 0.0004
        + np.vectorize(education_score.get)(education) * 5
        + np.vectorize(occupation_score.get)(occupation) * 4
        + rng.normal(0, 4, samples)
    )

    income = np.where(
        salary_score > 42,
        ">50K",
        "<=50K",
    )

    dataset = pd.DataFrame(
        {
            "age": age,
            "education": education,
            "occupation": occupation,
            "marital_status": marital_status,
            "hours_per_week": hours_per_week,
            "capital_gain": capital_gain,
            "years_experience": years_experience,
            "income": income,
        }
    )
    return dataset


# separate the features and target
def split_features_target(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    x = dataset.drop(columns="income")
    y = dataset["income"]
    return x, y


# function to create the preprocessing pipeline
def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    categorical = x.select_dtypes(include=["object"]).columns.tolist()
    numerical = x.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    transformer = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numerical),
            ("categorical", categorical_pipeline, categorical),
        ]
    )

    return transformer


# build the ml pipeline
def create_pipeline(
    preprocessor: ColumnTransformer, **model_parameters: Dict
) -> Pipeline:
    classifier = GradientBoostingClassifier(
        **model_parameters, random_state=RANDOM_STATE
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


# evaulate the trained model
def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> float:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    roc = roc_auc_score(y_test, probabilities)

    print("\nClassification report")
    print(classification_report(y_test, predictions))

    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC AUC:{roc:.3f}")
    return accuracy


# display the confusion matrix
def plot_confusion_matrix(
    model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
) -> None:
    ConfusionMatrixDisplay.from_estimator(
        model, x_test, y_test, cmap="viridis"
    )  # cmap= "blues"
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# display the roc curve
def plot_roc_curve(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    RocCurveDisplay.from_estimator(model, x_test, y_test)  # cmap= "blues"
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.show()


# plot the 10 most important feautures
def plot_feature_importance(model: Pipeline) -> None:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    imortance = classifier.feature_importances_

    importance_frame = (
        pd.DataFrame({"Feature": feature_names, "Importance": imortance})
        .sort_values("Importance", ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_frame, x="Importance", y="Feature", palette="viridis")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()


# Function to plot baseline and tuned accuracies
def compare_models(baseline: float, tuned: float) -> None:

    comparison = pd.DataFrame(
        {
            "Model": ["Baseline", "Fine Tuned"],
            "Accuracy": [baseline, tuned],
        }
    )

    plt.figure(figsize=(7, 5))
    sns.barplot(data=comparison, x="Model", y="Accuracy", palette="Set2")

    plt.ylim(0.0, 1.0)
    plt.title("Model Accuracy Comparison")
    plt.tight_layout()
    plt.show()


def objective(
    trial: optuna.Trial,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> float:
    """
    Objective function for Optuna.

    Parameters
    ----------
    trial:
        Current optimisation trial.

    x_train:
        Training features.

    y_train:
        Training labels.

    preprocessor:
        Data preprocessing pipeline.

    Returns
    -------
    float
        Mean cross-validation accuracy.
    """

    parameters = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            50,
            300,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.30,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            2,
            5,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0,
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split",
            2,
            10,
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            1,
            5,
        ),
    }

    pipeline = create_pipeline(
        preprocessor,
        **parameters,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    return scores.mean()


def optimise_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> Dict:
    """
    Perform hyperparameter optimisation.

    Parameters
    ----------
    x_train:
        Training features.

    y_train:
        Training labels.

    preprocessor:
        Data preprocessing pipeline.

    Returns
    -------
    dict
        Best parameter values.
    """

    study = optuna.create_study(
        direction="maximize",
        study_name="Gradient Boosting Optimisation",
    )

    study.optimize(
        lambda trial: objective(
            trial,
            x_train,
            y_train,
            preprocessor,
        ),
        n_trials=30,
        show_progress_bar=True,
    )

    print("\nBest Parameters\n")

    for key, value in study.best_params.items():
        print(f"{key:<20} {value}")

    print()

    return study.best_params


def train_baseline(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """
    Train the baseline model.

    Parameters
    ----------
    x_train:
        Training features.

    y_train:
        Training labels.

    preprocessor:
        Data preprocessing pipeline.

    Returns
    -------
    Pipeline
        Trained baseline model.
    """

    baseline = create_pipeline(
        preprocessor,
    )

    baseline.fit(
        x_train,
        y_train,
    )

    return baseline


def train_fine_tuned(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    parameters: Dict,
) -> Pipeline:
    """
    Train the optimised model.

    Parameters
    ----------
    x_train:
        Training features.

    y_train:
        Training labels.

    preprocessor:
        Data preprocessing pipeline.

    parameters:
        Best parameters from Optuna.

    Returns
    -------
    Pipeline
        Fine-tuned model.
    """

    tuned = create_pipeline(
        preprocessor,
        **parameters,
    )

    tuned.fit(
        x_train,
        y_train,
    )

    return tuned


def save_model(model: Pipeline, filename: Path = MODEL_FILE) -> None:
    joblib.dump(model, filename)
    print(f"\nModel saved to {filename.resolve()}")


def display_dataset_information(dataset: pd.DataFrame) -> None:

    print("=" * 60)
    print("Dataset Overview")
    print("=" * 60)

    print(f"Number of samples: {len(dataset):,}")
    print(f"Number of features: {dataset.shape[1] - 1}")
    print(f"\nFirst five records:\n{dataset.head()}")
    print("=" * 60)

    print(f"Target distribution:\n{dataset['income'].value_counts()}")

    print(f"Missing Values:{dataset.isna().sum()}")


def perform_cross_validation(
    pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series
) -> None:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_score(
        pipeline, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=1
    )
    print("\nCross Validation Accuracy")

    for (
        index,
        score,
    ) in enumerate(scores, start=1):
        print(f"Fold {index}: {score:.3f}")

    print(f"\nMean Accuracy:{scores.mean():.3f}")


def prepare_data() -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer
]:
    dataset = generate_dataset()

    display_dataset_information(dataset)

    x, y = split_features_target(dataset)

    preprocessor = build_preprocessor(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=RANDOM_STATE, test_size=0.2, stratify=y
    )

    return x_train, x_test, y_train, y_test, preprocessor


# -----------------------------------------------------------------------------
# 3. Main Function Exectution
# -----------------------------------------------------------------------------


def main() -> None:
    print_heading("Generating Synthetic Aduly Income Dataset")
    x_train, x_test, y_train, y_test, preprocessor = prepare_data()
    print_heading("Training Baseline Model")
    baseline_model = train_baseline(x_train, y_train, preprocessor)
    perform_cross_validation(baseline_model, x_train, y_train)
    print_heading("Baseline Model Performance")
    baseline_accuracy = evaluate_model(baseline_model, x_test, y_test)
    print_heading("Hyperparameter Optimisation")
    best_parameters = optimise_model(x_train, y_train, preprocessor)
    print_heading("Training Fine Tuned Model")
    tuned_model = train_fine_tuned(x_train, y_train, preprocessor, best_parameters)
    print_heading("Fine-Tuned Model Performance")
    tuned_accuracy = evaluate_model(tuned_model, x_test, y_test)
    print_heading("Model Comparison")
    improvement = tuned_accuracy - baseline_accuracy
    print(f"Baseline Accuracy: {baseline_accuracy:.3f}")
    print(f"Fine-Tuned Accuracy: {tuned_accuracy:.3f}")
    print(f"Improvement: {improvement:.3f}")
    print_heading("Generating visualisations")
    plot_confusion_matrix(tuned_model, x_test, y_test)
    plot_roc_curve(tuned_model, x_test, y_test)
    plot_feature_importance(tuned_model)
    compare_models(baseline_accuracy, tuned_accuracy)
    print_heading("Saving Model")
    save_model(tuned_model)
    print_heading("Program Complete")
    print("The optimised model has been successfully trained, evaluated and saved.")


# -----------------------------------------------------------------------------
# 4. Run the script
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
