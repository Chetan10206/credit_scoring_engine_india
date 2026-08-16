"""
Preprocessing utilities shared between training and inference.
Keeps feature engineering logic in ONE place so train/serve skew can't happen.
"""

import pandas as pd

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_length_years",
    "credit_history_length_years",
    "num_credit_lines",
    "existing_emi_monthly",
    "debt_to_income",
    "credit_utilization",
    "num_late_payments_12m",
    "num_derogatory_marks",
    "previous_defaults",
    "num_dependents",
    "loan_amount",
]

CATEGORICAL_FEATURES = ["city_tier", "employment_type", "loan_purpose", "residence_type"]

TARGET = "default"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Applied identically at train and inference time."""
    df = df.copy()
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"].clip(lower=1)
    df["late_payment_rate"] = df["num_late_payments_12m"] / (df["num_credit_lines"] + 1)
    df["emi_to_monthly_income"] = (df["existing_emi_monthly"] * 12) / df["annual_income"].clip(lower=1)
    return df


ENGINEERED_FEATURES = ["loan_to_income", "late_payment_rate", "emi_to_monthly_income"]

ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_FEATURES


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: engineer features, one-hot encode categoricals."""
    df = engineer_features(df)
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES)
    return df


def align_columns(df: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    """Ensure a dataframe (e.g. a single new applicant) has exactly the columns
    the model was trained on, in the same order, filling missing dummy columns
    with 0."""
    df = df.reindex(columns=expected_columns, fill_value=0)
    return df
