"""
Generates a synthetic dataset of Indian loan/credit applicants for the
Credit Scoring Engine (India edition).

This is SYNTHETIC data with realistic Indian-market distributions and
correlations (income in INR, employment type, city tier, existing EMI
burden, etc.) — it is NOT real bank or CIBIL data. Real bank data is not
publicly shareable for privacy/regulatory reasons. If you want to train on
real data instead, see the README for links to public Indian credit/loan
datasets on Kaggle (e.g. "Leading Indian Bank & CIBIL Real-World Dataset",
"Loan Prediction India", "Credit/Loan Dataset - Rural India"), and swap the
loading step in src/train_model.py to read that CSV, mapping its columns to
the feature names below.

Run:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 15000  # number of synthetic applicants


def generate_dataset(n=N):
    age = np.random.normal(35, 10, n).clip(21, 65).astype(int)

    city_tier = np.random.choice(
        ["Metro", "Tier-2", "Tier-3"], size=n, p=[0.35, 0.40, 0.25]
    )
    # Base monthly income (INR) varies a lot by city tier
    tier_income_mult = np.select(
        [city_tier == "Metro", city_tier == "Tier-2", city_tier == "Tier-3"],
        [1.5, 1.0, 0.7],
    )
    monthly_income = (
        np.random.lognormal(mean=10.0, sigma=0.55, size=n) * tier_income_mult
    ).clip(12000, 500000)
    annual_income = (monthly_income * 12).round(0)

    employment_type = np.random.choice(
        ["Salaried", "Self-Employed", "Business Owner"], size=n, p=[0.62, 0.23, 0.15]
    )

    employment_length_years = np.random.exponential(scale=5, size=n).clip(0, 35).astype(int)

    credit_history_length_years = (
        age - 21 - np.random.exponential(scale=4, size=n)
    ).clip(0, 40).astype(int)

    num_credit_lines = np.random.poisson(lam=4, size=n).clip(0, 20)  # credit cards + loans

    existing_emi_monthly = (
        np.random.beta(2, 6, n) * monthly_income * 0.9
    ).round(0)  # existing EMI outflow

    debt_to_income = (existing_emi_monthly / monthly_income.clip(min=1)).clip(0, 1.5)

    credit_utilization = np.random.beta(2, 3, n)  # fraction of credit card limit used

    num_late_payments_12m = np.random.poisson(
        lam=(credit_utilization * 3 + debt_to_income * 2), size=n
    ).clip(0, 15)

    num_derogatory_marks = np.random.poisson(lam=0.25, size=n).clip(0, 8)

    previous_defaults = np.random.binomial(1, p=0.07, size=n)

    num_dependents = np.random.poisson(lam=1.6, size=n).clip(0, 6)

    loan_amount = np.random.lognormal(mean=12.2, sigma=0.7, size=n).clip(20000, 5000000)

    loan_purpose = np.random.choice(
        ["home_loan", "personal_loan", "vehicle_loan", "education_loan",
         "gold_loan", "business_loan", "credit_card", "consumer_durable"],
        size=n,
        p=[0.22, 0.24, 0.14, 0.10, 0.08, 0.10, 0.08, 0.04],
    )

    residence_type = np.random.choice(
        ["Owned", "Rented", "Family-Owned"], size=n, p=[0.35, 0.40, 0.25]
    )

    # ---- Latent "true" default risk (logit) built from the features above ----
    logit = (
        -3.0
        + 2.4 * debt_to_income
        + 1.7 * credit_utilization
        + 0.32 * num_late_payments_12m
        + 0.5 * num_derogatory_marks
        + 1.8 * previous_defaults
        - 0.028 * credit_history_length_years
        - 0.0000025 * annual_income
        - 0.018 * employment_length_years
        + 0.35 * (loan_amount / annual_income.clip(min=1))
        + 0.08 * num_dependents
        + np.where(employment_type == "Self-Employed", 0.2, 0.0)
        + np.where(employment_type == "Business Owner", 0.15, 0.0)
        + np.where(residence_type == "Rented", 0.15, 0.0)
        + np.where(loan_purpose == "business_loan", 0.3, 0.0)
        + np.where(loan_purpose == "gold_loan", -0.15, 0.0)  # secured, lower risk
        + np.where(city_tier == "Tier-3", 0.1, 0.0)
        + np.random.normal(0, 0.4, n)  # unobserved noise
    )

    prob_default = 1 / (1 + np.exp(-logit))
    default = np.random.binomial(1, prob_default)

    df = pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income,
            "city_tier": city_tier,
            "employment_type": employment_type,
            "employment_length_years": employment_length_years,
            "credit_history_length_years": credit_history_length_years,
            "num_credit_lines": num_credit_lines,
            "existing_emi_monthly": existing_emi_monthly,
            "debt_to_income": debt_to_income.round(4),
            "credit_utilization": credit_utilization.round(4),
            "num_late_payments_12m": num_late_payments_12m,
            "num_derogatory_marks": num_derogatory_marks,
            "previous_defaults": previous_defaults,
            "num_dependents": num_dependents,
            "loan_amount": loan_amount.round(0),
            "loan_purpose": loan_purpose,
            "residence_type": residence_type,
            "default": default,  # target: 1 = defaulted / bad credit outcome
        }
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "data/credit_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(f"Default rate: {df['default'].mean():.2%}")
    print(df.head())
