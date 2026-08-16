"""
Basic tests for the Credit Scoring Engine (India edition).

Run (from project root, after training the model):
    pytest tests/ -v
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.scoring import CreditScorer, probability_to_score, score_to_grade

GOOD_APPLICANT = {
    "age": 40,
    "annual_income": 1800000,
    "city_tier": "Metro",
    "employment_type": "Salaried",
    "employment_length_years": 15,
    "credit_history_length_years": 18,
    "num_credit_lines": 3,
    "existing_emi_monthly": 10000,
    "debt_to_income": 0.08,
    "credit_utilization": 0.10,
    "num_late_payments_12m": 0,
    "num_derogatory_marks": 0,
    "previous_defaults": 0,
    "num_dependents": 2,
    "loan_amount": 200000,
    "loan_purpose": "home_loan",
    "residence_type": "Owned",
}

RISKY_APPLICANT = {
    "age": 24,
    "annual_income": 240000,
    "city_tier": "Tier-3",
    "employment_type": "Self-Employed",
    "employment_length_years": 1,
    "credit_history_length_years": 1,
    "num_credit_lines": 8,
    "existing_emi_monthly": 15000,
    "debt_to_income": 0.75,
    "credit_utilization": 0.92,
    "num_late_payments_12m": 6,
    "num_derogatory_marks": 3,
    "previous_defaults": 1,
    "num_dependents": 3,
    "loan_amount": 400000,
    "loan_purpose": "business_loan",
    "residence_type": "Rented",
}


@pytest.fixture(scope="module")
def scorer():
    return CreditScorer()


def test_probability_to_score_bounds():
    assert probability_to_score(0.0) <= 900
    assert probability_to_score(1.0) >= 300
    assert probability_to_score(0.001) > probability_to_score(0.5) > probability_to_score(0.9)


def test_score_to_grade_bands():
    grade, label = score_to_grade(820)
    assert grade == "A"
    grade, label = score_to_grade(350)
    assert grade == "E"


def test_good_applicant_scores_higher_than_risky(scorer):
    good_result = scorer.score(GOOD_APPLICANT)
    risky_result = scorer.score(RISKY_APPLICANT)

    assert good_result["credit_score"] > risky_result["credit_score"]
    assert good_result["probability_of_default"] < risky_result["probability_of_default"]


def test_score_within_valid_range(scorer):
    for applicant in (GOOD_APPLICANT, RISKY_APPLICANT):
        result = scorer.score(applicant)
        assert 300 <= result["credit_score"] <= 900
        assert result["risk_grade"] in {"A", "B", "C", "D", "E"}
        assert 0 <= result["probability_of_default"] <= 1


def test_explanation_returns_factors(scorer):
    result = scorer.score(RISKY_APPLICANT)
    assert len(result["top_factors"]) > 0
    for factor in result["top_factors"]:
        assert "feature" in factor
        assert "direction" in factor
