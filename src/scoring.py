"""
The CreditScorer wraps the trained model and turns a raw default probability
into a human-facing credit score on the CIBIL-style 300-900 scale used by
Indian credit bureaus (CIBIL, Experian, Equifax, CRIF High Mark), a risk
grade, and a plain-English explanation of the top factors driving the score.

Note: the 300-900 range and band cutoffs mirror how CIBIL scores are
commonly described publicly, but this is an independent model trained on
synthetic data — it does not reproduce or query actual CIBIL/bureau scores.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import align_columns, build_feature_frame

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

SCORE_MIN = 300
SCORE_MAX = 900

# Bands modeled on how CIBIL scores are commonly communicated to consumers.
GRADE_BANDS = [
    (750, SCORE_MAX, "A", "Excellent"),
    (700, 750, "B", "Good"),
    (650, 700, "C", "Fair"),
    (600, 650, "D", "Poor"),
    (SCORE_MIN, 600, "E", "Very Poor"),
]


def probability_to_score(prob_default: float) -> int:
    """Maps P(default) in [0,1] to a 300-900 score using a log-odds
    transform, so the score behaves like real bureau scores: risk changes
    near the middle move the score more than at the extremes.

    Calibrated against the empirical distribution of this model's predicted
    probabilities on the training population: median-risk applicants
    (~17% predicted default probability) land around 700, and the mapping
    stretches out toward 900 for very low risk and toward 300 for very
    high risk."""
    prob_default = np.clip(prob_default, 1e-6, 1 - 1e-6)
    odds_good = (1 - prob_default) / prob_default
    log_odds = np.log(odds_good)
    score = 580 + 75 * log_odds
    score = int(np.clip(score, SCORE_MIN, SCORE_MAX))
    return score


def score_to_grade(score: int):
    for lo, hi, grade, label in GRADE_BANDS:
        if lo <= score <= hi:
            return grade, label
    return "E", "Very Poor"


class CreditScorer:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model = joblib.load(os.path.join(model_dir, "model.joblib"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        with open(os.path.join(model_dir, "metadata.json")) as f:
            self.metadata = json.load(f)
        self.feature_columns = self.metadata["feature_columns"]
        self.needs_scaling = self.metadata["needs_scaling"]
        self.importance = {
            row["feature"]: row["importance"]
            for row in self.metadata["feature_importance"]
        }

    def _prepare(self, applicant: dict) -> pd.DataFrame:
        raw_df = pd.DataFrame([applicant])
        feat_df = build_feature_frame(raw_df)
        feat_df = align_columns(feat_df, self.feature_columns)
        return feat_df

    def score(self, applicant: dict) -> dict:
        feat_df = self._prepare(applicant)
        X = self.scaler.transform(feat_df) if self.needs_scaling else feat_df.values

        prob_default = float(self.model.predict_proba(X)[0, 1])
        credit_score = probability_to_score(prob_default)
        grade, label = score_to_grade(credit_score)

        # Always compute explanation on STANDARDIZED values (z-scores), so
        # features with naturally large raw magnitudes (like income) don't
        # dominate the ranking just because of their scale.
        X_std = self.scaler.transform(feat_df)
        std_row = pd.Series(X_std[0], index=self.feature_columns)
        explanation = self._explain(feat_df, std_row)

        return {
            "credit_score": credit_score,
            "score_range": f"{SCORE_MIN}-{SCORE_MAX}",
            "risk_grade": grade,
            "grade_label": label,
            "probability_of_default": round(prob_default, 4),
            "recommendation": self._recommend(grade),
            "top_factors": explanation,
        }

    def _recommend(self, grade: str) -> str:
        mapping = {
            "A": "Approve — excellent credit profile, eligible for best interest rates.",
            "B": "Approve — good credit profile, standard-to-preferred rates.",
            "C": "Approve with conditions — acceptable risk; standard rates, possibly with a co-applicant or slightly reduced eligible amount.",
            "D": "Manual review — weak profile. Consider secured lending (e.g. gold loan), a guarantor, or reduced loan amount.",
            "E": "Likely decline — high probability of default; not eligible for unsecured lending.",
        }
        return mapping.get(grade, "Manual review recommended.")

    def _explain(self, feat_df: pd.DataFrame, std_row: pd.Series, top_n: int = 5):
        """A simple, transparent explanation: for each feature, combine the
        model's global importance with how far this applicant's STANDARDIZED
        value sits from average, in the risky direction, to produce a ranked
        list of factors most likely driving this applicant's score. Using
        z-scores (rather than raw values) keeps large-magnitude features
        like income from automatically dominating the ranking. This is
        intentionally simple (not SHAP) so it's dependency-light and easy
        to audit."""
        raw_row = feat_df.iloc[0]

        # Directionality: for these features, HIGHER = riskier.
        risk_increases_with = {
            "debt_to_income", "credit_utilization", "num_late_payments_12m",
            "num_derogatory_marks", "previous_defaults", "loan_to_income",
            "late_payment_rate", "emi_to_monthly_income", "num_dependents",
        }
        # For these, HIGHER = safer.
        risk_decreases_with = {
            "annual_income", "credit_history_length_years",
            "employment_length_years",
        }

        contributions = []
        for feature, importance in self.importance.items():
            if feature not in std_row.index:
                continue
            z = std_row[feature]  # standardized value (z-score)
            raw_value = raw_row[feature]

            if feature in risk_increases_with:
                risk_signal = z
                direction = "increases risk" if z > 0 else "reduces risk"
            elif feature in risk_decreases_with:
                risk_signal = -z
                direction = "reduces risk" if z > 0 else "increases risk"
            else:
                # dummy / categorical columns: presence (1) => that category applies
                risk_signal = z
                direction = "applies" if raw_value > 0 else "not applicable"

            score_contribution = importance * abs(risk_signal)
            if score_contribution > 0:
                contributions.append(
                    {
                        "feature": feature,
                        "value": float(raw_value),
                        "direction": direction,
                        "influence": round(float(score_contribution), 4),
                    }
                )

        contributions.sort(key=lambda x: x["influence"], reverse=True)
        contributions = [c for c in contributions if c["direction"] != "not applicable"]
        return contributions[:top_n]


if __name__ == "__main__":
    scorer = CreditScorer()
    sample_applicant = {
        "age": 32,
        "annual_income": 900000,
        "city_tier": "Metro",
        "employment_type": "Salaried",
        "employment_length_years": 6,
        "credit_history_length_years": 8,
        "num_credit_lines": 4,
        "existing_emi_monthly": 12000,
        "debt_to_income": 0.18,
        "credit_utilization": 0.30,
        "num_late_payments_12m": 0,
        "num_derogatory_marks": 0,
        "previous_defaults": 0,
        "num_dependents": 1,
        "loan_amount": 500000,
        "loan_purpose": "personal_loan",
        "residence_type": "Owned",
    }
    result = scorer.score(sample_applicant)
    print(json.dumps(result, indent=2))
