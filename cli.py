"""
Command-line interface: score a single applicant from a JSON file, or from
a built-in demo example, without needing to run the API server.

Usage:
    python cli.py --json applicant.json
    python cli.py --demo          # scores a built-in example applicant
"""

import argparse
import json

from src.scoring import CreditScorer

DEMO_APPLICANT = {
    "age": 29,
    "annual_income": 650000,
    "city_tier": "Tier-2",
    "employment_type": "Salaried",
    "employment_length_years": 3,
    "credit_history_length_years": 5,
    "num_credit_lines": 3,
    "existing_emi_monthly": 8000,
    "debt_to_income": 0.22,
    "credit_utilization": 0.45,
    "num_late_payments_12m": 2,
    "num_derogatory_marks": 0,
    "previous_defaults": 0,
    "num_dependents": 0,
    "loan_amount": 300000,
    "loan_purpose": "vehicle_loan",
    "residence_type": "Rented",
}


def main():
    parser = argparse.ArgumentParser(description="Score a credit applicant from the command line.")
    parser.add_argument("--json", type=str, help="Path to a JSON file with applicant data.")
    parser.add_argument("--demo", action="store_true", help="Score a built-in demo applicant.")
    args = parser.parse_args()

    if args.demo:
        applicant = DEMO_APPLICANT
    elif args.json:
        with open(args.json) as f:
            applicant = json.load(f)
    else:
        parser.error("Provide --json <file> or --demo")
        return

    scorer = CreditScorer()
    result = scorer.score(applicant)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
