"""
FastAPI service exposing the Credit Scoring Engine (India edition).

Run (from the project root):
    uvicorn api.app:app --reload --port 8000

Then open:
    http://localhost:8000/          -> simple demo form
    http://localhost:8000/docs      -> interactive Swagger API docs
"""

import os
import sys
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.scoring import CreditScorer

app = FastAPI(
    title="Credit Scoring Engine API (India)",
    description="Scores a loan applicant and returns a CIBIL-style credit score "
    "(300-900), risk grade, default probability, and top contributing factors.",
    version="1.0.0",
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_scorer = None


def get_scorer() -> CreditScorer:
    global _scorer
    if _scorer is None:
        try:
            _scorer = CreditScorer()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run `python src/train_model.py` first.",
            )
    return _scorer


class Applicant(BaseModel):
    age: int = Field(..., ge=21, le=70, example=32)
    annual_income: float = Field(..., gt=0, description="Annual income in INR", example=900000)
    city_tier: Literal["Metro", "Tier-2", "Tier-3"] = Field(..., example="Metro")
    employment_type: Literal["Salaried", "Self-Employed", "Business Owner"] = Field(..., example="Salaried")
    employment_length_years: int = Field(..., ge=0, le=50, example=6)
    credit_history_length_years: int = Field(..., ge=0, le=50, example=8)
    num_credit_lines: int = Field(..., ge=0, le=30, description="credit cards + loans", example=4)
    existing_emi_monthly: float = Field(..., ge=0, description="Existing monthly EMI outflow in INR", example=12000)
    debt_to_income: float = Field(..., ge=0, le=2, description="fraction, e.g. 0.18 = 18%", example=0.18)
    credit_utilization: float = Field(..., ge=0, le=2, description="fraction of credit card limit used", example=0.30)
    num_late_payments_12m: int = Field(..., ge=0, le=50, example=0)
    num_derogatory_marks: int = Field(..., ge=0, le=20, example=0)
    previous_defaults: Literal[0, 1] = Field(..., example=0)
    num_dependents: int = Field(..., ge=0, le=10, example=1)
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in INR", example=500000)
    loan_purpose: Literal[
        "home_loan", "personal_loan", "vehicle_loan", "education_loan",
        "gold_loan", "business_loan", "credit_card", "consumer_durable",
    ] = Field(..., example="personal_loan")
    residence_type: Literal["Owned", "Rented", "Family-Owned"] = Field(..., example="Owned")


class ScoreFactor(BaseModel):
    feature: str
    value: float
    direction: str
    influence: float


class ScoreResponse(BaseModel):
    credit_score: int
    score_range: str
    risk_grade: str
    grade_label: str
    probability_of_default: float
    recommendation: str
    top_factors: list[ScoreFactor]


@app.get("/", response_class=HTMLResponse)
def demo_form():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            return f.read()
    return "<h1>Credit Scoring Engine (India)</h1><p>See <a href='/docs'>/docs</a> for the API.</p>"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score_applicant(applicant: Applicant):
    scorer = get_scorer()
    result = scorer.score(applicant.model_dump())
    return result


@app.get("/model-info")
def model_info():
    scorer = get_scorer()
    return {
        "model_name": scorer.metadata["model_name"],
        "metrics": scorer.metadata["metrics"],
        "top_global_features": scorer.metadata["feature_importance"][:10],
    }
