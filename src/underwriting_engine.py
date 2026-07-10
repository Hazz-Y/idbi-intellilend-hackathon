"""
IntelliLend — Engine C: Prudent Underwriting Recommendation
============================================================
Combines Engine A (Lead Score) + Engine B (Income Estimate) into:
  1. Loan Readiness Score (composite, 0-100)
  2. Recommended eligible loan amount (based on FOIR caps)
  3. Risk flags for underwriter review
  4. Loan product-specific parameters

This is a human-in-the-loop engine — it recommends, not decides.
Final loan decisions are made by RMs / underwriters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Loan product configuration (FOIR = Fixed Obligations to Income Ratio)
# ---------------------------------------------------------------------------

LOAN_CONFIG = {
    "Personal Loan": {
        "max_foir": 0.50,       # 50% max of income
        "max_amount": 25_00_000,  # ₹25 Lakh
        "min_amount": 50_000,
        "interest_rate": 0.12,  # 12% p.a.
        "default_tenure_months": 60,
        "min_income": 20_000,
        "min_lead_score": 30,
    },
    "Home Loan": {
        "max_foir": 0.45,
        "max_amount": 1_00_00_000,  # ₹1 Crore
        "min_amount": 5_00_000,
        "interest_rate": 0.085,  # 8.5% p.a.
        "default_tenure_months": 240,  # 20 years
        "min_income": 30_000,
        "min_lead_score": 35,
    },
    "Mortgage Loan": {
        "max_foir": 0.45,
        "max_amount": 75_00_000,  # ₹75 Lakh
        "min_amount": 3_00_000,
        "interest_rate": 0.095,  # 9.5% p.a.
        "default_tenure_months": 180,  # 15 years
        "min_income": 25_000,
        "min_lead_score": 35,
    },
    "Auto Loan": {
        "max_foir": 0.40,
        "max_amount": 20_00_000,  # ₹20 Lakh
        "min_amount": 1_00_000,
        "interest_rate": 0.095,  # 9.5% p.a.
        "default_tenure_months": 84,  # 7 years
        "min_income": 20_000,
        "min_lead_score": 30,
    },
}


# ---------------------------------------------------------------------------
# Eligible loan amount calculator
# ---------------------------------------------------------------------------

def compute_eligible_amount(
    monthly_income: float,
    existing_emi: float,
    loan_product: str,
    config: dict = None,
) -> float:
    """Compute eligible loan amount using standard EMI capacity formula.

    Available EMI = (monthly_income × max_FOIR) - existing_EMI
    Eligible Amount = Available_EMI × [(1+r)^n - 1] / [r × (1+r)^n]
      where r = monthly interest rate, n = tenure in months

    Args:
        monthly_income: Estimated monthly income
        existing_emi: Current monthly EMI obligations
        loan_product: One of the 4 loan products
        config: Optional override for loan config

    Returns:
        Eligible loan amount (capped at product max)
    """
    if config is None:
        config = LOAN_CONFIG.get(loan_product, LOAN_CONFIG["Personal Loan"])

    max_foir = config["max_foir"]
    annual_rate = config["interest_rate"]
    tenure = config["default_tenure_months"]
    max_amt = config["max_amount"]
    min_amt = config["min_amount"]

    # Available EMI capacity
    available_emi = (monthly_income * max_foir) - existing_emi
    if available_emi <= 0:
        return 0.0

    # Monthly interest rate
    r = annual_rate / 12

    # Present value of annuity formula
    if r > 0:
        pv_factor = ((1 + r) ** tenure - 1) / (r * (1 + r) ** tenure)
    else:
        pv_factor = tenure

    eligible = available_emi * pv_factor

    # Apply caps
    eligible = min(eligible, max_amt)
    eligible = max(eligible, 0)

    # Below minimum → not eligible
    if eligible < min_amt:
        return 0.0

    return round(eligible, -3)  # Round to nearest 1000


# ---------------------------------------------------------------------------
# Risk flag detection
# ---------------------------------------------------------------------------

def detect_risk_flags(row: pd.Series) -> List[str]:
    """Detect risk flags for underwriter review.

    Args:
        row: Single customer row with all features + predictions

    Returns:
        List of risk flag descriptions
    """
    flags = []

    # 1. Irregular income pattern
    cashflow_cv = row.get("cashflow_cv", 0)
    if cashflow_cv > 0.40:
        flags.append("Irregular income pattern (cash-flow CV > 40%)")
    elif cashflow_cv > 0.30:
        flags.append("Moderately variable income pattern")

    # 2. High existing EMI obligation
    emi_ratio = row.get("emi_to_income_ratio", 0)
    if emi_ratio > 0.40:
        flags.append(f"High existing EMI burden ({emi_ratio*100:.0f}% of income)")
    elif emi_ratio > 0.30:
        flags.append(f"Moderate existing EMI burden ({emi_ratio*100:.0f}% of income)")

    # 3. Income-declaration mismatch
    deviation = row.get("income_deviation_pct", 0)
    if abs(deviation) > 30:
        direction = "under-declared" if deviation < 0 else "over-declared"
        flags.append(f"Significant income {direction} ({abs(deviation):.0f}% deviation)")
    elif abs(deviation) > 20:
        direction = "under-declared" if deviation < 0 else "over-declared"
        flags.append(f"Income {direction} ({abs(deviation):.0f}% deviation)")

    # 4. Low confidence in income estimate
    confidence = row.get("income_confidence_band", "Medium")
    if confidence == "Low":
        flags.append("Low confidence in income estimate — manual verification recommended")

    # 5. Very short tenure with bank
    tenure = row.get("tenure_months", 0)
    if tenure < 6:
        flags.append(f"Very short banking relationship ({tenure} months)")
    elif tenure < 12:
        flags.append(f"Short banking relationship ({tenure} months)")

    # 6. Sudden large inflow (non-organic income suspicion)
    # Proxy: if monthly_credit_std is very high relative to mean
    credit_mean = row.get("monthly_credit_mean", 0)
    credit_std = row.get("monthly_credit_std", 0)
    if credit_mean > 0 and credit_std / credit_mean > 0.6:
        flags.append("Unusual variance in credit inflows — verify source of funds")

    # 7. Low digital engagement for high-score lead
    lead_score = row.get("lead_score", 0)
    digital_eng = row.get("digital_engagement_composite", 0)
    if lead_score > 70 and digital_eng < 0.2:
        flags.append("High lead score but low digital engagement — verify intent")

    return flags


# ---------------------------------------------------------------------------
# Loan Readiness Score
# ---------------------------------------------------------------------------

def compute_loan_readiness_score(
    lead_score: float,
    income_confidence: str,
    emi_ratio: float,
    cashflow_cv: float,
    tenure_months: int,
    n_risk_flags: int,
) -> float:
    """Compute composite Loan Readiness Score (0-100).

    Weighted formula:
    - Lead Score contribution (40%)
    - Income confidence contribution (25%)
    - EMI capacity contribution (15%)
    - Cash-flow stability contribution (10%)
    - Relationship & risk adjustment (10%)
    """
    # Lead score component (already 0-100)
    lead_component = lead_score * 0.40

    # Income confidence component
    confidence_map = {"High": 90, "Medium": 60, "Low": 30}
    income_component = confidence_map.get(income_confidence, 50) * 0.25

    # EMI capacity (lower existing ratio → higher capacity → higher score)
    emi_capacity = max(0, (1 - emi_ratio)) * 100
    emi_component = emi_capacity * 0.15

    # Cash-flow stability (lower CV → more stable → higher score)
    stability = max(0, (1 - cashflow_cv)) * 100
    stability_component = stability * 0.10

    # Relationship + risk
    tenure_score = min(tenure_months / 60, 1) * 100  # max at 5 years
    risk_penalty = min(n_risk_flags * 10, 40)  # each flag costs 10 points, max 40
    relationship_component = max(0, tenure_score - risk_penalty) * 0.10

    total = (
        lead_component +
        income_component +
        emi_component +
        stability_component +
        relationship_component
    )

    return round(min(max(total, 0), 100), 1)


# ---------------------------------------------------------------------------
# Main underwriting pipeline
# ---------------------------------------------------------------------------

def run_underwriting_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full underwriting engine on scored + income-estimated data.

    Expects columns: lead_score, estimated_income, income_confidence_band,
    emi_to_income_ratio, cashflow_cv, loan_product_interest, etc.

    Adds columns:
    - loan_readiness_score
    - eligible_loan_amount
    - risk_flags (list of strings)
    - risk_flag_count
    - recommendation (Pre-qualified / Review Required / Not Recommended)
    """
    df = df.copy()

    # Compute per-row outputs
    readiness_scores = []
    eligible_amounts = []
    all_risk_flags = []
    recommendations = []

    for idx, row in df.iterrows():
        # Risk flags
        flags = detect_risk_flags(row)
        all_risk_flags.append(flags)

        # Loan readiness
        readiness = compute_loan_readiness_score(
            lead_score=row.get("lead_score", 0),
            income_confidence=row.get("income_confidence_band", "Medium"),
            emi_ratio=row.get("emi_to_income_ratio", 0),
            cashflow_cv=row.get("cashflow_cv", 0.3),
            tenure_months=row.get("tenure_months", 0),
            n_risk_flags=len(flags),
        )
        readiness_scores.append(readiness)

        # Eligible amount
        income = row.get("estimated_income", 0)
        existing_emi = row.get("avg_monthly_emi", 0)
        product = row.get("loan_product_interest", "Personal Loan")
        eligible = compute_eligible_amount(income, existing_emi, product)
        eligible_amounts.append(eligible)

        # Recommendation
        if readiness >= 65 and len(flags) <= 1 and eligible > 0:
            rec = "Pre-qualified"
        elif readiness >= 40 and eligible > 0:
            rec = "Review Required"
        else:
            rec = "Not Recommended"
        recommendations.append(rec)

    df["loan_readiness_score"] = readiness_scores
    df["eligible_loan_amount"] = eligible_amounts
    df["risk_flags"] = all_risk_flags
    df["risk_flag_count"] = [len(f) for f in all_risk_flags]
    df["recommendation"] = recommendations

    # Summary
    rec_counts = df["recommendation"].value_counts()
    print(f"\n  📋 Underwriting Engine Results:")
    print(f"     Total processed:   {len(df):,}")
    print(f"     Pre-qualified:     {rec_counts.get('Pre-qualified', 0):,}")
    print(f"     Review Required:   {rec_counts.get('Review Required', 0):,}")
    print(f"     Not Recommended:   {rec_counts.get('Not Recommended', 0):,}")
    print(f"     Avg Readiness:     {df['loan_readiness_score'].mean():.1f}")
    print(f"     Avg Eligible Amt:  ₹{df[df['eligible_loan_amount'] > 0]['eligible_loan_amount'].mean():,.0f}")

    return df
