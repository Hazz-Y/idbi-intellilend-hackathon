"""
IntelliLend — Pipeline Sanity Tests
=====================================
Basic integration and unit tests to validate the full pipeline.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDataGeneration:
    """Tests for synthetic data generator."""

    def test_customer_generation(self):
        from data.generate_synthetic_data import generate_customers
        df = generate_customers(100)
        assert len(df) == 100
        assert "customer_id" in df.columns
        assert "true_monthly_income" in df.columns
        assert df["age"].between(21, 65).all()
        assert set(df["occupation"].unique()).issubset({"salaried", "self_employed", "gig_worker"})
        assert set(df["city_tier"].unique()).issubset({"Tier 1", "Tier 2", "Tier 3"})

    def test_transaction_generation(self):
        from data.generate_synthetic_data import generate_customers, generate_transactions
        custs = generate_customers(10)
        txns = generate_transactions(custs)
        assert len(txns) > 0
        assert "txn_type" in txns.columns
        assert set(txns["txn_type"].unique()).issubset({"credit", "debit"})
        assert (txns["amount"] > 0).all()

    def test_behavioral_signals(self):
        from data.generate_synthetic_data import generate_customers, generate_behavioral_signals
        custs = generate_customers(10)
        behavior = generate_behavioral_signals(custs)
        assert len(behavior) == 10 * 12  # 12 months per customer
        assert (behavior["app_logins"] >= 0).all()

    def test_income_ranges_realistic(self):
        from data.generate_synthetic_data import generate_customers
        df = generate_customers(1000)
        salaried = df[df["occupation"] == "salaried"]["true_monthly_income"]
        assert salaried.median() > 25_000
        assert salaried.median() < 250_000


class TestFeatureEngineering:
    """Tests for feature engineering pipeline."""

    def test_cashflow_stability(self):
        from src.features import compute_cashflow_stability
        txns = pd.DataFrame({
            "customer_id": ["A"] * 12,
            "txn_date": pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%Y-%m-%d"),
            "txn_type": ["credit"] * 12,
            "amount": [50000] * 12,  # perfectly stable
        })
        result = compute_cashflow_stability(txns)
        assert len(result) == 1
        assert result.iloc[0]["cashflow_cv"] == 0.0  # zero variance

    def test_emi_ratio(self):
        from src.features import compute_emi_ratio
        txns = pd.DataFrame({
            "customer_id": ["A"] * 24,
            "txn_date": (["2024-01-15"] * 12 + ["2024-01-10"] * 12),
            "txn_type": ["credit"] * 12 + ["debit"] * 12,
            "amount": [100_000] * 12 + [20_000] * 12,
            "category": ["Salary"] * 12 + ["EMI"] * 12,
        })
        result = compute_emi_ratio(txns)
        assert len(result) == 1
        ratio = result.iloc[0]["emi_to_income_ratio"]
        assert 0.15 <= ratio <= 0.25  # ~20% EMI to income

    def test_rfm_scores_normalized(self):
        from src.features import compute_rfm_scores
        txns = pd.DataFrame({
            "customer_id": ["A"] * 10 + ["B"] * 5,
            "txn_date": pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y-%m-%d").tolist() +
                        pd.date_range("2024-01-01", periods=5, freq="D").strftime("%Y-%m-%d").tolist(),
            "txn_type": ["credit"] * 15,
            "amount": [1000] * 15,
        })
        result = compute_rfm_scores(txns)
        assert len(result) == 2
        for col in ["recency_score", "frequency_score", "monetary_score"]:
            assert result[col].between(0, 1).all()


class TestUnderwritingEngine:
    """Tests for underwriting engine."""

    def test_eligible_amount_computation(self):
        from src.underwriting_engine import compute_eligible_amount
        # ₹1L income, no existing EMI, Personal Loan
        eligible = compute_eligible_amount(100_000, 0, "Personal Loan")
        assert eligible > 0
        assert eligible <= 25_00_000

    def test_zero_capacity(self):
        from src.underwriting_engine import compute_eligible_amount
        # EMI exceeds FOIR capacity
        eligible = compute_eligible_amount(50_000, 50_000, "Personal Loan")
        assert eligible == 0

    def test_loan_readiness_score_range(self):
        from src.underwriting_engine import compute_loan_readiness_score
        score = compute_loan_readiness_score(
            lead_score=80, income_confidence="High",
            emi_ratio=0.1, cashflow_cv=0.1,
            tenure_months=36, n_risk_flags=0,
        )
        assert 0 <= score <= 100
        assert score > 60  # Good profile should score high

    def test_risk_flags(self):
        from src.underwriting_engine import detect_risk_flags
        row = pd.Series({
            "cashflow_cv": 0.50,
            "emi_to_income_ratio": 0.45,
            "income_deviation_pct": -35,
            "income_confidence_band": "Low",
            "tenure_months": 3,
            "monthly_credit_mean": 100000,
            "monthly_credit_std": 80000,
            "lead_score": 75,
            "digital_engagement_composite": 0.1,
        })
        flags = detect_risk_flags(row)
        assert len(flags) >= 3  # Should have multiple flags


class TestExplainability:
    """Tests for explainability module."""

    def test_explanation_generation(self):
        from src.explainability import generate_explanation
        shap_values = np.array([0.5, -0.3, 0.2, 0.1, -0.05])
        feature_values = pd.Series({
            "monthly_credit_mean": 80000,
            "emi_to_income_ratio": 0.15,
            "digital_engagement_composite": 0.7,
            "cashflow_cv": 0.12,
            "age": 32,
        })
        feature_names = list(feature_values.index)

        result = generate_explanation(
            shap_values, feature_values, feature_names,
            loan_product="Home Loan", top_n=3
        )
        assert "summary" in result
        assert "details" in result
        assert len(result["details"]) == 3
        assert result["details"][0]["feature"] == "monthly_credit_mean"  # highest abs SHAP


class TestEndToEnd:
    """End-to-end integration test (small scale)."""

    def test_mini_pipeline(self):
        from data.generate_synthetic_data import (
            generate_customers, generate_transactions,
            generate_behavioral_signals, assign_conversion_labels,
        )
        from src.features import build_feature_matrix

        # Small dataset
        custs = generate_customers(50)
        txns = generate_transactions(custs)
        behavior = generate_behavioral_signals(custs)
        labels = assign_conversion_labels(custs, txns)
        custs_full = custs.merge(labels, on="customer_id", how="left")

        # Feature engineering
        features = build_feature_matrix(custs_full, txns, behavior)
        assert len(features) == 50
        assert "cashflow_cv" in features.columns or "monthly_credit_mean" in features.columns
        assert "lead_score" not in features.columns  # Not yet scored


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
