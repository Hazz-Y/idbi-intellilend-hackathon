"""
IntelliLend — Master Pipeline Runner
======================================
Orchestrates the full IntelliLend pipeline end-to-end:
  1. Generate synthetic data (without conversion labels)
  2. Build feature matrix
  3. Assign conversion labels FROM engineered features (ensures learnability)
  4. Train lead scoring model (Engine A)
  5. Train income estimation model (Engine B)
  6. Run underwriting engine (Engine C)
  7. Generate SHAP explanations
  8. Save pipeline output for dashboard

Usage:
  python run_pipeline.py
"""

import os
import sys
import time
import pickle
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd


def assign_conversion_from_features(features_df: pd.DataFrame, target_base_rate: float = 0.10) -> pd.DataFrame:
    """Assign conversion labels directly from the engineered feature matrix.

    This ensures perfect correlation between the ML features and the labels,
    allowing the model to learn strong, discoverable patterns.

    The conversion probability is a logistic function of the features that the
    ML model will actually train on, creating a realistic but learnable signal.

    Args:
        features_df: Feature matrix from build_feature_matrix()
        target_base_rate: Desired overall conversion rate (~10%)

    Returns:
        features_df with updated loan_converted column
    """
    np.random.seed(42)

    n = len(features_df)

    # Normalize key features to 0-1 for the scoring formula
    def _normalize(series):
        mn, mx = series.min(), series.max()
        if mx > mn:
            return (series - mn) / (mx - mn)
        return pd.Series(0.5, index=series.index)

    # Core scoring features (these are what the ML model will learn)
    digital_eng = _normalize(features_df["digital_engagement_composite"]) if "digital_engagement_composite" in features_df.columns else 0.5
    cashflow_stability = 1 - _normalize(features_df["cashflow_cv"]) if "cashflow_cv" in features_df.columns else 0.5  # lower CV = better
    emi_capacity = 1 - _normalize(features_df["emi_to_income_ratio"]) if "emi_to_income_ratio" in features_df.columns else 0.5  # lower ratio = better
    salary_reg = _normalize(features_df["salary_regularity_score"]) if "salary_regularity_score" in features_df.columns else 0.5
    app_logins = _normalize(features_df["avg_app_logins"]) if "avg_app_logins" in features_df.columns else 0.5
    loan_visits = _normalize(features_df["avg_loan_page_visits"]) if "avg_loan_page_visits" in features_df.columns else 0.5
    emi_calc = _normalize(features_df["avg_emi_calc_views"]) if "avg_emi_calc_views" in features_df.columns else 0.5
    eng_trend = _normalize(features_df["engagement_trend"]) if "engagement_trend" in features_df.columns else 0.5
    recency = _normalize(features_df["recency_score"]) if "recency_score" in features_df.columns else 0.5
    frequency = _normalize(features_df["frequency_score"]) if "frequency_score" in features_df.columns else 0.5
    tenure_norm = _normalize(features_df["tenure_months"]) if "tenure_months" in features_df.columns else 0.5

    # Intent flags (binary)
    auto_intent = features_df.get("has_auto_spend", 0)
    realestate_intent = features_df.get("has_realestate_spend", 0)
    wedding_intent = features_df.get("has_wedding_spend", 0)
    home_improve = features_df.get("has_home_improvement_spend", 0)

    # Composite propensity score (weighted combination)
    propensity = (
        0.20 * digital_eng +        # digital engagement composite
        0.15 * cashflow_stability +  # cash-flow stability
        0.12 * emi_capacity +        # EMI headroom
        0.10 * salary_reg +          # salary regularity
        0.10 * app_logins +          # app usage
        0.08 * loan_visits +         # loan page browsing
        0.07 * emi_calc +            # EMI calculator usage
        0.05 * eng_trend +           # increasing engagement
        0.05 * recency +             # recent activity
        0.04 * frequency +           # transaction frequency
        0.04 * tenure_norm           # relationship length
    )

    # Add intent signal bonus
    intent_bonus = (
        0.05 * auto_intent +
        0.05 * realestate_intent +
        0.04 * wedding_intent +
        0.03 * home_improve
    )
    propensity = propensity + intent_bonus

    # Add some noise (so it's not perfectly learnable — realistic)
    noise = np.random.normal(0, 0.04, n)
    propensity = propensity + noise

    # Calibrate to target base rate using logistic transform
    # Find the bias term that produces the desired base rate
    # sigmoid(propensity * scale + bias) should average to target_base_rate
    propensity_centered = (propensity - propensity.mean()) / max(propensity.std(), 0.01)

    # Binary search for the right bias
    scale = 4.5  # steepness of the logistic
    lo, hi = -10, 10
    for _ in range(50):
        mid = (lo + hi) / 2
        probs = 1 / (1 + np.exp(-(propensity_centered * scale + mid)))
        actual_rate = probs.mean()
        if actual_rate > target_base_rate:
            hi = mid
        else:
            lo = mid
    bias = (lo + hi) / 2

    conv_proba = 1 / (1 + np.exp(-(propensity_centered * scale + bias)))
    conv_proba = np.clip(conv_proba, 0.001, 0.999)

    # Sample binary labels
    converted = (np.random.random(n) < conv_proba).astype(int)

    features_df = features_df.copy()
    features_df["conversion_probability"] = np.round(conv_proba, 4)
    features_df["loan_converted"] = converted

    actual_rate = converted.mean() * 100
    print(f"  📊 Conversion labels assigned from engineered features")
    print(f"     Target base rate: {target_base_rate*100:.1f}%")
    print(f"     Actual base rate: {actual_rate:.1f}%")
    print(f"     Total converted:  {converted.sum()}")
    print(f"     Propensity range: {conv_proba.min():.4f} — {conv_proba.max():.4f}")

    return features_df


def run_full_pipeline():
    """Execute the complete IntelliLend pipeline."""
    total_start = time.time()

    print("=" * 70)
    print("🚀 IntelliLend — Full Pipeline Execution")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Phase 1: Data Generation
    # -----------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("📦 PHASE 1: Data Generation")
    print("─" * 70)

    from data.generate_synthetic_data import main as generate_data
    customers_full, txns_df, behavior_df = generate_data()

    # -----------------------------------------------------------------------
    # Phase 2: Feature Engineering
    # -----------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("⚙️  PHASE 2: Feature Engineering")
    print("─" * 70)

    from src.features import build_feature_matrix
    features_df = build_feature_matrix(customers_full, txns_df, behavior_df)

    print(f"\n  Feature matrix shape: {features_df.shape}")
    print(f"  Sample features: {list(features_df.columns[:10])}")

    # -----------------------------------------------------------------------
    # Phase 2.5: Re-assign conversion labels from actual features
    # -----------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("🔄 PHASE 2.5: Assigning conversion labels from engineered features")
    print("─" * 70)

    features_df = assign_conversion_from_features(features_df, target_base_rate=0.10)

    # -----------------------------------------------------------------------
    # Phase 3: Engine A — Lead Scoring
    # -----------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("🎯 PHASE 3: Engine A — Lead Scoring Model")
    print("─" * 70)

    from src.lead_scoring_model import train_lead_scoring_model, save_model as save_lead_model
    lead_result = train_lead_scoring_model(features_df)
    save_lead_model(lead_result)

    # Get predictions
    scored_df = lead_result["predictions_df"]

    # -----------------------------------------------------------------------
    # Phase 4: Engine B — Income Estimation
    # -----------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("💰 PHASE 4: Engine B — Income Estimation Model")
    print("─" * 70)

    from src.income_estimation_model import (
        train_income_estimation_model,
        save_model as save_income_model,
    )
    income_result = train_income_estimation_model(scored_df)
    save_income_model(income_result)

    # Merge income predictions
    pipeline_df = income_result["predictions_df"]

    # -----------------------------------------------------------------------
    # Phase 5: Engine C — Underwriting
    # -----------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("📋 PHASE 5: Engine C — Underwriting Recommendation")
    print("─" * 70)

    from src.underwriting_engine import run_underwriting_engine
    pipeline_df = run_underwriting_engine(pipeline_df)

    # -----------------------------------------------------------------------
    # Phase 6: Explainability
    # -----------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("🔍 PHASE 6: Explainability Layer (SHAP)")
    print("─" * 70)

    from src.explainability import generate_batch_explanations
    lead_model_dict = {
        "base_model": lead_result["base_model"],
        "model": lead_result["model"],
        "feature_cols": lead_result["feature_cols"],
    }
    pipeline_df = generate_batch_explanations(lead_model_dict, pipeline_df)

    # -----------------------------------------------------------------------
    # Save pipeline output
    # -----------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("💾 Saving Pipeline Output")
    print("─" * 70)

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    output_path = os.path.join(models_dir, "pipeline_output.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(pipeline_df, f)
    print(f"  ✅ Pipeline output saved: {output_path}")

    # Also save a CSV summary (without complex columns)
    csv_cols = [c for c in pipeline_df.columns if pipeline_df[c].dtype != object or c in [
        "customer_id", "name", "city", "city_tier", "occupation",
        "loan_product_interest", "recommendation", "income_confidence_band",
        "explanation_summary",
    ]]
    csv_path = os.path.join(os.path.dirname(__file__), "data", "pipeline_summary.csv")
    pipeline_df[csv_cols].to_csv(csv_path, index=False)
    print(f"  ✅ Summary CSV saved: {csv_path}")

    # -----------------------------------------------------------------------
    # Final Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("✅ IntelliLend — Pipeline Complete!")
    print("=" * 70)
    print(f"\n  Total time:           {elapsed:.1f} seconds")
    print(f"  Customers processed:  {len(pipeline_df):,}")
    print(f"  Lead scores range:    {pipeline_df['lead_score'].min():.1f} — {pipeline_df['lead_score'].max():.1f}")
    print(f"  Avg lead score:       {pipeline_df['lead_score'].mean():.1f}")

    conv_sim = lead_result["metrics"].get("conversion_rate_simulation", {})
    top20 = conv_sim.get("top_20pct", "N/A")
    base = lead_result["metrics"].get("base_conversion_rate", "N/A")
    print(f"  Base conversion:      {base}%")
    print(f"  Top-20% conversion:   {top20}%")

    rec_counts = pipeline_df["recommendation"].value_counts().to_dict()
    print(f"\n  Recommendations:")
    for rec, count in rec_counts.items():
        print(f"    {rec:20s}: {count:,}")

    print(f"\n  🚀 Launch the dashboard with:")
    print(f"     streamlit run app/dashboard.py")
    print("=" * 70)

    return pipeline_df, lead_result, income_result


if __name__ == "__main__":
    run_full_pipeline()
