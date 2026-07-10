"""
IntelliLend — Explainability Layer
====================================
Converts SHAP feature importance values into plain-English, business-readable
explanations for each customer's lead score.

No paid LLM required — uses template-based sentence generation keyed off
the top SHAP features and their values.

Compliant with RBI Fair Practices Code: every score has a transparent,
auditable explanation.
"""

import numpy as np
import pandas as pd
import shap
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# Feature → Business Language mapping
# ---------------------------------------------------------------------------

FEATURE_TEMPLATES = {
    # Income & salary
    "monthly_credit_mean": {
        "positive": "Strong average monthly inflows of ₹{value:,.0f}",
        "negative": "Low average monthly inflows of ₹{value:,.0f}",
    },
    "estimated_salary": {
        "positive": "Consistent salary credit pattern of ~₹{value:,.0f}/month",
        "negative": "Inconsistent or low salary pattern (~₹{value:,.0f}/month)",
    },
    "salary_regularity_score": {
        "positive": "Highly regular income credit pattern ({value:.0%} regularity)",
        "negative": "Irregular income credit pattern ({value:.0%} regularity)",
    },
    "has_salary_credit": {
        "positive": "Identifiable recurring salary credits detected",
        "negative": "No clear salary credit pattern found",
    },
    "declared_monthly_income": {
        "positive": "Declared income of ₹{value:,.0f}/month supports eligibility",
        "negative": "Low declared income of ₹{value:,.0f}/month",
    },

    # Cash-flow stability
    "cashflow_cv": {
        "positive": "Stable cash-flow pattern (low variability of {value:.1%})",
        "negative": "Variable cash-flow pattern (CV of {value:.1%})",
    },
    "monthly_credit_std": {
        "positive": "Consistent monthly credit amounts",
        "negative": "Highly variable monthly credit amounts",
    },

    # EMI & obligations
    "emi_to_income_ratio": {
        "positive": "Low existing EMI burden ({value:.0%} of income)",
        "negative": "High existing EMI burden ({value:.0%} of income)",
    },
    "avg_monthly_emi": {
        "positive": "Manageable existing EMI of ₹{value:,.0f}/month",
        "negative": "Significant existing EMI of ₹{value:,.0f}/month",
    },

    # Digital engagement
    "digital_engagement_composite": {
        "positive": "Active digital banking engagement (score: {value:.2f})",
        "negative": "Limited digital banking engagement (score: {value:.2f})",
    },
    "avg_loan_page_visits": {
        "positive": "Frequent loan product page visits ({value:.1f}/month) indicate interest",
        "negative": "Low loan product page exploration ({value:.1f}/month)",
    },
    "avg_emi_calc_views": {
        "positive": "Active use of EMI calculator ({value:.1f}/month) shows purchase intent",
        "negative": "Minimal EMI calculator usage ({value:.1f}/month)",
    },
    "avg_app_logins": {
        "positive": "Frequent app usage ({value:.1f} logins/month)",
        "negative": "Infrequent app usage ({value:.1f} logins/month)",
    },
    "engagement_trend": {
        "positive": "Increasing digital engagement trend (+{value:.0%})",
        "negative": "Declining digital engagement trend ({value:.0%})",
    },

    # Spend patterns
    "has_auto_spend": {
        "positive": "Recent auto dealer/showroom transactions detected → Auto Loan intent signal",
        "negative": "No auto-related spend detected",
    },
    "has_realestate_spend": {
        "positive": "Real estate broker/property transactions detected → Home/Mortgage Loan intent",
        "negative": "No real estate spend detected",
    },
    "has_wedding_spend": {
        "positive": "Wedding/event-related spend detected → Personal Loan intent signal",
        "negative": "No wedding-related spend",
    },
    "has_education_spend": {
        "positive": "Education-related spending detected → Personal/Education Loan intent",
        "negative": "No education-related spend",
    },
    "has_home_improvement_spend": {
        "positive": "Home improvement spending detected → Mortgage/Home Loan intent",
        "negative": "No home improvement spend",
    },
    "total_monthly_debit": {
        "positive": "Healthy monthly spending of ₹{value:,.0f} indicates active lifestyle",
        "negative": "Low monthly spending of ₹{value:,.0f}",
    },
    "spend_concentration_hhi": {
        "positive": "Diversified spending across categories",
        "negative": "Concentrated spending pattern",
    },

    # RFM
    "recency_score": {
        "positive": "Recent banking activity (high recency)",
        "negative": "Low recent banking activity",
    },
    "frequency_score": {
        "positive": "Frequent transactions (active banking relationship)",
        "negative": "Infrequent transactions",
    },
    "monetary_score": {
        "positive": "High transaction volume (strong monetary relationship)",
        "negative": "Low transaction volume",
    },

    # Demographics
    "age": {
        "positive": "Age profile ({value:.0f} yrs) aligns with loan product lifecycle",
        "negative": "Age profile ({value:.0f} yrs) is outside typical range for this product",
    },
    "tenure_months": {
        "positive": "Long banking relationship ({value:.0f} months) with IDBI Bank",
        "negative": "Relatively new banking relationship ({value:.0f} months)",
    },
    "existing_products": {
        "positive": "Multiple existing products ({value:.0f}) show relationship depth",
        "negative": "Limited existing product holdings ({value:.0f})",
    },
    "occupation_encoded": {
        "positive": "Stable employment category supports income reliability",
        "negative": "Employment category indicates variable income",
    },
    "city_tier_encoded": {
        "positive": "Metropolitan location supports higher income potential",
        "negative": "Non-metro location may limit income ceiling",
    },

    # Loan product interest flags
    "interest_personal_loan": {
        "positive": "Personal Loan interest indicators detected",
        "negative": "No Personal Loan interest signals",
    },
    "interest_home_loan": {
        "positive": "Home Loan interest indicators detected",
        "negative": "No Home Loan interest signals",
    },
    "interest_mortgage_loan": {
        "positive": "Mortgage Loan interest indicators detected",
        "negative": "No Mortgage Loan interest signals",
    },
    "interest_auto_loan": {
        "positive": "Auto Loan interest indicators detected",
        "negative": "No Auto Loan interest signals",
    },
}


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model_dict: dict,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> Tuple[np.ndarray, List[str]]:
    """Compute SHAP values for the given model and feature matrix.

    Args:
        model_dict: Dict containing 'base_model' (LightGBM/XGBoost) and 'feature_cols'
        X: Feature DataFrame
        max_samples: Max samples for SHAP background (for speed)

    Returns:
        (shap_values array, feature_names list)
    """
    base_model = model_dict.get("base_model", model_dict.get("model"))
    feature_cols = model_dict["feature_cols"]
    available = [c for c in feature_cols if c in X.columns]
    X_subset = X[available].fillna(0)

    # Use TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(base_model)

    # Compute SHAP values
    shap_vals = explainer.shap_values(X_subset)

    # For binary classification, shap_values may return a list [class_0, class_1]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # Take class 1 (positive class) SHAP values

    return shap_vals, available


# ---------------------------------------------------------------------------
# Plain-English explanation generator
# ---------------------------------------------------------------------------

def generate_explanation(
    shap_values: np.ndarray,
    feature_values: pd.Series,
    feature_names: List[str],
    loan_product: str = "",
    top_n: int = 4,
) -> Dict:
    """Generate a plain-English explanation for a single customer's score.

    Args:
        shap_values: 1D array of SHAP values for this customer
        feature_values: Series of feature values for this customer
        feature_names: List of feature names
        loan_product: The customer's loan product interest
        top_n: Number of top features to include

    Returns:
        Dict with:
        - summary: One-sentence business explanation
        - details: List of factor dicts {feature, direction, description, shap_value}
        - shap_dict: Full SHAP values dict
    """
    # Sort features by absolute SHAP value
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[::-1][:top_n]

    details = []
    summary_parts = []

    for idx in top_indices:
        feat_name = feature_names[idx]
        shap_val = shap_values[idx]
        feat_val = feature_values.get(feat_name, 0)
        direction = "positive" if shap_val > 0 else "negative"

        # Get template
        templates = FEATURE_TEMPLATES.get(feat_name)
        if templates:
            template = templates[direction]
            try:
                description = template.format(value=feat_val)
            except (ValueError, KeyError):
                description = template
        else:
            # Fallback generic template
            if direction == "positive":
                description = f"{feat_name.replace('_', ' ').title()} contributes positively"
            else:
                description = f"{feat_name.replace('_', ' ').title()} contributes negatively"

        details.append({
            "feature": feat_name,
            "direction": direction,
            "description": description,
            "shap_value": round(float(shap_val), 4),
            "feature_value": round(float(feat_val), 4) if isinstance(feat_val, (int, float, np.floating, np.integer)) else feat_val,
        })

        if direction == "positive":
            summary_parts.append(description)

    # Build summary sentence
    if loan_product:
        product_text = f" → strong {loan_product} intent"
    else:
        product_text = ""

    if summary_parts:
        summary = " + ".join(summary_parts[:3]) + product_text
    else:
        summary = f"Mixed signals detected for {loan_product or 'loan'} eligibility — review recommended"

    # Full SHAP dict
    shap_dict = {feature_names[i]: round(float(shap_values[i]), 4) for i in range(len(feature_names))}

    return {
        "summary": summary,
        "details": details,
        "shap_dict": shap_dict,
    }


def generate_batch_explanations(
    model_dict: dict,
    features_df: pd.DataFrame,
    top_n: int = 4,
) -> pd.DataFrame:
    """Generate SHAP-based explanations for all customers.

    Adds columns:
    - explanation_summary: One-line business-readable explanation
    - explanation_details: List of top factor dicts
    - shap_values_dict: Full SHAP values per customer
    """
    feature_cols = model_dict["feature_cols"]
    available = [c for c in feature_cols if c in features_df.columns]
    X = features_df[available].fillna(0)

    print(f"  🔍 Computing SHAP values for {len(X)} customers...")
    shap_vals, feat_names = compute_shap_values(model_dict, features_df)

    summaries = []
    details_list = []
    shap_dicts = []

    for i in range(len(features_df)):
        loan_product = features_df.iloc[i].get("loan_product_interest", "")
        result = generate_explanation(
            shap_values=shap_vals[i],
            feature_values=features_df.iloc[i],
            feature_names=feat_names,
            loan_product=loan_product,
            top_n=top_n,
        )
        summaries.append(result["summary"])
        details_list.append(result["details"])
        shap_dicts.append(result["shap_dict"])

    features_df = features_df.copy()
    features_df["explanation_summary"] = summaries
    features_df["explanation_details"] = details_list
    features_df["shap_values_dict"] = shap_dicts

    print(f"  ✅ Generated {len(summaries)} explanations")
    return features_df
