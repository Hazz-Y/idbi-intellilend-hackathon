"""
IntelliLend — Feature Engineering Pipeline
===========================================
Transforms raw transaction + behavioral data into ML-ready features for
lead scoring (Engine A) and income estimation (Engine B).

Feature groups:
  1. Cash-flow stability (coefficient of variation of monthly inflows)
  2. Salary / recurring credit detection
  3. EMI-to-income ratio (inferred from existing debit patterns)
  4. RFM (Recency, Frequency, Monetary) scores
  5. Merchant-category spend patterns & big-ticket flags
  6. Digital engagement composite
  7. Demographic & relationship features
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# 1. Cash-flow stability
# ---------------------------------------------------------------------------

def compute_cashflow_stability(txns: pd.DataFrame) -> pd.DataFrame:
    """Compute coefficient of variation (CV) of monthly credit inflows
    per customer. Lower CV → more stable income.

    Returns: DataFrame with [customer_id, monthly_credit_mean,
             monthly_credit_std, cashflow_cv]
    """
    credits = txns[txns["txn_type"] == "credit"].copy()
    credits["month"] = pd.to_datetime(credits["txn_date"]).dt.to_period("M")

    monthly = (
        credits
        .groupby(["customer_id", "month"])["amount"]
        .sum()
        .reset_index()
    )

    stats = (
        monthly
        .groupby("customer_id")["amount"]
        .agg(monthly_credit_mean="mean", monthly_credit_std="std")
        .fillna(0)
        .reset_index()
    )

    stats["cashflow_cv"] = (
        stats["monthly_credit_std"] /
        stats["monthly_credit_mean"].replace(0, 1)
    )
    stats["cashflow_cv"] = stats["cashflow_cv"].clip(0, 2)

    return stats


# ---------------------------------------------------------------------------
# 2. Salary / recurring credit detection
# ---------------------------------------------------------------------------

def detect_salary_pattern(txns: pd.DataFrame) -> pd.DataFrame:
    """Detect recurring, regular credit patterns that look like salary.

    Heuristics:
    - Credits in "Salary" category → direct signal
    - For non-salary credits: check if there's a consistent single
      large credit each month (regularity score)

    Returns: DataFrame with [customer_id, has_salary_credit,
             salary_regularity_score, estimated_salary]
    """
    credits = txns[txns["txn_type"] == "credit"].copy()
    credits["month"] = pd.to_datetime(credits["txn_date"]).dt.to_period("M")

    results = []
    for cid, group in credits.groupby("customer_id"):
        salary_txns = group[group["category"] == "Salary"]

        if len(salary_txns) > 0:
            # Direct salary detected
            months_with_salary = salary_txns["month"].nunique()
            total_months = group["month"].nunique()
            regularity = months_with_salary / max(total_months, 1)

            avg_salary = salary_txns.groupby("month")["amount"].sum().mean()

            results.append({
                "customer_id": cid,
                "has_salary_credit": 1,
                "salary_regularity_score": round(regularity, 4),
                "estimated_salary": round(avg_salary, 2),
            })
        else:
            # Try to find largest recurring credit per month
            monthly_max = group.groupby("month")["amount"].max()
            if len(monthly_max) >= 3:
                cv = monthly_max.std() / monthly_max.mean() if monthly_max.mean() > 0 else 1
                regularity = max(0, 1 - cv)  # low cv → high regularity
            else:
                regularity = 0.0

            results.append({
                "customer_id": cid,
                "has_salary_credit": 0,
                "salary_regularity_score": round(regularity, 4),
                "estimated_salary": round(monthly_max.mean() if len(monthly_max) > 0 else 0, 2),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 3. EMI-to-income ratio
# ---------------------------------------------------------------------------

def compute_emi_ratio(txns: pd.DataFrame) -> pd.DataFrame:
    """Compute existing EMI burden as ratio of total monthly debits
    tagged as EMI to total monthly credits.

    Returns: DataFrame with [customer_id, avg_monthly_emi,
             avg_monthly_income_proxy, emi_to_income_ratio, emi_count]
    """
    credits = txns[txns["txn_type"] == "credit"].copy()
    credits["month"] = pd.to_datetime(credits["txn_date"]).dt.to_period("M")
    monthly_income = credits.groupby("customer_id")["amount"].sum() / 12

    emi_debits = txns[(txns["txn_type"] == "debit") & (txns["category"] == "EMI")].copy()
    emi_debits["month"] = pd.to_datetime(emi_debits["txn_date"]).dt.to_period("M")

    # Monthly EMI total
    monthly_emi = emi_debits.groupby("customer_id")["amount"].sum() / 12
    emi_count = emi_debits.groupby("customer_id")["amount"].count() / 12

    # Merge
    all_cids = txns["customer_id"].unique()
    result = pd.DataFrame({"customer_id": all_cids})
    result = result.merge(
        monthly_income.rename("avg_monthly_income_proxy").reset_index(),
        on="customer_id", how="left"
    )
    result = result.merge(
        monthly_emi.rename("avg_monthly_emi").reset_index(),
        on="customer_id", how="left"
    )
    result = result.merge(
        emi_count.rename("emi_count").reset_index(),
        on="customer_id", how="left"
    )
    result = result.fillna(0)

    result["emi_to_income_ratio"] = (
        result["avg_monthly_emi"] /
        result["avg_monthly_income_proxy"].replace(0, 1)
    )
    result["emi_to_income_ratio"] = result["emi_to_income_ratio"].clip(0, 1)

    return result


# ---------------------------------------------------------------------------
# 4. RFM scores
# ---------------------------------------------------------------------------

def compute_rfm_scores(txns: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary scores (0–1 normalized) based
    on transaction behavior.

    - Recency: days since last transaction (inverted & normalized)
    - Frequency: total transaction count (normalized)
    - Monetary: total credit amount (normalized)
    """
    txns_copy = txns.copy()
    txns_copy["txn_date"] = pd.to_datetime(txns_copy["txn_date"])
    ref_date = txns_copy["txn_date"].max() + pd.Timedelta(days=1)

    rfm = txns_copy.groupby("customer_id").agg(
        recency=("txn_date", lambda x: (ref_date - x.max()).days),
        frequency=("amount", "count"),
        monetary=("amount", lambda x: x[txns_copy.loc[x.index, "txn_type"] == "credit"].sum()),
    ).reset_index()

    # Normalize to 0-1 using min-max
    for col in ["recency", "frequency", "monetary"]:
        col_min = rfm[col].min()
        col_max = rfm[col].max()
        if col_max > col_min:
            rfm[f"{col}_score"] = (rfm[col] - col_min) / (col_max - col_min)
        else:
            rfm[f"{col}_score"] = 0.5

    # Invert recency (lower days = higher score)
    rfm["recency_score"] = 1 - rfm["recency_score"]

    return rfm[["customer_id", "recency_score", "frequency_score", "monetary_score"]]


# ---------------------------------------------------------------------------
# 5. Spend pattern analysis
# ---------------------------------------------------------------------------

def analyze_spend_patterns(txns: pd.DataFrame) -> pd.DataFrame:
    """Analyze merchant category spend distributions and flag intent signals.

    Features:
    - Spend concentration (HHI across categories)
    - Big-ticket flags: auto dealer, real estate, wedding, education
    - Total monthly debit amount
    """
    debits = txns[txns["txn_type"] == "debit"].copy()

    # Category spend proportions
    cust_cat = debits.groupby(["customer_id", "category"])["amount"].sum().reset_index()
    cust_total = debits.groupby("customer_id")["amount"].sum().reset_index()
    cust_total.columns = ["customer_id", "total_debit"]

    results = []
    for cid, group in cust_cat.groupby("customer_id"):
        total = group["amount"].sum()
        proportions = group["amount"] / total if total > 0 else group["amount"]
        hhi = float((proportions ** 2).sum())  # Herfindahl index

        # Intent flags (has any spend in these categories?)
        cats = set(group["category"])
        has_auto = 1 if "Auto_Dealer" in cats else 0
        has_realestate = 1 if "Real_Estate_Broker" in cats else 0
        has_wedding = 1 if "Wedding_Services" in cats else 0
        has_education = 1 if "Education" in cats else 0
        has_home_improve = 1 if "Home_Improvement" in cats else 0
        has_jewellery = 1 if "Jewellery" in cats else 0

        results.append({
            "customer_id": cid,
            "spend_concentration_hhi": round(hhi, 4),
            "has_auto_spend": has_auto,
            "has_realestate_spend": has_realestate,
            "has_wedding_spend": has_wedding,
            "has_education_spend": has_education,
            "has_home_improvement_spend": has_home_improve,
            "has_jewellery_spend": has_jewellery,
            "total_monthly_debit": round(total / 12, 2),
        })

    df = pd.DataFrame(results)
    return df.merge(cust_total, on="customer_id", how="left")


# ---------------------------------------------------------------------------
# 6. Digital engagement composite
# ---------------------------------------------------------------------------

def compute_digital_engagement(behavior_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate behavioral signals into a per-customer engagement feature set.

    Features:
    - avg_app_logins: mean monthly app logins
    - avg_emi_calc_views: mean monthly EMI calculator usage
    - avg_loan_page_visits: mean monthly loan page visits
    - engagement_trend: slope of engagement over time (increasing = positive signal)
    - digital_engagement_composite: weighted composite score (0–1)
    """
    agg = behavior_df.groupby("customer_id").agg(
        avg_app_logins=("app_logins", "mean"),
        avg_emi_calc_views=("emi_calculator_views", "mean"),
        avg_loan_page_visits=("loan_page_visits", "mean"),
        total_product_views=("product_page_visits", "sum"),
    ).reset_index()

    # Engagement trend (simple: compare last 3 months vs first 3 months)
    behavior_sorted = behavior_df.sort_values(["customer_id", "month"])
    trends = []
    for cid, group in behavior_sorted.groupby("customer_id"):
        engagement = group["app_logins"] + group["loan_page_visits"]
        if len(engagement) >= 6:
            first_half = engagement.iloc[:len(engagement)//2].mean()
            second_half = engagement.iloc[len(engagement)//2:].mean()
            trend = (second_half - first_half) / max(first_half, 1)
        else:
            trend = 0.0
        trends.append({"customer_id": cid, "engagement_trend": round(trend, 4)})

    trend_df = pd.DataFrame(trends)
    agg = agg.merge(trend_df, on="customer_id", how="left")

    # Composite score (normalized weighted sum)
    max_logins = max(agg["avg_app_logins"].max(), 1)
    max_emi = max(agg["avg_emi_calc_views"].max(), 1)
    max_loan = max(agg["avg_loan_page_visits"].max(), 1)

    agg["digital_engagement_composite"] = (
        0.3 * (agg["avg_app_logins"] / max_logins) +
        0.3 * (agg["avg_emi_calc_views"] / max_emi) +
        0.3 * (agg["avg_loan_page_visits"] / max_loan) +
        0.1 * agg["engagement_trend"].clip(-1, 1).apply(lambda x: (x + 1) / 2)
    ).clip(0, 1)

    return agg


# ---------------------------------------------------------------------------
# 7. Build full feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    customers_df: pd.DataFrame,
    txns_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
) -> pd.DataFrame:
    """Orchestrate all feature engineering and merge into a single DataFrame.

    Returns a ML-ready DataFrame with customer_id + all engineered features +
    target labels (loan_converted, true_monthly_income).
    """
    print("  ⚙️  Computing cash-flow stability...")
    cashflow = compute_cashflow_stability(txns_df)

    print("  ⚙️  Detecting salary patterns...")
    salary = detect_salary_pattern(txns_df)

    print("  ⚙️  Computing EMI ratios...")
    emi = compute_emi_ratio(txns_df)

    print("  ⚙️  Computing RFM scores...")
    rfm = compute_rfm_scores(txns_df)

    print("  ⚙️  Analyzing spend patterns...")
    spend = analyze_spend_patterns(txns_df)

    print("  ⚙️  Computing digital engagement...")
    digital = compute_digital_engagement(behavior_df)

    # Start with customer demographics
    features = customers_df[[
        "customer_id", "name", "city", "age", "city_tier", "occupation",
        "declared_monthly_income", "tenure_months", "existing_products",
        "digital_engagement_score", "loan_product_interest",
    ]].copy()

    # Add targets if available
    if "true_monthly_income" in customers_df.columns:
        features["true_monthly_income"] = customers_df["true_monthly_income"]
    if "loan_converted" in customers_df.columns:
        features["loan_converted"] = customers_df["loan_converted"]
    if "conversion_probability" in customers_df.columns:
        features["conversion_probability"] = customers_df["conversion_probability"]
    if "cashflow_cv" in customers_df.columns:
        features["cashflow_cv_label"] = customers_df["cashflow_cv"]
    if "existing_emi_ratio" in customers_df.columns:
        features["existing_emi_ratio_label"] = customers_df["existing_emi_ratio"]

    # Merge engineered features
    for df in [cashflow, salary, emi, rfm, spend, digital]:
        features = features.merge(df, on="customer_id", how="left")

    # Encode categoricals
    features["occupation_encoded"] = features["occupation"].map({
        "salaried": 2, "self_employed": 1, "gig_worker": 0
    })
    features["city_tier_encoded"] = features["city_tier"].map({
        "Tier 1": 2, "Tier 2": 1, "Tier 3": 0
    })

    # Loan product one-hot
    for product in ["Personal Loan", "Home Loan", "Mortgage Loan", "Auto Loan"]:
        col_name = "interest_" + product.lower().replace(" ", "_")
        features[col_name] = (features["loan_product_interest"] == product).astype(int)

    # Fill NaN
    features = features.fillna(0)

    print(f"  ✅ Feature matrix built: {features.shape[0]} rows × {features.shape[1]} columns")
    return features


# ---------------------------------------------------------------------------
# Feature list for ML models
# ---------------------------------------------------------------------------

LEAD_SCORING_FEATURES = [
    "age", "occupation_encoded", "city_tier_encoded",
    "declared_monthly_income", "tenure_months", "existing_products",
    "monthly_credit_mean", "monthly_credit_std", "cashflow_cv",
    "has_salary_credit", "salary_regularity_score", "estimated_salary",
    "avg_monthly_emi", "emi_to_income_ratio", "emi_count",
    "recency_score", "frequency_score", "monetary_score",
    "spend_concentration_hhi",
    "has_auto_spend", "has_realestate_spend", "has_wedding_spend",
    "has_education_spend", "has_home_improvement_spend", "has_jewellery_spend",
    "total_monthly_debit",
    "avg_app_logins", "avg_emi_calc_views", "avg_loan_page_visits",
    "total_product_views", "engagement_trend", "digital_engagement_composite",
    "interest_personal_loan", "interest_home_loan",
    "interest_mortgage_loan", "interest_auto_loan",
]

INCOME_ESTIMATION_FEATURES = [
    "age", "occupation_encoded", "city_tier_encoded",
    "tenure_months", "existing_products",
    "monthly_credit_mean", "monthly_credit_std", "cashflow_cv",
    "has_salary_credit", "salary_regularity_score", "estimated_salary",
    "avg_monthly_emi", "emi_to_income_ratio",
    "recency_score", "frequency_score", "monetary_score",
    "total_monthly_debit", "total_debit",
    "avg_app_logins", "digital_engagement_composite",
]
