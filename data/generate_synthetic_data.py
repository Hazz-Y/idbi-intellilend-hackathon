"""
IntelliLend — Synthetic Data Generator
======================================
Generates ~5,000 realistic synthetic customer profiles with 12 months of
transaction history, behavioral signals, and ground-truth labels for
lead scoring and income estimation model training.

All data is 100% synthetic. No real customer PII is used.
Seed is fixed for reproducibility.

Author: IntelliLend Team (IDBI Bank Hackathon)
"""

import os
import sys
import sqlite3
import random
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
NUM_CUSTOMERS = 5000
MONTHS = 12  # months of transaction history
DB_PATH = os.path.join(os.path.dirname(__file__), "intellilend.db")
CSV_PATH = os.path.join(os.path.dirname(__file__), "sample_data.csv")

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_IN")
Faker.seed(SEED)

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------
TIER1_CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad",
]
TIER2_CITIES = [
    "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore",
    "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Coimbatore",
    "Kochi", "Chandigarh", "Guwahati", "Ranchi", "Mysuru",
]
TIER3_CITIES = [
    "Mathura", "Shimla", "Udaipur", "Jodhpur", "Agartala",
    "Bilaspur", "Rourkela", "Siliguri", "Ujjain", "Dhanbad",
    "Bareilly", "Moradabad", "Aligarh", "Jalandhar", "Salem",
]

OCCUPATION_DIST = {"salaried": 0.55, "self_employed": 0.30, "gig_worker": 0.15}

MERCHANT_CATEGORIES = [
    "Groceries", "Fuel", "Dining", "Electronics", "Apparel",
    "Travel", "Education", "Healthcare", "Utilities", "Entertainment",
    "Insurance", "Jewellery", "Home_Improvement", "Auto_Dealer",
    "Real_Estate_Broker", "Wedding_Services",
]

LOAN_PRODUCTS = ["Personal Loan", "Home Loan", "Mortgage Loan", "Auto Loan"]

# Income ranges by occupation (monthly, INR)
INCOME_RANGES = {
    "salaried":      (25_000, 250_000),
    "self_employed":  (15_000, 400_000),
    "gig_worker":     (8_000,  80_000),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _pick_city():
    """Pick a city with weighted tier distribution: T1 40%, T2 35%, T3 25%."""
    tier = np.random.choice(["T1", "T2", "T3"], p=[0.40, 0.35, 0.25])
    if tier == "T1":
        return random.choice(TIER1_CITIES), "Tier 1"
    elif tier == "T2":
        return random.choice(TIER2_CITIES), "Tier 2"
    else:
        return random.choice(TIER3_CITIES), "Tier 3"


def _generate_true_income(occupation: str) -> float:
    """Sample a realistic monthly income from a log-normal distribution
    bounded by occupation-specific range."""
    lo, hi = INCOME_RANGES[occupation]
    mu = np.log((lo + hi) / 2)
    sigma = 0.5
    income = np.random.lognormal(mu, sigma)
    return float(np.clip(income, lo, hi))


def _declared_income(true_income: float, occupation: str) -> float:
    """Simulate declared income — salaried is accurate; self-employed may
    under-declare; gig workers often don't have formal declarations."""
    if occupation == "salaried":
        # Small deviation
        return true_income * np.random.uniform(0.95, 1.05)
    elif occupation == "self_employed":
        # May under-declare by 10-40%
        return true_income * np.random.uniform(0.60, 1.0)
    else:
        # Gig workers — wider deviation
        return true_income * np.random.uniform(0.50, 1.10)


def _loan_product_interest(occupation: str, true_income: float, age: int):
    """Assign a primary loan product interest based on realistic heuristics."""
    weights = {
        "Personal Loan": 1.0,
        "Home Loan": 1.0,
        "Mortgage Loan": 1.0,
        "Auto Loan": 1.0,
    }

    # Income influences
    if true_income > 100_000:
        weights["Home Loan"] += 2.0
        weights["Auto Loan"] += 1.5
        weights["Mortgage Loan"] += 1.5
    elif true_income > 50_000:
        weights["Auto Loan"] += 1.0
        weights["Personal Loan"] += 1.5
    else:
        weights["Personal Loan"] += 3.0

    # Age influences
    if 25 <= age <= 35:
        weights["Home Loan"] += 1.5
        weights["Auto Loan"] += 1.0
    elif 35 <= age <= 50:
        weights["Mortgage Loan"] += 1.5
        weights["Home Loan"] += 1.0
    elif age > 50:
        weights["Personal Loan"] += 1.0

    # Occupation
    if occupation == "salaried":
        weights["Home Loan"] += 1.0
    elif occupation == "self_employed":
        weights["Mortgage Loan"] += 1.5
        weights["Personal Loan"] += 1.0

    products = list(weights.keys())
    probs = np.array([weights[p] for p in products])
    probs /= probs.sum()
    return np.random.choice(products, p=probs)


def _conversion_probability(
    true_income: float,
    occupation: str,
    age: int,
    city_tier: str,
    digital_engagement: float,
    existing_emi_ratio: float,
    cashflow_cv: float,
    loan_product: str,
) -> float:
    """Compute a realistic probability of conversion using a logistic-style
    formula driven by multiple factors. Base rate ≈ 10-12%."""
    logit = -3.5  # base (sigmoid(-3.5) ≈ 0.03, overall ~10-12% after signals)

    # Higher income → higher conversion
    if true_income > 150_000:
        logit += 0.4
    elif true_income > 75_000:
        logit += 0.2

    # Salaried converts more easily
    if occupation == "salaried":
        logit += 0.3
    elif occupation == "self_employed":
        logit += 0.05

    # Young professionals more likely for auto/personal
    if 25 <= age <= 40:
        logit += 0.15

    # Metro cities — higher conversion
    if city_tier == "Tier 1":
        logit += 0.2
    elif city_tier == "Tier 2":
        logit += 0.1

    # Digital engagement (0-1 scale) — strongest signal
    logit += digital_engagement * 1.2

    # Low existing EMI burden → more room for new loan
    if existing_emi_ratio < 0.2:
        logit += 0.3
    elif existing_emi_ratio < 0.35:
        logit += 0.1
    else:
        logit -= 0.4

    # Cash-flow stability (lower CV → more stable → better)
    if cashflow_cv < 0.15:
        logit += 0.25
    elif cashflow_cv < 0.30:
        logit += 0.05
    else:
        logit -= 0.3

    # Product-specific adjustments
    if loan_product == "Home Loan" and true_income > 80_000:
        logit += 0.2
    if loan_product == "Auto Loan" and 25 <= age <= 45:
        logit += 0.15

    prob = 1 / (1 + np.exp(-logit))
    return float(prob)


# ---------------------------------------------------------------------------
# Customer generation
# ---------------------------------------------------------------------------

def generate_customers(n: int = NUM_CUSTOMERS) -> pd.DataFrame:
    """Generate n synthetic customer profiles."""
    records = []
    for i in range(n):
        cust_id = f"IDBI{100000 + i}"
        name = fake.name()
        age = int(np.clip(np.random.normal(38, 10), 21, 65))
        city, city_tier = _pick_city()

        # Occupation
        occ = np.random.choice(
            list(OCCUPATION_DIST.keys()),
            p=list(OCCUPATION_DIST.values()),
        )

        true_income = _generate_true_income(occ)
        decl_income = _declared_income(true_income, occ)

        # Tenure with bank (months)
        tenure_months = int(np.clip(np.random.exponential(36), 3, 180))

        # Existing products (0-4)
        existing_products = int(np.clip(np.random.poisson(1.5), 0, 4))

        # Digital engagement score (0-1)
        if occ == "salaried":
            digital_eng = float(np.clip(np.random.beta(5, 3), 0, 1))
        elif occ == "gig_worker":
            digital_eng = float(np.clip(np.random.beta(6, 2), 0, 1))
        else:
            digital_eng = float(np.clip(np.random.beta(3, 4), 0, 1))

        loan_product = _loan_product_interest(occ, true_income, age)

        records.append({
            "customer_id": cust_id,
            "name": name,
            "age": age,
            "city": city,
            "city_tier": city_tier,
            "occupation": occ,
            "true_monthly_income": round(true_income, 2),
            "declared_monthly_income": round(decl_income, 2),
            "tenure_months": tenure_months,
            "existing_products": existing_products,
            "digital_engagement_score": round(digital_eng, 4),
            "loan_product_interest": loan_product,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Transaction generation
# ---------------------------------------------------------------------------

def generate_transactions(customers_df: pd.DataFrame) -> pd.DataFrame:
    """Generate 12 months of realistic transaction history for each customer."""
    all_txns = []
    base_date = datetime(2024, 1, 1)

    for _, cust in customers_df.iterrows():
        cid = cust["customer_id"]
        income = cust["true_monthly_income"]
        occ = cust["occupation"]

        for month_offset in range(MONTHS):
            month_start = base_date + timedelta(days=30 * month_offset)

            # --- Salary / Income Credits ---
            if occ == "salaried":
                # Regular salary on ~1st of month, small variation
                sal_day = month_start + timedelta(days=random.randint(0, 3))
                sal_amount = income * np.random.uniform(0.97, 1.03)
                all_txns.append({
                    "customer_id": cid,
                    "txn_date": sal_day.strftime("%Y-%m-%d"),
                    "txn_type": "credit",
                    "amount": round(sal_amount, 2),
                    "category": "Salary",
                    "merchant": cust.get("name", "Employer") + " Employer",
                    "channel": "NEFT",
                })
            elif occ == "self_employed":
                # Multiple business inflows, irregular
                n_inflows = random.randint(2, 6)
                for _ in range(n_inflows):
                    day = month_start + timedelta(days=random.randint(0, 29))
                    amt = (income / n_inflows) * np.random.uniform(0.3, 2.0)
                    all_txns.append({
                        "customer_id": cid,
                        "txn_date": day.strftime("%Y-%m-%d"),
                        "txn_type": "credit",
                        "amount": round(amt, 2),
                        "category": "Business_Income",
                        "merchant": fake.company(),
                        "channel": random.choice(["NEFT", "RTGS", "UPI"]),
                    })
            else:  # gig worker
                # Many small credits
                n_inflows = random.randint(5, 15)
                for _ in range(n_inflows):
                    day = month_start + timedelta(days=random.randint(0, 29))
                    amt = (income / n_inflows) * np.random.uniform(0.4, 1.8)
                    all_txns.append({
                        "customer_id": cid,
                        "txn_date": day.strftime("%Y-%m-%d"),
                        "txn_type": "credit",
                        "amount": round(amt, 2),
                        "category": "Gig_Payment",
                        "merchant": random.choice([
                            "Swiggy", "Zomato", "Uber", "Ola", "Freelancer",
                            "Upwork", "Urban Company", "Dunzo",
                        ]),
                        "channel": "UPI",
                    })

            # --- Regular Spends (debits) ---
            n_spends = random.randint(8, 25)
            monthly_spend_budget = income * np.random.uniform(0.35, 0.65)
            for _ in range(n_spends):
                day = month_start + timedelta(days=random.randint(0, 29))
                cat = random.choice(MERCHANT_CATEGORIES[:10])  # everyday categories
                amt = (monthly_spend_budget / n_spends) * np.random.uniform(0.2, 2.5)
                all_txns.append({
                    "customer_id": cid,
                    "txn_date": day.strftime("%Y-%m-%d"),
                    "txn_type": "debit",
                    "amount": round(amt, 2),
                    "category": cat,
                    "merchant": fake.company(),
                    "channel": random.choice(["UPI", "Debit Card", "Net Banking"]),
                })

            # --- Existing EMI debits (if any) ---
            if cust["existing_products"] > 0:
                n_emis = min(cust["existing_products"], random.randint(1, 3))
                for emi_idx in range(n_emis):
                    emi_day = month_start + timedelta(days=random.randint(5, 10))
                    emi_amt = income * np.random.uniform(0.05, 0.15)
                    all_txns.append({
                        "customer_id": cid,
                        "txn_date": emi_day.strftime("%Y-%m-%d"),
                        "txn_type": "debit",
                        "amount": round(emi_amt, 2),
                        "category": "EMI",
                        "merchant": random.choice([
                            "HDFC EMI", "SBI Card EMI", "Bajaj Finance EMI",
                            "IDBI Loan EMI", "ICICI EMI",
                        ]),
                        "channel": "Auto_Debit",
                    })

            # --- Occasional big-ticket purchase (intent signal) ---
            if random.random() < 0.08:  # ~8% chance per month
                day = month_start + timedelta(days=random.randint(0, 29))
                product = cust["loan_product_interest"]
                if product == "Auto Loan":
                    cat = "Auto_Dealer"
                    amt = np.random.uniform(50_000, 500_000)
                elif product == "Home Loan":
                    cat = "Real_Estate_Broker"
                    amt = np.random.uniform(100_000, 1_000_000)
                elif product == "Personal Loan":
                    cat = random.choice(["Wedding_Services", "Education", "Travel"])
                    amt = np.random.uniform(20_000, 200_000)
                else:  # Mortgage
                    cat = "Home_Improvement"
                    amt = np.random.uniform(50_000, 300_000)
                all_txns.append({
                    "customer_id": cid,
                    "txn_date": day.strftime("%Y-%m-%d"),
                    "txn_type": "debit",
                    "amount": round(amt, 2),
                    "category": cat,
                    "merchant": fake.company(),
                    "channel": random.choice(["Debit Card", "Net Banking", "UPI"]),
                })

    return pd.DataFrame(all_txns)


# ---------------------------------------------------------------------------
# Behavioral signals generation
# ---------------------------------------------------------------------------

def generate_behavioral_signals(customers_df: pd.DataFrame) -> pd.DataFrame:
    """Generate per-customer-per-month behavioral/digital engagement signals."""
    records = []
    base_date = datetime(2024, 1, 1)

    for _, cust in customers_df.iterrows():
        cid = cust["customer_id"]
        eng = cust["digital_engagement_score"]
        product = cust["loan_product_interest"]

        for month_offset in range(MONTHS):
            month_label = (base_date + timedelta(days=30 * month_offset)).strftime("%Y-%m")

            # App logins — correlated with engagement score
            app_logins = int(np.clip(np.random.poisson(eng * 20), 0, 60))

            # EMI calculator usage
            emi_calc_views = int(np.random.poisson(eng * 3))

            # Loan page visits
            loan_page_visits = int(np.random.poisson(eng * 4))

            # Product-specific page visits
            product_page_visits = int(np.random.poisson(eng * 2)) if random.random() < 0.5 else 0

            records.append({
                "customer_id": cid,
                "month": month_label,
                "app_logins": app_logins,
                "emi_calculator_views": emi_calc_views,
                "loan_page_visits": loan_page_visits,
                "product_page_visits": product_page_visits,
                "viewed_product": product if product_page_visits > 0 else "",
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Conversion labels (ground truth)
# ---------------------------------------------------------------------------

def assign_conversion_labels(
    customers_df: pd.DataFrame,
    txns_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute conversion probability and sample binary label.

    Uses transaction-derived features that directly match what the ML model
    will learn from, ensuring strong feature→label correlation.
    """
    # Pre-compute per-customer transaction aggregates
    credit_txns = txns_df[txns_df["txn_type"] == "credit"].copy()
    debit_txns = txns_df[txns_df["txn_type"] == "debit"].copy()

    # Monthly credit totals + CV
    credit_txns["month"] = pd.to_datetime(credit_txns["txn_date"]).dt.to_period("M")
    credit_monthly = (
        credit_txns
        .groupby(["customer_id", "month"])["amount"]
        .sum()
        .reset_index()
    )
    credit_stats = (
        credit_monthly
        .groupby("customer_id")["amount"]
        .agg(credit_mean="mean", credit_std="std")
        .fillna(0)
    )
    credit_stats["cv"] = credit_stats["credit_std"] / credit_stats["credit_mean"].replace(0, 1)

    # Existing EMI ratio
    emi_debits = debit_txns[debit_txns["category"] == "EMI"].copy()
    emi_debits["month"] = pd.to_datetime(emi_debits["txn_date"]).dt.to_period("M")
    monthly_emi = emi_debits.groupby("customer_id")["amount"].sum() / 12

    # Big-ticket spend flags (intent signals from transactions)
    intent_cats = {"Auto_Dealer", "Real_Estate_Broker", "Wedding_Services",
                   "Home_Improvement", "Education", "Jewellery"}
    intent_spends = debit_txns[debit_txns["category"].isin(intent_cats)]
    has_intent_spend = set(intent_spends["customer_id"].unique())

    # Total debit amount per customer (spending power proxy)
    total_debit = debit_txns.groupby("customer_id")["amount"].sum()

    labels = []
    for _, cust in customers_df.iterrows():
        cid = cust["customer_id"]
        income = cust["true_monthly_income"]
        eng = cust["digital_engagement_score"]

        # Cash-flow CV (from actual transactions)
        cashflow_cv = credit_stats.loc[cid, "cv"] if cid in credit_stats.index else 0.3

        # EMI ratio (from actual transactions)
        emi = monthly_emi.get(cid, 0)
        emi_ratio = emi / income if income > 0 else 0

        # Spending power (percentile-based signal)
        spend = total_debit.get(cid, 0)

        # ---- Conversion probability: ~10-12% base, >30% for best leads ----
        logit = -4.5  # base: sigmoid(-4.5) ≈ 1.1%

        # Digital engagement — reward HIGH engagement strongly
        # eng ranges 0-1 with mean ~0.5; only top quartile gets real boost
        if eng > 0.75:
            logit += 1.8   # strong signal for highly engaged
        elif eng > 0.60:
            logit += 0.8
        elif eng > 0.45:
            logit += 0.3
        elif eng < 0.25:
            logit -= 0.5

        # Cash-flow stability: reward very stable, penalize unstable
        if cashflow_cv < 0.10:
            logit += 1.2
        elif cashflow_cv < 0.18:
            logit += 0.5
        elif cashflow_cv > 0.35:
            logit -= 0.8

        # EMI capacity: low existing burden is a strong signal
        if emi_ratio < 0.05:
            logit += 0.8
        elif emi_ratio < 0.15:
            logit += 0.3
        elif emi_ratio > 0.35:
            logit -= 1.0

        # Big-ticket intent spend from transactions
        if cid in has_intent_spend:
            logit += 0.7

        # Income level
        if income > 180_000:
            logit += 0.4
        elif income < 25_000:
            logit -= 0.5

        # Occupation
        if cust["occupation"] == "salaried":
            logit += 0.4
        elif cust["occupation"] == "gig_worker":
            logit -= 0.3

        # Tenure with bank
        tenure = cust["tenure_months"]
        if tenure > 60:
            logit += 0.3
        elif tenure < 12:
            logit -= 0.4

        # City tier
        if cust["city_tier"] == "Tier 1":
            logit += 0.2

        prob = 1 / (1 + np.exp(-logit))
        converted = int(np.random.random() < prob)
        labels.append({
            "customer_id": cid,
            "conversion_probability": round(prob, 4),
            "loan_converted": converted,
            "cashflow_cv": round(float(cashflow_cv), 4),
            "existing_emi_ratio": round(float(emi_ratio), 4),
        })

    return pd.DataFrame(labels)


# ---------------------------------------------------------------------------
# SQLite storage
# ---------------------------------------------------------------------------

def store_to_sqlite(
    customers_df: pd.DataFrame,
    txns_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    db_path: str = DB_PATH,
):
    """Write all DataFrames to SQLite database."""
    # Merge labels into customers
    customers_full = customers_df.merge(labels_df, on="customer_id", how="left")

    conn = sqlite3.connect(db_path)
    customers_full.to_sql("customers", conn, if_exists="replace", index=False)
    txns_df.to_sql("transactions", conn, if_exists="replace", index=False)
    behavior_df.to_sql("behavioral_signals", conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ Data stored to SQLite: {db_path}")
    return customers_full


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(customers_full: pd.DataFrame, txns_df: pd.DataFrame):
    """Print summary statistics to validate realism."""
    print("\n" + "=" * 70)
    print("📊 IntelliLend — Synthetic Data Summary")
    print("=" * 70)

    print(f"\nTotal Customers:      {len(customers_full):,}")
    print(f"Total Transactions:   {len(txns_df):,}")
    print(f"Months of History:    {MONTHS}")

    print(f"\n--- Income Distribution (Monthly, ₹) ---")
    for occ in ["salaried", "self_employed", "gig_worker"]:
        subset = customers_full[customers_full["occupation"] == occ]
        print(f"  {occ:15s}:  median ₹{subset['true_monthly_income'].median():>10,.0f}  |  "
              f"mean ₹{subset['true_monthly_income'].mean():>10,.0f}  |  "
              f"n={len(subset)}")

    print(f"\n--- Occupation Distribution ---")
    print(customers_full["occupation"].value_counts().to_string())

    print(f"\n--- City Tier Distribution ---")
    print(customers_full["city_tier"].value_counts().to_string())

    print(f"\n--- Loan Product Interest ---")
    print(customers_full["loan_product_interest"].value_counts().to_string())

    conv_rate = customers_full["loan_converted"].mean() * 100
    print(f"\n--- Conversion Metrics ---")
    print(f"  Base conversion rate: {conv_rate:.1f}%")
    print(f"  Total converted:      {customers_full['loan_converted'].sum()}")
    print(f"  Total not converted:  {(1 - customers_full['loan_converted']).sum():.0f}")

    print(f"\n--- Income Declaration Deviation ---")
    customers_full["income_deviation_pct"] = (
        (customers_full["declared_monthly_income"] - customers_full["true_monthly_income"])
        / customers_full["true_monthly_income"]
        * 100
    )
    print(f"  Mean deviation: {customers_full['income_deviation_pct'].mean():.1f}%")
    print(f"  Customers with >20% under-declaration: "
          f"{(customers_full['income_deviation_pct'] < -20).sum()}")

    print("\n" + "=" * 70)
    print("✅ Data generation complete. All data is synthetic and reproducible (seed=42).")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("🚀 IntelliLend — Generating Synthetic Data...")
    print(f"   Customers: {NUM_CUSTOMERS}")
    print(f"   Months:    {MONTHS}")
    print(f"   Seed:      {SEED}")

    # Step 1: Customers
    print("\n[1/5] Generating customer profiles...")
    customers_df = generate_customers()

    # Step 2: Transactions
    print("[2/5] Generating transaction histories (this may take a moment)...")
    txns_df = generate_transactions(customers_df)

    # Step 3: Behavioral signals
    print("[3/5] Generating behavioral signals...")
    behavior_df = generate_behavioral_signals(customers_df)

    # Step 4: Conversion labels
    print("[4/5] Computing conversion labels...")
    labels_df = assign_conversion_labels(customers_df, txns_df)

    # Step 5: Store
    print("[5/5] Storing to SQLite & CSV...")
    customers_full = store_to_sqlite(customers_df, txns_df, behavior_df, labels_df)

    # Export sample CSV
    customers_full.to_csv(CSV_PATH, index=False)
    print(f"✅ Sample CSV exported: {CSV_PATH}")

    # Summary
    print_summary(customers_full, txns_df)

    return customers_full, txns_df, behavior_df


if __name__ == "__main__":
    main()
