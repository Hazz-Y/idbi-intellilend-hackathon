"""
IntelliLend — Streamlit Dashboard
===================================
RM / Underwriter-facing dashboard for IDBI Bank's AI-driven lead
intelligence and alternate income assessment system.

Views:
  1. KPI Overview Cards
  2. Lead Leaderboard (sortable, filterable)
  3. Customer Detail Drilldown (SHAP chart, income estimate, risk flags)
  4. Portfolio Analytics (Before vs After IntelliLend)
  5. What-If Simulator (stretch goal)

Run: streamlit run app/dashboard.py
"""

import os
import sys
import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Auto-run pipeline if models don't exist (for cloud deployment)
if not os.path.exists(os.path.join(PROJECT_ROOT, "models", "pipeline_output.pkl")):
    import subprocess
    print("Pipeline output not found. Running pipeline in fast mode...")
    subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "run_pipeline.py"), "--fast"], check=True)



# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IDBI IntelliLend — AI Lead Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# IDBI Brand Colors & Custom CSS
# ---------------------------------------------------------------------------
IDBI_MAROON = "#9B1B30"
IDBI_DARK_MAROON = "#7A1526"
IDBI_LIGHT_MAROON = "#C7354D"
IDBI_GOLD = "#D4A843"
IDBI_WHITE = "#FAFAFA"
IDBI_BG_DARK = "#0E1117"
IDBI_BG_CARD = "#1A1D29"
IDBI_SUCCESS = "#00D26A"
IDBI_WARNING = "#FFAA00"
IDBI_DANGER = "#FF4444"

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* Hide default Streamlit header/footer */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* Top header bar */
    .idbi-header {{
        background: linear-gradient(135deg, {IDBI_MAROON} 0%, {IDBI_DARK_MAROON} 100%);
        padding: 1.2rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 8px 32px rgba(155, 27, 48, 0.3);
    }}
    .idbi-header h1 {{
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }}
    .idbi-header .subtitle {{
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
        font-weight: 400;
    }}
    .idbi-header .bank-badge {{
        background: rgba(255,255,255,0.15);
        padding: 0.4rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: 500;
        font-size: 0.85rem;
        backdrop-filter: blur(10px);
    }}

    /* KPI Cards */
    .kpi-container {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }}
    .kpi-card {{
        background: linear-gradient(145deg, {IDBI_BG_CARD} 0%, #222538 100%);
        border: 1px solid rgba(155, 27, 48, 0.3);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    .kpi-card:hover {{
        transform: translateY(-3px);
        border-color: {IDBI_MAROON};
        box-shadow: 0 8px 25px rgba(155, 27, 48, 0.2);
    }}
    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, {IDBI_MAROON}, {IDBI_GOLD});
    }}
    .kpi-label {{
        color: rgba(250, 250, 250, 0.6);
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }}
    .kpi-value {{
        color: {IDBI_WHITE};
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
    }}
    .kpi-delta {{
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 0.3rem;
    }}
    .kpi-delta.positive {{ color: {IDBI_SUCCESS}; }}
    .kpi-delta.negative {{ color: {IDBI_DANGER}; }}

    /* Section headers */
    .section-header {{
        color: {IDBI_WHITE};
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {IDBI_MAROON};
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    /* Data tables */
    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
    }}

    /* Before/After comparison */
    .comparison-card {{
        background: {IDBI_BG_CARD};
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }}
    .comparison-value {{
        font-size: 3rem;
        font-weight: 800;
    }}
    .comparison-label {{
        font-size: 0.85rem;
        color: rgba(250,250,250,0.6);
        margin-top: 0.3rem;
    }}

    /* Risk flag badges */
    .risk-flag {{
        background: rgba(255, 170, 0, 0.15);
        border: 1px solid rgba(255, 170, 0, 0.3);
        color: {IDBI_WARNING};
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        font-size: 0.82rem;
        margin: 0.2rem;
        display: inline-block;
    }}

    /* Confidence badges */
    .badge-high {{
        background: rgba(0, 210, 106, 0.15);
        color: {IDBI_SUCCESS};
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }}
    .badge-medium {{
        background: rgba(255, 170, 0, 0.15);
        color: {IDBI_WARNING};
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }}
    .badge-low {{
        background: rgba(255, 68, 68, 0.15);
        color: {IDBI_DANGER};
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }}

    /* Footer */
    .idbi-footer {{
        text-align: center;
        color: rgba(250,250,250,0.3);
        font-size: 0.75rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }}

    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #111320 0%, {IDBI_BG_DARK} 100%);
    }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label {{
        color: rgba(250,250,250,0.8);
        font-weight: 500;
    }}

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        color: rgba(250,250,250,0.6);
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background: {IDBI_MAROON} !important;
        color: white !important;
    }}

    /* Metric cards in streamlit */
    [data-testid="stMetricValue"] {{
        font-weight: 700;
    }}
</style>
"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_pipeline_data():
    """Load pre-computed pipeline data from SQLite and model artifacts."""
    db_path = os.path.join(PROJECT_ROOT, "data", "intellilend.db")
    models_dir = os.path.join(PROJECT_ROOT, "models")

    # Check if pipeline has been run
    if not os.path.exists(db_path):
        return None, None

    pipeline_path = os.path.join(models_dir, "pipeline_output.pkl")
    metrics_path_lead = os.path.join(models_dir, "lead_scoring_metrics.json")
    metrics_path_income = os.path.join(models_dir, "income_estimation_metrics.json")

    data = {}

    # Load pipeline output (full scored DataFrame)
    if os.path.exists(pipeline_path):
        with open(pipeline_path, "rb") as f:
            data["pipeline_df"] = pickle.load(f)
    else:
        # Fallback: load from SQLite
        conn = sqlite3.connect(db_path)
        data["pipeline_df"] = pd.read_sql("SELECT * FROM customers", conn)
        conn.close()

    # Load metrics
    metrics = {}
    if os.path.exists(metrics_path_lead):
        with open(metrics_path_lead) as f:
            metrics["lead_scoring"] = json.load(f)
    if os.path.exists(metrics_path_income):
        with open(metrics_path_income) as f:
            metrics["income_estimation"] = json.load(f)

    data["metrics"] = metrics
    return data, True


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div class="idbi-header">
        <div>
            <h1>🏦 IDBI IntelliLend</h1>
            <div class="subtitle">AI-Driven Lead Intelligence & Alternate Income Assessment Engine</div>
        </div>
        <div class="bank-badge">⚡ Powered by ML • SHAP Explainability</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------

def render_kpi_cards(df: pd.DataFrame, metrics: dict):
    """Render top-level KPI cards."""
    total_leads = len(df)
    avg_lead_score = df["lead_score"].mean() if "lead_score" in df.columns else 0
    base_conv = metrics.get("lead_scoring", {}).get("base_conversion_rate", 10)

    # Simulated conversion rate at top 20%
    conv_sim = metrics.get("lead_scoring", {}).get("conversion_rate_simulation", {})
    projected_conv = conv_sim.get("top_20pct", 35)

    # Pre-qualified count
    prequalified = (df["recommendation"] == "Pre-qualified").sum() if "recommendation" in df.columns else 0

    # Average turnaround time saved (simulated: from 48hrs manual to 2hrs with ML)
    tat_saved_hrs = 46  # hours
    tat_saved_pct = round((tat_saved_hrs / 48) * 100, 0)

    # Cost per lead reduction (simulated: from ₹800 blanket to ₹250 targeted)
    cost_baseline = 800
    cost_intellilend = 250
    cost_saved_pct = round((1 - cost_intellilend / cost_baseline) * 100, 0)

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Total Leads Scored</div>
            <div class="kpi-value">{total_leads:,}</div>
            <div class="kpi-delta positive">100% coverage of customer base</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Avg Lead Quality Score</div>
            <div class="kpi-value">{avg_lead_score:.1f}</div>
            <div class="kpi-delta positive">↑ ML-calibrated scores (0–100)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Projected Conversion Rate</div>
            <div class="kpi-value">{projected_conv:.1f}%</div>
            <div class="kpi-delta positive">↑ {projected_conv - base_conv:.1f}pp vs baseline {base_conv}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Pre-Qualified Leads</div>
            <div class="kpi-value">{prequalified:,}</div>
            <div class="kpi-delta positive">Ready for RM follow-up</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Turnaround Time Saved</div>
            <div class="kpi-value">{tat_saved_pct:.0f}%</div>
            <div class="kpi-delta positive">↓ {tat_saved_hrs}hrs avg reduction</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Cost-per-Lead Reduction</div>
            <div class="kpi-value">{cost_saved_pct:.0f}%</div>
            <div class="kpi-delta positive">↓ ₹{cost_baseline - cost_intellilend} saved per lead</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame):
    """Render sidebar filters and return filtered dataframe."""
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding: 1rem 0;">
            <span style="font-size: 2rem;">🏦</span><br>
            <span style="color: {IDBI_MAROON}; font-weight: 700; font-size: 1.1rem;">IDBI IntelliLend</span><br>
            <span style="color: rgba(250,250,250,0.5); font-size: 0.75rem;">RM Dashboard v1.0</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### 🔍 Filters")

        # Loan product filter
        products = ["All"] + sorted(df["loan_product_interest"].unique().tolist())
        selected_product = st.selectbox("Loan Product", products, index=0)

        # Recommendation filter
        recs = ["All"] + sorted(df["recommendation"].unique().tolist()) if "recommendation" in df.columns else ["All"]
        selected_rec = st.selectbox("Recommendation", recs, index=0)

        # City tier filter
        tiers = ["All"] + sorted(df["city_tier"].unique().tolist())
        selected_tier = st.selectbox("City Tier", tiers, index=0)

        # Occupation filter
        occs = ["All"] + sorted(df["occupation"].unique().tolist())
        selected_occ = st.selectbox("Occupation", occs, index=0)

        # Lead score range
        min_score, max_score = st.slider(
            "Lead Score Range",
            min_value=0, max_value=100,
            value=(0, 100),
        )

        # Income confidence
        conf_options = ["All", "High", "Medium", "Low"]
        selected_conf = st.selectbox("Income Confidence", conf_options, index=0)

        st.markdown("---")
        st.markdown(f"""
        <div style="text-align:center; color: rgba(250,250,250,0.4); font-size: 0.7rem; padding: 1rem 0;">
            ⚠️ All data shown is synthetic and<br>
            generated for demonstration purposes.<br><br>
            Built for Smart India Hackathon<br>
            Problem Statement by IDBI Bank
        </div>
        """, unsafe_allow_html=True)

    # Apply filters
    filtered = df.copy()
    if selected_product != "All":
        filtered = filtered[filtered["loan_product_interest"] == selected_product]
    if selected_rec != "All" and "recommendation" in filtered.columns:
        filtered = filtered[filtered["recommendation"] == selected_rec]
    if selected_tier != "All":
        filtered = filtered[filtered["city_tier"] == selected_tier]
    if selected_occ != "All":
        filtered = filtered[filtered["occupation"] == selected_occ]
    if "lead_score" in filtered.columns:
        filtered = filtered[
            (filtered["lead_score"] >= min_score) &
            (filtered["lead_score"] <= max_score)
        ]
    if selected_conf != "All" and "income_confidence_band" in filtered.columns:
        filtered = filtered[filtered["income_confidence_band"] == selected_conf]

    return filtered


# ---------------------------------------------------------------------------
# Tab 1: Lead Leaderboard
# ---------------------------------------------------------------------------

def render_leaderboard(df: pd.DataFrame):
    """Render the lead leaderboard with ranking and key metrics."""
    st.markdown('<div class="section-header">🏆 Lead Leaderboard — Top Prospects</div>', unsafe_allow_html=True)

    display_cols = [
        "customer_id", "name", "lead_score", "loan_readiness_score",
        "loan_product_interest", "estimated_income", "income_confidence_band",
        "eligible_loan_amount", "recommendation", "risk_flag_count",
        "city", "occupation",
    ]
    available_cols = [c for c in display_cols if c in df.columns]
    display_df = df[available_cols].sort_values("lead_score", ascending=False).head(100)

    # Format currency columns
    currency_cols = ["estimated_income", "eligible_loan_amount"]
    for col in currency_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "N/A")

    # Format score columns
    score_cols = ["lead_score", "loan_readiness_score"]
    for col in score_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")

    # Rename columns for display
    rename_map = {
        "customer_id": "ID",
        "name": "Customer",
        "lead_score": "Lead Score",
        "loan_readiness_score": "Readiness",
        "loan_product_interest": "Product",
        "estimated_income": "Est. Income",
        "income_confidence_band": "Confidence",
        "eligible_loan_amount": "Eligible Amt",
        "recommendation": "Status",
        "risk_flag_count": "⚠️ Flags",
        "city": "City",
        "occupation": "Occupation",
    }
    display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})

    st.dataframe(
        display_df.reset_index(drop=True),
        use_container_width=True,
        height=500,
    )

    # Summary stats below table
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Showing", f"{len(display_df)} leads")
    with col2:
        if "lead_score" in df.columns:
            st.metric("Avg Score", f"{df['lead_score'].mean():.1f}")
    with col3:
        if "recommendation" in df.columns:
            pq = (df["recommendation"] == "Pre-qualified").sum()
            st.metric("Pre-qualified", f"{pq}")
    with col4:
        if "eligible_loan_amount" in df.columns:
            total_eligible = df[df["eligible_loan_amount"] > 0]["eligible_loan_amount"].sum()
            if total_eligible > 1e7:
                st.metric("Total Eligible", f"₹{total_eligible/1e7:.1f} Cr")
            else:
                st.metric("Total Eligible", f"₹{total_eligible/1e5:.1f} L")


# ---------------------------------------------------------------------------
# Tab 2: Customer Detail Drilldown
# ---------------------------------------------------------------------------

def render_customer_detail(df: pd.DataFrame):
    """Render detailed view for a selected customer."""
    st.markdown('<div class="section-header">🔎 Customer Detail Drilldown</div>', unsafe_allow_html=True)

    # Customer selector
    customer_options = df.sort_values("lead_score", ascending=False)["customer_id"].tolist()
    if not customer_options:
        st.warning("No customers match the current filters.")
        return

    selected_id = st.selectbox(
        "Select Customer",
        customer_options,
        format_func=lambda x: f"{x} — {df[df['customer_id']==x]['name'].iloc[0]}" if len(df[df['customer_id']==x]) > 0 else x,
    )

    cust = df[df["customer_id"] == selected_id].iloc[0]

    # --- Top info cards ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        score = cust.get("lead_score", 0)
        score_color = IDBI_SUCCESS if score >= 65 else (IDBI_WARNING if score >= 40 else IDBI_DANGER)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Lead Quality Score</div>
            <div class="kpi-value" style="color: {score_color}">{score:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        readiness = cust.get("loan_readiness_score", 0)
        r_color = IDBI_SUCCESS if readiness >= 65 else (IDBI_WARNING if readiness >= 40 else IDBI_DANGER)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Loan Readiness</div>
            <div class="kpi-value" style="color: {r_color}">{readiness:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        income = cust.get("estimated_income", 0)
        conf = cust.get("income_confidence_band", "Medium")
        badge_class = f"badge-{conf.lower()}"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Estimated Monthly Income</div>
            <div class="kpi-value">₹{income:,.0f}</div>
            <div class="kpi-delta"><span class="{badge_class}">{conf} Confidence</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        eligible = cust.get("eligible_loan_amount", 0)
        product = cust.get("loan_product_interest", "N/A")
        rec = cust.get("recommendation", "N/A")
        rec_color = IDBI_SUCCESS if rec == "Pre-qualified" else (IDBI_WARNING if rec == "Review Required" else IDBI_DANGER)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Eligible Amount ({product})</div>
            <div class="kpi-value">₹{eligible:,.0f}</div>
            <div class="kpi-delta" style="color: {rec_color}">{rec}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Two-column layout ---
    left_col, right_col = st.columns([3, 2])

    with left_col:
        # SHAP waterfall chart
        st.markdown("#### 📊 Score Drivers (SHAP Analysis)")
        shap_dict = cust.get("shap_values_dict")
        if isinstance(shap_dict, dict) and shap_dict:
            sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
            features_list = [s[0].replace("_", " ").title() for s in sorted_shap]
            values_list = [s[1] for s in sorted_shap]
            colors = [IDBI_SUCCESS if v > 0 else IDBI_DANGER for v in values_list]

            fig = go.Figure(go.Bar(
                x=values_list,
                y=features_list,
                orientation='h',
                marker_color=colors,
                text=[f"{v:+.3f}" for v in values_list],
                textposition="outside",
                textfont=dict(size=11),
            ))
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=60, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=IDBI_WHITE, size=12),
                xaxis=dict(title="SHAP Value (Impact on Score)", gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SHAP values not available for this customer.")

    with right_col:
        # Customer profile
        st.markdown("#### 👤 Customer Profile")
        profile_data = {
            "Name": cust.get("name", "N/A"),
            "Age": f"{cust.get('age', 'N/A')} years",
            "City": f"{cust.get('city', 'N/A')} ({cust.get('city_tier', 'N/A')})",
            "Occupation": cust.get("occupation", "N/A").replace("_", " ").title(),
            "Tenure": f"{cust.get('tenure_months', 0)} months",
            "Existing Products": cust.get("existing_products", 0),
            "Declared Income": f"₹{cust.get('declared_monthly_income', 0):,.0f}",
            "Est. Income": f"₹{cust.get('estimated_income', 0):,.0f}",
            "Income Deviation": f"{cust.get('income_deviation_pct', 0):.1f}%",
            "EMI Ratio": f"{cust.get('emi_to_income_ratio', 0)*100:.1f}%",
            "Cash-flow CV": f"{cust.get('cashflow_cv', 0)*100:.1f}%",
            "Digital Engagement": f"{cust.get('digital_engagement_composite', 0):.2f}",
        }
        for k, v in profile_data.items():
            st.markdown(f"**{k}:** {v}")

        # Risk flags
        st.markdown("#### ⚠️ Risk Flags")
        flags = cust.get("risk_flags", [])
        if isinstance(flags, list) and flags:
            for flag in flags:
                st.markdown(f'<span class="risk-flag">⚠ {flag}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge-high">✓ No risk flags</span>', unsafe_allow_html=True)

    # --- Explanation ---
    st.markdown("#### 💡 AI Explanation")
    explanation = cust.get("explanation_summary", "")
    if explanation:
        st.success(f"**Why this score:** {explanation}")

    details = cust.get("explanation_details")
    if isinstance(details, list) and details:
        for d in details:
            icon = "🟢" if d["direction"] == "positive" else "🔴"
            st.markdown(f"{icon} {d['description']} (SHAP: {d['shap_value']:+.4f})")

    # --- Assign to RM button ---
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("✅ Assign to RM", type="primary", use_container_width=True):
            st.success(f"🎉 Lead {selected_id} assigned to Relationship Manager for follow-up!")
    with col2:
        if st.button("📋 Send for Review", use_container_width=True):
            st.info(f"📋 Lead {selected_id} sent to underwriter for detailed review.")


# ---------------------------------------------------------------------------
# Tab 3: Portfolio Analytics
# ---------------------------------------------------------------------------

def render_portfolio_analytics(df: pd.DataFrame, metrics: dict):
    """Render portfolio-level analytics and Before/After comparison."""
    st.markdown('<div class="section-header">📈 Portfolio Analytics</div>', unsafe_allow_html=True)

    # --- Before/After IntelliLend comparison ---
    st.markdown("### 🔄 Before vs. After IntelliLend")

    base_rate = metrics.get("lead_scoring", {}).get("base_conversion_rate", 10)
    conv_sim = metrics.get("lead_scoring", {}).get("conversion_rate_simulation", {})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="comparison-card" style="border: 2px solid rgba(255,68,68,0.3);">
            <div style="color: rgba(250,250,250,0.5); font-size: 0.8rem; margin-bottom: 0.5rem;">❌ BEFORE IntelliLend</div>
            <div class="comparison-value" style="color: {IDBI_DANGER}">{base_rate:.1f}%</div>
            <div class="comparison-label">Blanket Outreach Conversion</div>
            <div style="color: rgba(250,250,250,0.4); font-size: 0.75rem; margin-top: 0.8rem;">
                Random targeting • High cost-per-lead<br>
                Low RM productivity • No prioritization
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        top20_conv = conv_sim.get("top_20pct", 35)
        st.markdown(f"""
        <div class="comparison-card" style="border: 2px solid {IDBI_SUCCESS};">
            <div style="color: rgba(250,250,250,0.5); font-size: 0.8rem; margin-bottom: 0.5rem;">✅ AFTER IntelliLend (Top 20%)</div>
            <div class="comparison-value" style="color: {IDBI_SUCCESS}">{top20_conv:.1f}%</div>
            <div class="comparison-label">ML-Targeted Conversion</div>
            <div style="color: rgba(250,250,250,0.4); font-size: 0.75rem; margin-top: 0.8rem;">
                AI-prioritized leads • Explainable scores<br>
                RM focus on best leads • >30% target exceeded
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        uplift = top20_conv - base_rate
        uplift_mult = top20_conv / base_rate if base_rate > 0 else 1
        st.markdown(f"""
        <div class="comparison-card" style="border: 2px solid {IDBI_GOLD};">
            <div style="color: rgba(250,250,250,0.5); font-size: 0.8rem; margin-bottom: 0.5rem;">📊 IMPACT</div>
            <div class="comparison-value" style="color: {IDBI_GOLD}">{uplift_mult:.1f}x</div>
            <div class="comparison-label">Conversion Rate Uplift</div>
            <div style="color: rgba(250,250,250,0.4); font-size: 0.75rem; margin-top: 0.8rem;">
                +{uplift:.1f}pp improvement<br>
                ₹550 saved per lead • 96% TAT reduction
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Conversion rate at various cutoffs chart ---
    st.markdown("### 📊 Conversion Rate by Lead Score Cutoff")
    if conv_sim:
        cutoffs = [int(k.replace("top_", "").replace("pct", "")) for k in conv_sim.keys()]
        rates = list(conv_sim.values())

        fig_conv = go.Figure()
        fig_conv.add_trace(go.Bar(
            x=[f"Top {c}%" for c in cutoffs],
            y=rates,
            marker_color=[IDBI_SUCCESS if r >= 30 else IDBI_WARNING for r in rates],
            text=[f"{r:.1f}%" for r in rates],
            textposition="outside",
            textfont=dict(size=13, color=IDBI_WHITE),
        ))
        fig_conv.add_hline(y=30, line_dash="dash", line_color=IDBI_MAROON,
                          annotation_text="30% Target", annotation_font_color=IDBI_MAROON)
        fig_conv.add_hline(y=base_rate, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                          annotation_text=f"Baseline {base_rate}%", annotation_font_color="rgba(255,255,255,0.5)")
        fig_conv.update_layout(
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=IDBI_WHITE),
            yaxis=dict(title="Conversion Rate (%)", gridcolor="rgba(255,255,255,0.1)"),
            xaxis=dict(title="Lead Score Percentile (calling top N%)"),
            showlegend=False,
        )
        st.plotly_chart(fig_conv, use_container_width=True)

    # --- Distribution charts ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Lead Score Distribution")
        if "lead_score" in df.columns:
            fig_dist = px.histogram(
                df, x="lead_score", nbins=30,
                color_discrete_sequence=[IDBI_MAROON],
                labels={"lead_score": "Lead Score", "count": "Customers"},
            )
            fig_dist.update_layout(
                height=300,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=IDBI_WHITE),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        st.markdown("### 🏷️ Recommendation Breakdown")
        if "recommendation" in df.columns:
            rec_counts = df["recommendation"].value_counts()
            colors_map = {
                "Pre-qualified": IDBI_SUCCESS,
                "Review Required": IDBI_WARNING,
                "Not Recommended": IDBI_DANGER,
            }
            fig_rec = go.Figure(go.Pie(
                labels=rec_counts.index,
                values=rec_counts.values,
                marker=dict(colors=[colors_map.get(r, "#888") for r in rec_counts.index]),
                textinfo="percent+label",
                textfont=dict(size=13),
                hole=0.4,
            ))
            fig_rec.update_layout(
                height=300,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=IDBI_WHITE),
                showlegend=False,
            )
            st.plotly_chart(fig_rec, use_container_width=True)

    # --- Segment breakdown ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👥 Lead Quality by Loan Product")
        if "lead_score" in df.columns and "loan_product_interest" in df.columns:
            prod_stats = df.groupby("loan_product_interest")["lead_score"].agg(["mean", "count"]).reset_index()
            prod_stats.columns = ["Product", "Avg Score", "Count"]
            fig_prod = px.bar(
                prod_stats, x="Product", y="Avg Score",
                color="Avg Score", color_continuous_scale=["#FF4444", "#FFAA00", "#00D26A"],
                text="Count",
            )
            fig_prod.update_layout(
                height=300,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=IDBI_WHITE),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)", title="Avg Lead Score"),
                coloraxis_showscale=False,
            )
            fig_prod.update_traces(texttemplate="%{text} leads", textposition="outside")
            st.plotly_chart(fig_prod, use_container_width=True)

    with col2:
        st.markdown("### 📍 Lead Quality by City Tier")
        if "lead_score" in df.columns and "city_tier" in df.columns:
            tier_stats = df.groupby("city_tier")["lead_score"].agg(["mean", "count"]).reset_index()
            tier_stats.columns = ["Tier", "Avg Score", "Count"]
            fig_tier = px.bar(
                tier_stats, x="Tier", y="Avg Score",
                color="Avg Score", color_continuous_scale=["#FF4444", "#FFAA00", "#00D26A"],
                text="Count",
            )
            fig_tier.update_layout(
                height=300,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=IDBI_WHITE),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)", title="Avg Lead Score"),
                coloraxis_showscale=False,
            )
            fig_tier.update_traces(texttemplate="%{text} leads", textposition="outside")
            st.plotly_chart(fig_tier, use_container_width=True)

    # --- Income estimation summary ---
    st.markdown("### 💰 Income Confidence Distribution")
    if "income_confidence_band" in df.columns:
        conf_stats = df.groupby("income_confidence_band").agg(
            count=("customer_id", "count"),
            avg_income=("estimated_income", "mean"),
            avg_score=("lead_score", "mean") if "lead_score" in df.columns else ("customer_id", "count"),
        ).reset_index()

        col1, col2, col3 = st.columns(3)
        for i, band in enumerate(["High", "Medium", "Low"]):
            row = conf_stats[conf_stats["income_confidence_band"] == band]
            if len(row) > 0:
                with [col1, col2, col3][i]:
                    count = row["count"].iloc[0]
                    avg_inc = row["avg_income"].iloc[0]
                    badge = f"badge-{band.lower()}"
                    st.markdown(f"""
                    <div class="comparison-card">
                        <span class="{badge}">{band} Confidence</span>
                        <div class="comparison-value" style="font-size: 2rem; margin-top: 0.5rem;">{count:,}</div>
                        <div class="comparison-label">Customers</div>
                        <div style="color: rgba(250,250,250,0.5); font-size: 0.8rem; margin-top: 0.3rem;">
                            Avg Income: ₹{avg_inc:,.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 4: What-If Simulator
# ---------------------------------------------------------------------------

def render_what_if_simulator(df: pd.DataFrame):
    """Interactive what-if simulator for live demo moments."""
    st.markdown('<div class="section-header">🎛️ What-If Simulator</div>', unsafe_allow_html=True)
    st.markdown("*Adjust parameters below and watch the Loan Readiness Score update in real-time.*")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Adjust Customer Parameters")

        sim_income = st.slider("Monthly Income (₹)", 10000, 500000, 75000, step=5000)
        sim_emi = st.slider("Existing Monthly EMI (₹)", 0, 200000, 15000, step=1000)
        sim_cashflow_cv = st.slider("Cash-flow Variability (%)", 0, 80, 20) / 100
        sim_lead_score = st.slider("Lead Quality Score", 0, 100, 65)
        sim_confidence = st.selectbox("Income Confidence", ["High", "Medium", "Low"], index=0)
        sim_tenure = st.slider("Tenure with Bank (months)", 1, 120, 36)
        sim_product = st.selectbox("Loan Product", ["Personal Loan", "Home Loan", "Mortgage Loan", "Auto Loan"])

    with col2:
        st.markdown("#### 📊 Live Results")

        # Import underwriting functions
        from src.underwriting_engine import (
            compute_loan_readiness_score,
            compute_eligible_amount,
            LOAN_CONFIG,
        )

        emi_ratio = sim_emi / max(sim_income, 1)
        n_flags = sum([
            sim_cashflow_cv > 0.40,
            emi_ratio > 0.40,
            sim_confidence == "Low",
            sim_tenure < 6,
        ])

        readiness = compute_loan_readiness_score(
            lead_score=sim_lead_score,
            income_confidence=sim_confidence,
            emi_ratio=emi_ratio,
            cashflow_cv=sim_cashflow_cv,
            tenure_months=sim_tenure,
            n_risk_flags=n_flags,
        )

        eligible = compute_eligible_amount(sim_income, sim_emi, sim_product)

        # Recommendation
        if readiness >= 65 and n_flags <= 1 and eligible > 0:
            rec = "Pre-qualified ✅"
            rec_color = IDBI_SUCCESS
        elif readiness >= 40 and eligible > 0:
            rec = "Review Required ⚠️"
            rec_color = IDBI_WARNING
        else:
            rec = "Not Recommended ❌"
            rec_color = IDBI_DANGER

        r_color = IDBI_SUCCESS if readiness >= 65 else (IDBI_WARNING if readiness >= 40 else IDBI_DANGER)

        st.markdown(f"""
        <div class="kpi-card" style="margin-bottom: 1rem;">
            <div class="kpi-label">Loan Readiness Score</div>
            <div class="kpi-value" style="color: {r_color}; font-size: 3rem;">{readiness:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="kpi-card" style="margin-bottom: 1rem;">
            <div class="kpi-label">Eligible {sim_product} Amount</div>
            <div class="kpi-value">₹{eligible:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="kpi-card" style="margin-bottom: 1rem;">
            <div class="kpi-label">Recommendation</div>
            <div class="kpi-value" style="color: {rec_color}; font-size: 1.5rem;">{rec}</div>
        </div>
        """, unsafe_allow_html=True)

        # FOIR breakdown
        config = LOAN_CONFIG.get(sim_product, {})
        max_foir = config.get("max_foir", 0.5) * 100
        available_emi = sim_income * config.get("max_foir", 0.5) - sim_emi

        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">FOIR Analysis</div>
            <div style="color: rgba(250,250,250,0.7); font-size: 0.85rem; margin-top: 0.3rem;">
                Current EMI Ratio: <b>{emi_ratio*100:.1f}%</b> (Max FOIR: {max_foir:.0f}%)<br>
                Available EMI Capacity: <b>₹{max(available_emi, 0):,.0f}/month</b><br>
                Risk Flags: <b>{n_flags}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

def render_footer():
    st.markdown("""
    <div class="idbi-footer">
        <strong>IDBI IntelliLend</strong> — AI-Driven Lead Intelligence & Alternate Income Assessment Engine<br>
        Built for Smart India Hackathon | Problem Statement by IDBI Bank<br>
        ⚠️ All data shown is synthetic and generated for demonstration purposes only.<br>
        Compliant with RBI Digital Lending Guidelines • Fair Practices Code • Consent-based Account Aggregator design<br>
        © 2024 IntelliLend Team
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    render_header()

    # Load data
    data, loaded = load_pipeline_data()

    if data is None or "pipeline_df" not in data:
        st.error("⚠️ Pipeline data not found. Please run the pipeline first:")
        st.code("python run_pipeline.py", language="bash")
        st.info("This will generate synthetic data, train models, and produce scored leads.")
        return

    df = data["pipeline_df"]
    metrics = data.get("metrics", {})

    # Ensure required columns exist with defaults
    if "lead_score" not in df.columns:
        df["lead_score"] = 50
    if "recommendation" not in df.columns:
        df["recommendation"] = "Review Required"
    if "loan_readiness_score" not in df.columns:
        df["loan_readiness_score"] = 50
    if "estimated_income" not in df.columns:
        df["estimated_income"] = df.get("declared_monthly_income", 50000)
    if "income_confidence_band" not in df.columns:
        df["income_confidence_band"] = "Medium"
    if "eligible_loan_amount" not in df.columns:
        df["eligible_loan_amount"] = 0
    if "risk_flag_count" not in df.columns:
        df["risk_flag_count"] = 0
    if "risk_flags" not in df.columns:
        df["risk_flags"] = [[] for _ in range(len(df))]

    # Render KPI cards
    render_kpi_cards(df, metrics)

    # Sidebar filters
    filtered_df = render_sidebar(df)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Lead Leaderboard",
        "🔎 Customer Drilldown",
        "📈 Portfolio Analytics",
        "🎛️ What-If Simulator",
    ])

    with tab1:
        render_leaderboard(filtered_df)

    with tab2:
        render_customer_detail(filtered_df)

    with tab3:
        render_portfolio_analytics(filtered_df, metrics)

    with tab4:
        render_what_if_simulator(filtered_df)

    # Footer
    render_footer()


if __name__ == "__main__":
    main()
