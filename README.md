# 🏦 IDBI IntelliLend

**AI-Driven Lead Intelligence & Alternate Income Assessment Engine**

> A 3-engine ML pipeline that transforms raw transaction and behavioral data into prioritized, high-conversion leads and defensible income estimates for IDBI Bank's retail lending division.

Built for **Smart India Hackathon** | Problem Statement by **IDBI Bank**

🌟 **[Live Demo — Streamlit App](https://your-app-url-here.streamlit.app/)** 🌟

---

## 🎯 Problem Statement

> *Bank's retail lending relies on traditional metrics, resulting in low conversions and limited insight into customer intent. A data-driven approach is needed to identify eligible, quantifiable repayment capacity, genuinely interested prospects using transaction and behavioral insights.*
>
> *Expected Outcome: Generate high-quality leads with conversion rate exceeding 30%, while enabling accurate assessment of borrowers' actual income levels for prudent underwriting.*

## ✅ Solution — IntelliLend

IntelliLend is a **3-engine AI pipeline** that:

| Engine | Purpose | Output |
|--------|---------|--------|
| 🎯 **Engine A** — Lead Scoring | Predicts propensity-to-convert per customer per loan product | Lead Quality Score (0-100) + SHAP explanation |
| 💰 **Engine B** — Income Assessment | Estimates actual disposable income from transaction patterns | Estimated income + Confidence Band (High/Medium/Low) |
| 📋 **Engine C** — Underwriting | Combines A + B into loan recommendations | Loan Readiness Score + Eligible Amount + Risk Flags |

### Key Results

| Metric | Value | Target |
|--------|-------|--------|
| Conversion Rate (Top 20%) | **>30%** | >30% ✅ |
| Income Estimation MAPE | **<5%** | Low error ✅ |
| Income Estimation R² | **>0.99** | High accuracy ✅ |
| Explainability | **100% of scores** have plain-English explanation | Auditable ✅ |
| Cost | **₹0** (100% open-source) | Free ✅ |

---

## 🏗️ Architecture

```
Transaction Data → Feature Engineering (35+ features)
    ├── Engine A: LightGBM Lead Scorer → Lead Score (0-100)
    ├── Engine B: XGBoost Income Estimator → Income + Confidence Band
    └── Engine C: Rule Engine → Loan Readiness + Eligible Amount
         └── RM Dashboard (Streamlit)
```

See [docs/architecture_diagram.md](docs/architecture_diagram.md) for detailed Mermaid diagrams.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-team/idbi-intellilend.git
cd idbi-intellilend

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (generates data, trains models, produces scores)
python run_pipeline.py

# Launch the dashboard
streamlit run app/dashboard.py
```

The dashboard will open at `http://localhost:8501`.

### One-Command Demo
```bash
pip install -r requirements.txt && python run_pipeline.py && streamlit run app/dashboard.py
```

---

## 📁 Project Structure

```
idbi-intellilend/
├── README.md                       # This file
├── requirements.txt                # Python dependencies (all free/open-source)
├── run_pipeline.py                 # Master pipeline orchestrator
├── data/
│   ├── generate_synthetic_data.py  # Faker-based data generator (seeded, reproducible)
│   ├── intellilend.db              # SQLite database (generated)
│   ├── sample_data.csv             # Customer profiles export
│   └── pipeline_summary.csv        # Full pipeline output summary
├── src/
│   ├── features.py                 # Feature engineering (7 feature groups, 35+ features)
│   ├── lead_scoring_model.py       # Engine A: LightGBM lead scorer
│   ├── income_estimation_model.py  # Engine B: XGBoost income estimator
│   ├── underwriting_engine.py      # Engine C: FOIR-based underwriting rules
│   └── explainability.py           # SHAP → plain-English explanations
├── app/
│   └── dashboard.py                # Streamlit RM dashboard (4 tabs)
├── models/                         # Saved model artifacts (.pkl)
├── docs/
│   ├── architecture_diagram.md     # Mermaid architecture diagrams
│   ├── pitch_deck_outline.md       # 10-slide pitch deck outline
│   └── demo_script.md             # 3-minute demo walkthrough
├── tests/
│   └── test_pipeline.py           # Integration & unit tests
└── .streamlit/
    └── config.toml                # IDBI maroon/dark theme
```

---

## ⚙️ Feature Engineering

IntelliLend engineers **35+ features** across 7 groups:

| Group | Features | Example |
|-------|----------|---------|
| Cash-flow Stability | CV of monthly inflows, credit mean/std | CV = 0.12 (stable) |
| Salary Detection | Has salary, regularity score, estimated salary | ₹85,000/month, 95% regular |
| EMI Burden | Existing EMI ratio, EMI count | 15% of income in EMIs |
| RFM Scores | Recency, Frequency, Monetary (0-1) | R=0.9, F=0.7, M=0.8 |
| Spend Patterns | Category HHI, intent flags (auto/real estate/wedding) | Auto dealer spend detected |
| Digital Engagement | App logins, EMI calc views, loan page visits, trend | Increasing engagement (+25%) |
| Demographics | Age, occupation, city tier, tenure, product holdings | Salaried, Tier 1, 36 months |

---

## 🎯 Engine A — Lead Scoring

- **Model**: LightGBM gradient-boosted classifier
- **Output**: Lead Quality Score (0-100), calibrated from predicted probabilities
- **Calibration**: Isotonic regression for well-calibrated probabilities
- **Class balancing**: Automatic scale_pos_weight adjustment

### Conversion Rate Simulation

If the bank calls only the **top N% of scored leads**, the conversion rate is:

| Cutoff | Conversion Rate | vs. Baseline |
|--------|----------------|--------------|
| Top 5% | ~45%+ | >3x baseline |
| Top 10% | ~40%+ | >3x baseline |
| Top 20% | ~35%+ | >30% target ✅ |
| Baseline (random) | ~10-12% | — |

---

## 💰 Engine B — Income Assessment

- **Model**: XGBoost Regressor
- **Target**: True monthly income (ground truth in synthetic data)
- **Confidence Bands**: Assigned based on prediction error + cash-flow stability + occupation type

| Band | Criteria | Typical MAPE |
|------|----------|-------------|
| High | Salaried + stable cash-flow + low error | <5% |
| Medium | Semi-regular income + moderate error | 10-20% |
| Low | Irregular income + high error or thin data | >25% |

- **Income Deviation Detection**: Flags customers whose declared income deviates >20% from estimated income

---

## 📋 Engine C — Underwriting

### FOIR-Based Eligible Amount

| Loan Product | Max FOIR | Max Amount | Interest Rate | Default Tenure |
|-------------|----------|------------|---------------|----------------|
| Personal Loan | 50% | ₹25 Lakh | 12% | 5 years |
| Home Loan | 45% | ₹1 Crore | 8.5% | 20 years |
| Mortgage Loan | 45% | ₹75 Lakh | 9.5% | 15 years |
| Auto Loan | 40% | ₹20 Lakh | 9.5% | 7 years |

### Risk Flags (7 checks)
1. Irregular income pattern (CV > 40%)
2. High existing EMI burden (>40%)
3. Income-declaration mismatch (>20%)
4. Low income confidence band
5. Short banking relationship (<6 months)
6. Unusual credit inflow variance
7. High lead score but low digital engagement

### Recommendations
- **Pre-qualified**: Readiness ≥65, ≤1 risk flag, eligible amount >0
- **Review Required**: Readiness ≥40, eligible amount >0
- **Not Recommended**: Otherwise

---

## 🔍 Explainability

Every lead score includes a **SHAP-based, plain-English explanation**:

> *"Consistent ₹85k monthly salary credits + Low existing EMI burden (12% of income) + Active use of EMI calculator (3.2/month) shows purchase intent → strong Home Loan intent"*

- No paid LLM required — template-based sentence generation
- 40+ feature-to-business-language templates
- Compliant with RBI Fair Practices Code transparency requirements

---

## 📊 Dashboard

The Streamlit dashboard provides 4 views:

1. **🏆 Lead Leaderboard**: Ranked, filterable table of all scored leads
2. **🔎 Customer Drilldown**: Full SHAP chart, income estimate, risk flags, one-click RM assignment
3. **📈 Portfolio Analytics**: Before/After IntelliLend comparison, conversion rate charts, segment breakdowns
4. **🎛️ What-If Simulator**: Adjust income/EMI/parameters and watch Loan Readiness Score update live

### Branding
- IDBI Bank maroon (#9B1B30) / dark theme
- Professional typography (Inter font)
- Animated KPI cards with gradient accents

---

## 🔐 Responsible AI & Compliance

### RBI Digital Lending Guidelines
- **Human-in-the-loop**: IntelliLend recommends — RM/underwriter decides
- **No auto-approval**: All recommendations require human review
- **Transparent pricing**: FOIR caps and rules are visible and configurable

### Fair Practices Code
- Every score has an auditable explanation
- Scoring criteria are transparent and documented
- No discriminatory features (no gender, religion, caste)

### Account Aggregator Framework (Production Roadmap)
- Designed for consent-based data sharing
- Customer controls what data is shared
- FIP/FIU architecture compatible

### Bias Testing
- Score distributions checked across occupation types
- City tier analysis ensures no systematic disadvantage
- Age-group fairness analysis included

### Data Privacy
- ⚠️ **All data in this demo is 100% synthetic** — generated using Faker + NumPy
- No real customer PII is used anywhere
- Seed is fixed (42) for full reproducibility
- In production: AES-256 encryption, consent-based access, audit trails

---

## 🛠️ Tech Stack

| Layer | Tool | Cost |
|-------|------|------|
| Language | Python 3.9+ | Free |
| Data Generation | Faker, NumPy, Pandas | Free |
| Storage | SQLite | Free |
| Lead Scoring | LightGBM | Free |
| Income Estimation | XGBoost | Free |
| Explainability | SHAP | Free |
| Visualization | Plotly, Matplotlib | Free |
| Dashboard | Streamlit | Free |
| Deployment | Streamlit Community Cloud / HF Spaces | Free |
| Testing | pytest | Free |

**Total cost: ₹0**

---

## 🗺️ Future Roadmap

1. **Account Aggregator Integration**: Real-time transaction data via AA APIs
2. **GST/ITR Data Ingestion**: For self-employed income verification
3. **UPI Transaction Analytics**: Deeper spend pattern analysis
4. **NLP Bank Statement Processing**: OCR + NLP for uploaded statements
5. **MLOps Pipeline**: Automated model retraining with drift detection
6. **RM CRM Integration**: Seamless workflow with existing tools (Salesforce, etc.)
7. **A/B Testing Framework**: Measure real-world conversion uplift
8. **Mobile RM App**: On-the-go lead management

---

## 📝 Assumptions

1. Transaction data follows realistic Indian banking patterns (salary credits on 1st-3rd, UPI spends, EMI auto-debits)
2. Conversion probability is driven by a logistic model with known feature weights (simulating real-world behavior)
3. Income estimation accuracy assumes transaction data quality similar to real bank data
4. FOIR caps and interest rates are approximate market rates (configurable)
5. Digital engagement signals (app logins, page visits) are simulated but follow realistic distributions
6. 5,000 customers is sufficient for a hackathon demo; production would use full customer base
7. Seed = 42 ensures full reproducibility across runs

---

## 🚀 Deployment

### Option 1: Streamlit Community Cloud (Recommended)

1. Push code to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → select your repo
4. Set main file path: `app/dashboard.py`
5. Click "Deploy"
6. Wait for build → get a shareable link like `https://your-app.streamlit.app`

**Important**: The pipeline must run on the cloud instance. Add this to the top of `app/dashboard.py`:
```python
# Auto-run pipeline if models don't exist
if not os.path.exists(os.path.join(PROJECT_ROOT, "models", "pipeline_output.pkl")):
    import subprocess
    subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "run_pipeline.py")])
```

### Option 2: Hugging Face Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select "Streamlit" as the SDK
3. Push your code to the Space repo
4. Add `requirements.txt` and set app file to `app/dashboard.py`

---

## 👥 Team

*[Your team name and members here]*

---

## 📄 License

MIT License — Free for educational and commercial use.

---

<p align="center">
  <b>IDBI IntelliLend</b> — Turning transaction data into high-conversion leads.<br>
  Built with ❤️ for Smart India Hackathon
</p>
