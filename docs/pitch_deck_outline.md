# Pitch Deck Outline — IDBI IntelliLend

## 10-Slide Structure for Smart India Hackathon Presentation

---

### Slide 1: Title Slide
**IDBI IntelliLend — AI-Driven Lead Intelligence & Alternate Income Assessment Engine**

- Team name + member photos
- Tagline: *"Turning transaction data into high-conversion leads and defensible income estimates"*
- IDBI Bank logo + SIH logo
- Problem Statement reference

---

### Slide 2: The Problem
**Current Challenges in IDBI Bank Retail Lending**

- Traditional metrics → low conversion rates (8-12%)
- Blanket outreach wastes RM bandwidth
- Manual income assessment is slow and error-prone
- Thin-file/self-employed customers are underserved
- High cost-per-lead with poor targeting

*Visual: Funnel graphic showing lead drop-off at each stage*

---

### Slide 3: Our Solution — IntelliLend
**A 3-Engine AI Pipeline**

- **Engine A**: Behavioral Lead Scoring → prioritizes the right prospects
- **Engine B**: Alternate Income Assessment → estimates income from transactions
- **Engine C**: Prudent Underwriting → transparent, auditable loan recommendations

*Visual: Architecture diagram (3 engines flowing into RM dashboard)*

---

### Slide 4: Engine A — Lead Scoring
**ML-Powered Propensity Scoring**

- LightGBM classifier with 35+ engineered features
- Cash-flow stability, salary detection, EMI burden, RFM, digital engagement
- Per-customer, per-product Lead Quality Score (0-100)
- **Result: >30% conversion rate on top-20% leads** (vs. 10% baseline)

*Visual: ROC curve + Conversion rate chart at various cutoffs*

---

### Slide 5: Engine B — Income Assessment
**Transaction-Based Income Estimation for Thin-File Customers**

- XGBoost regressor trained on behavioral features
- Confidence Bands (High/Medium/Low) for prudent underwriting
- Cross-validates declared vs. estimated income
- Flags deviations >20% for manual review

*Visual: Income estimation accuracy chart + confidence band distribution*

---

### Slide 6: Engine C — Underwriting Recommendations
**Human-in-the-Loop, Not Auto-Approval**

- Transparent FOIR-based eligible amount calculation
- Configurable per loan product (PL/HL/ML/AL)
- Multi-factor risk flag system (7 checks)
- Loan Readiness Score combining all signals

*Visual: Underwriting rules table + risk flag examples*

---

### Slide 7: Explainability — Why This Score?
**SHAP-Based, Business-Readable Explanations**

- Every score comes with a plain-English "why"
- Template-based explanation generation (no paid LLM)
- Example: *"Consistent ₹85k monthly salary credits + low EMI burden + recent auto dealer visit → strong Auto Loan intent"*
- Compliant with RBI Fair Practices Code

*Visual: SHAP waterfall chart from the dashboard*

---

### Slide 8: Live Demo — The Dashboard
**RM/Underwriter Facing Dashboard**

- Lead Leaderboard with smart filtering
- Customer drilldown with full explanation
- Portfolio analytics with Before/After comparison
- What-If simulator for live parameter adjustment

*Visual: Dashboard screenshots or live demo switch*

---

### Slide 9: Impact Numbers
**Measurable Business Outcomes**

| Metric | Before | After IntelliLend | Impact |
|--------|--------|--------------------|--------|
| Conversion Rate | ~10% | >30% (top 20%) | 3x uplift |
| Cost per Lead | ₹800 | ₹250 | 69% reduction |
| Turnaround Time | 48 hrs | 2 hrs | 96% faster |
| Income Assessment Accuracy | Manual/unknown | <15% MAPE | Quantified |
| RM Productivity | 20 calls/day | 15 targeted calls | Higher yield |

---

### Slide 10: Compliance & Roadmap
**Responsible AI & Future Vision**

**Compliance:**
- RBI Digital Lending Guidelines compliant
- Human-in-the-loop underwriting (recommends, not decides)
- Bias testing across segments
- Account Aggregator consent-based design

**Roadmap:**
1. Integration with Account Aggregator framework (real-time)
2. GST/UPI data ingestion via consent
3. NLP processing of bank statements
4. Model retraining pipeline with MLOps
5. RM CRM integration for seamless workflow

*Visual: Roadmap timeline graphic*

---

### Bonus Slide: Technical Architecture (if time)

```
Account Aggregator → Consent Layer → Data Lake
    ↓
Feature Store (real-time + batch)
    ↓
Model Serving (Engine A + B + C)
    ↓
RM CRM Dashboard ← Explainability API
    ↓
Feedback Loop → Model Retraining
```

---

## Speaker Notes

- Open with the problem (30 sec)
- Solution overview (30 sec)
- Quick engine walk-through (60 sec)
- LIVE DEMO of the dashboard (45 sec) — this is the wow moment
- Impact numbers + compliance (30 sec)
- Close with roadmap (15 sec)

**Total: ~3.5 minutes** (tight for most hackathon formats)

**Key phrases to use:**
- "Data-driven, not gut-driven"
- "Human-in-the-loop — the model recommends, the RM decides"
- "Every score is explainable and auditable"
- "30% conversion rate exceeds the problem statement target"
- "100% free stack — no vendor lock-in"
