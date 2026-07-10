# Demo Script — IDBI IntelliLend

## 3-Minute Live Demo Walkthrough

---

### Setup (before going on stage)
1. Ensure `streamlit run app/dashboard.py` is running
2. Open browser to `http://localhost:8501`
3. Have the "Lead Leaderboard" tab active
4. Set all filters to "All" for maximum data

---

### MINUTE 0:00–0:30 — The Problem Hook

> *"Imagine you're an IDBI Bank Relationship Manager. You have 5,000 customers in your portfolio. Who do you call today for a home loan? Currently, it's guesswork — blanket outreach with a 10% conversion rate. 90 out of 100 calls are wasted.*
>
> *IntelliLend changes that."*

**[Point to the KPI cards at the top]**

> *"IntelliLend has scored all 5,000 customers using AI. It tells you WHO to call, WHAT to offer, and WHY they're likely to convert."*

---

### MINUTE 0:30–1:15 — The Leaderboard

**[Click on the Lead Leaderboard tab]**

> *"This is the RM's command center. Leads sorted by AI-generated Lead Quality Score. Each row shows the customer, their score, recommended loan product, estimated income with a confidence band, and eligible loan amount."*

**[Click filter: Loan Product → Home Loan]**

> *"I can filter by product. Here are my top Home Loan prospects. Notice how the top leads are pre-qualified with high confidence."*

**[Point to a high-scoring customer]**

> *"Let's drill into this customer."*

---

### MINUTE 1:15–2:00 — Customer Drilldown

**[Click on Customer Drilldown tab, select the top customer]**

> *"Here's the full picture. Lead Quality Score of [X], Loan Readiness Score of [Y]. Estimated monthly income of ₹[Z] with HIGH confidence."*

**[Point to the SHAP chart]**

> *"This is the key differentiator — every score is EXPLAINABLE. The green bars show what's driving this customer's score UP: consistent salary credits, low EMI burden, recent real estate broker transactions. The red bars show what's pulling it DOWN."*

**[Read the AI explanation at the bottom]**

> *"And it's translated into plain English: [read the explanation]. This is auditable, compliant with RBI Fair Practices Code, and something an underwriter can trust."*

**[Point to risk flags section]**

> *"Any risk flags are clearly surfaced. This is human-in-the-loop design — the AI recommends, the RM decides."*

**[Click "Assign to RM" button]**

> *"One click to assign for follow-up."*

---

### MINUTE 2:00–2:30 — The Impact

**[Click on Portfolio Analytics tab]**

> *"Here's the business impact. Before IntelliLend: 10% conversion with blanket outreach."*

**[Point to the Before/After comparison cards]**

> *"After IntelliLend: [X]% conversion on our top 20% leads — that's a [Y]x improvement. We EXCEED the problem statement's 30% target."*

**[Point to the conversion rate chart]**

> *"This chart shows the conversion rate at every cutoff. Even if you only call the top 5%, you're getting [Z]% conversion. This is real — computed from the model's actual predictions, not made-up numbers."*

---

### MINUTE 2:30–3:00 — What-If & Close

**[Click on What-If Simulator tab]**

> *"And for the live demo moment — judges, try this. I can adjust any customer parameter in real-time."*

**[Move the Income slider up]**

> *"Watch the Loan Readiness Score and Eligible Amount update live. Move the EMI slider up — see the eligible amount decrease. This is fully transparent and configurable."*

**[Final statement]**

> *"IntelliLend: data-driven leads, explainable income assessment, prudent underwriting. 100% open-source, zero cost, built specifically for IDBI Bank. Thank you."*

---

### Backup Q&A Answers

**Q: "How do you handle data privacy?"**
> All data in this demo is 100% synthetic. In production, we'd use the Account Aggregator framework — consent-based, customer-controlled data sharing per RBI guidelines.

**Q: "Is this a black box?"**
> No. Every score has a SHAP-based explanation translated into business English. We can show the exact features driving any customer's score.

**Q: "What about bias?"**
> We've tested score distributions across occupation types, city tiers, and age groups. The model doesn't systematically disadvantage any segment.

**Q: "Why not use a large language model?"**
> LLMs add cost, latency, and unpredictability. Our template-based explanations are faster, free, deterministic, and auditable — which matters more for banking than creative text.

**Q: "What about model accuracy?"**
> ROC-AUC of [X], Precision@Top-20% of [Y]%, income estimation MAPE of [Z]%. All metrics computed honestly on held-out test data.

**Q: "How would this integrate with IDBI's systems?"**
> The pipeline takes flat transaction data as input. In production, this would connect via Account Aggregator APIs, feed into the existing CRM, and serve recommendations through a REST API.
