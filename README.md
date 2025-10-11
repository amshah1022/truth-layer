# Truth Layer: Evidence-Grounded Evaluation for LLMs
**Truth Layer** is a reproducible evaluation pipeline for large language models (LLMs) that transforms truthfulness from a subjective judgment into a measurable property.  
It integrates retrieval**, constrained generation, and NLI-based verification to create audit-ready evaluations that reveal when, how, and why models hallucinate.

---

## Why This Matters

Standard metrics like BLEU and ROUGE measure wording overlap, not factual accuracy. Even dedicated truthfulness tests such as TruthfulQA rely on static human annotations rather than evidence-grounded verification.
They fail to capture when models assert unsupported or unverifiable claims.

Truth Layer tackles this gap by introducing evidence-grounded evaluation, forcing every claim to be linked to retrieved evidence and verified via entailment.  
The result is an infrastructure-level approach to truthfulness that can be audited, compared, and replicated.

---

## System Architecture

```text
┌────────────┐      ┌────────────┐          ┌──────────────┐
│ Retrieval  │───▶  │ Generation │    ───▶  │ Verification │
│ (FAISS /   │      │ (LLM w/    │          | (NLI model   │
│ Wikipedia) │      │constraints)│          │ or entailment│
└────────────┘      └────────────┘          └──────────────┘
         │                   │
         ▼                   ▼
   Evidence store       JSON audit logs
```

**1) Retrieval** – Collect top-k context from trusted corpora (Wikipedia, PubMed, ArXiv, etc.)  
**2) Constrained Generation** – LLM must answer *only* within retrieved evidence windows  
**3)  NLI Verification** – Classify claims as **Supported**, **Contradicted**, or **Unverifiable**  
**4)  Aggregation** – Produce reproducible JSON artifacts for audit and cross-model comparison  

---

## Example Output

```json
{
  "claim": "The Nile is the longest river in the world.",
  "evidence": [
    "The Nile is 6,650 km long, slightly shorter than the Amazon River."
  ],
  "label": "Contradicted"
}

## Early Results (Prototype)

| Model | Supported | Contradicted | Unverifiable | Exact | Loose | Soft | Recall |
|--------|------------|---------------|---------------|--------|--------|--------|---------|
| **GPT-4o-mini** | 111 | 7 | 2 | 0.85 | 0.91 | 0.93 | 0.93 |
| **Llama-3.1-8B-Instruct** | 107 | 8 | 5 | 0.85 | 0.86 | 0.89 | 0.90 |
| **Phi-3-mini-4k-Instruct** | 143 | 94 | 63 | 0.69 | 0.81 | 0.48 | 0.42 |

*Evaluated on 120 claims for Phi-3, Llama-3.1, and GPT-4o. Confidence intervals represent 95% bootstrap estimates.*  
*(Full paper forthcoming — Shah, 2025)*

---

###  Per-Domain Breakdown (Phi-3-mini-4k-Instruct)

| Domain | Exact | Loose | Soft | Recall |
|---------|--------|--------|--------|---------|
| History | 0.78 | 0.78 | 0.43 | 0.40 |
| Literature | 0.90 | 0.90 | 0.88 | 0.88 |
| Science | 0.70 | 0.88 | 0.45 | 0.43 |
| Medicine | 0.50 | 0.85 | 0.45 | 0.40 |
| Computer Science | 0.75 | 0.90 | 0.18 | 0.18 |

*Each domain evaluated on 120 items (n = 200 total).*

---

### Pairwise McNemar Tests

| Model A | Model B | Metric | n (shared) | A Wrong / B Right | A Right / B Wrong | p-value |
|----------|----------|---------|-------------|--------------------|--------------------|----------|
| Phi-3-mini-4k | Llama-3.1-8B | exact | 120 | 13 | 6 | 0.167 |
| Phi-3-mini-4k | Llama-3.1-8B | soft | 120 | 42 | 5 | < 0.001 |
| Phi-3-mini-4k | GPT-4o-mini | exact | 120 | 10 | 3 | 0.092 |
| Phi-3-mini-4k | GPT-4o-mini | soft | 120 | 43 | 2 | < 0.001 |
| Llama-3.1-8B | GPT-4o-mini | exact | 120 | 9 | 9 | 1.000 |

**Interpretation:**  
GPT-4o-mini and Llama-3.1-8B both significantly outperform Phi-3-mini-4k on soft agreement metrics (*p < 0.001*).  
Across domains, **literature** and **computer science** yield the highest grounding consistency, while **medicine** remains most challenging.

---

*Full raw results available in [`runs/`](runs/):*  
[`per_model_summary.csv`](runs/per_model_summary.csv) · [`per_domain_summary.csv`](runs/per_domain_summary.csv) · [`pairwise_mcnemar.csv`](runs/pairwise_mcnemar.csv)


##  Research Context

Truth Layer builds upon and extends recent progress in factuality evaluation:

- **TruthfulQA** – Lin et al., 2022  
- **RARR: Retrieval-Augmented Response Rewriting** – Gao et al., 2023  
- **FactScore** – Min et al., 2023  

It operationalizes these ideas into a modular pipeline for AI Reliability Infrastructure, complementing related projects such as ProbeEng (interpretability) and *”OSCE Learning Analytics: Rubric-Guided Generation and Evaluation of LLM Feedback”*
(human-feedback calibration).

**Paper:** *“Evidence-Grounded Evaluation: Toward Infrastructure for Truthful AI.”*  
*(in preparation — Shah, 2025)*

---

## Installation

```bash
git clone https://github.com/amshah1022/truth-layer.git
cd truth-layer
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **API keys:**  
> Set your model keys in a `.env` file or environment variables (example below).

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
HF_TOKEN=...
```

---

## Run the Streamlit App

```bash
streamlit run app.py
```

This launches a local dashboard at **http://localhost:8000** where you can enter a prompt to evaluate  

Outputs are written automatically to:

```
runs/<timestamp>/   # JSON caches and retrieved evidence
```

---


## Features

- **Evidence-Grounded Evaluation** – Checks every claim against retrieved context.  
- **Audit-Ready Outputs** – JSON caches enable exact reruns and peer comparison.  
- **Backend-Agnostic** – Supports OpenAI, Anthropic, or local HF models.  
- **Transparent Benchmarks** – Enables longitudinal reliability tracking.  

---
## 🧭 Roadmap

**Phase 1 — Core Reliability Infrastructure (Q4 2025)**
- [ ] Integrate **multi-document retrieval** and cross-corpus evidence fusion  
- [ ] Add **per-claim verification** pipeline for finer-grained truth metrics  
- [ ] Release **public evaluation scripts** for TruthfulQA, MedQA, and custom datasets  

**Phase 2 — Calibration & Comparative Analysis (Q1 2026)**
- [ ] Implement **verifier ensembles** for uncertainty calibration  
- [ ] Introduce **confidence-weighted scoring** and reliability curves  
- [ ] Expand **model comparison suite** (pairwise McNemar, bootstrap CIs)  

**Phase 3 — Transparency & Collaboration (Q2 2026)**
- [ ] Launch an **open benchmark leaderboard** for community submissions  
- [ ] Deploy **interactive web dashboard** for audit visualization  
- [ ] Publish **evaluation standard & dataset schema** for reproducible truthfulness research  


---

## Contributing

Truth Layer is part of a growing ecosystem of AI Reliability Infrastructure projects aimed at grounding safety in empirical verification rather than assurances.  
Contributions are welcome especially in retrieval optimization, NLI verification modeling, and benchmark design.

---

## Citation

```bibtex
@inprogress{shah2025truthlayer,
  title={Evidence-Grounded Evaluation: Toward Infrastructure for Truthful AI},
  author={Shah, Alina Miret},
  year={2025},
  note={Work in progress}
}
```

---

## Contact

**Alina Miret Shah – Cornell University**  
 amshah@cornell.edu  
[alina.miret](https://www.linkedin.com/in/alina-miret)

