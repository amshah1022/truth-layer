# Truth Layer: Evidence-Grounded Evaluation for LLMs
**Truth Layer** is a reproducible evaluation pipeline for large language models (LLMs) that transforms truthfulness from a subjective judgment into a measurable property.  
It integrates retrieval, constrained generation, and NLI-based verification to create audit-ready evaluations that reveal when, how, and why models hallucinate.

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
│ (BM25 /   │       │ (LLM w/    │           | (NLI model   │
│ Wikipedia) │      │constraints)│          │ or entailment│
└────────────┘      └────────────┘          └──────────────┘
         │                   │
         ▼                   ▼
Evidence cache          CSV / JSON summaries
(retrieved passages)   (per claim & per model results) 
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
```

---

## Early Results (Prototype)

| Model | Supported | Contradicted | Unverifiable | Exact | Loose | Soft | Recall |
|--------|------------|---------------|---------------|--------|--------|--------|---------|
| **GPT-4o-mini** | 111 | 7 | 2 | 0.85 | 0.91 | 0.93 | 0.93 |
| **Llama-3.1-8B-Instruct** | 107 | 8 | 5 | 0.85 | 0.86 | 0.89 | 0.90 |
| **Phi-3-mini-4k-Instruct** | 143 | 94 | 63 | 0.69 | 0.81 | 0.48 | 0.42 |

*Evaluated on 120 claims for Phi-3, Llama-3.1, and GPT-4o. Confidence intervals represent 95% bootstrap estimates.*  
*(Full paper forthcoming — Shah, 2025)*

---

### Per-Domain Breakdown (Phi-3-mini-4k-Instruct)

model,domain,n,exact_mean,exact_lo,exact_hi,loose_mean,loose_lo,loose_hi,soft_mean,soft_lo,soft_hi,recall_mean,recall_lo,recall_hi
microsoft/Phi-3-mini-4k-instruct,history,40,0.775,0.65,0.9,0.775,0.65,0.9,0.425,0.275,0.575,0.4,0.25,0.55
microsoft/Phi-3-mini-4k-instruct,literature,40,0.9,0.8,0.975,0.9,0.8,0.975,0.875,0.775,0.975,0.875,0.775,0.975
microsoft/Phi-3-mini-4k-instruct,science,40,0.7,0.55,0.825,0.875,0.75,0.975,0.45,0.3,0.6,0.425,0.275,0.575
microsoft/Phi-3-mini-4k-instruct,medicine,40,0.5,0.35,0.65,0.85,0.725,0.95,0.45,0.3,0.6,0.4,0.25,0.55
microsoft/Phi-3-mini-4k-instruct,cs,40,0.75,0.6,0.875,0.9,0.8,0.975,0.175,0.075,0.3,0.175,0.075,0.3
microsoft/Phi-3-mini-4k-instruct,civics,40,0.425,0.275,0.575,0.475,0.325,0.625,0.4,0.25,0.55,0.225,0.1,0.35
microsoft/Phi-3-mini-4k-instruct,ambiguous,20,0.85,0.7,1.0,0.95,0.85,1.0,0.5,0.3,0.7,0.4,0.2,0.6
microsoft/Phi-3-mini-4k-instruct,multihop,20,0.7,0.5,0.9,0.75,0.55,0.9,0.6,0.4,0.8,0.45,0.25,0.65
microsoft/Phi-3-mini-4k-instruct,confusion,20,0.75,0.55,0.9,0.85,0.7,1.0,0.5,0.3,0.7,0.45,0.25,0.65
meta-llama/Llama-3.1-8B-Instruct,history,40,0.85,0.725,0.95,0.85,0.725,0.95,0.9,0.8,0.975,0.9,0.8,0.975
meta-llama/Llama-3.1-8B-Instruct,literature,40,0.85,0.725,0.95,0.85,0.725,0.95,0.875,0.775,0.975,0.9,0.8,0.975
meta-llama/Llama-3.1-8B-Instruct,science,40,0.85,0.725,0.95,0.875,0.775,0.975,0.9,0.8,0.975,0.9,0.8,0.975
gpt-4o-mini,history,40,0.9,0.8,0.975,0.9,0.8,0.975,0.875,0.775,0.975,0.9,0.8,0.975
gpt-4o-mini,literature,40,0.925,0.85,1.0,0.925,0.85,1.0,0.95,0.875,1.0,0.975,0.925,1.0
gpt-4o-mini,science,40,0.725,0.575,0.85,0.9,0.8,0.975,0.95,0.875,1.0,0.925,0.825,1.0

*Each domain evaluated on 120 items.*

---

### Pairwise McNemar Tests

model_A,model_B,metric,n_shared,A_wrong_B_right,A_right_B_wrong,p_value
microsoft/Phi-3-mini-4k-instruct,meta-llama/Llama-3.1-8B-Instruct,exact,120,13,6,0.167068
microsoft/Phi-3-mini-4k-instruct,meta-llama/Llama-3.1-8B-Instruct,soft,120,42,5,0.0
microsoft/Phi-3-mini-4k-instruct,gpt-4o-mini,exact,120,10,3,0.092285
microsoft/Phi-3-mini-4k-instruct,gpt-4o-mini,soft,120,43,2,0.0
meta-llama/Llama-3.1-8B-Instruct,gpt-4o-mini,exact,120,9,9,1.0
meta-llama/Llama-3.1-8B-Instruct,gpt-4o-mini,soft,120,6,2,0.289062


**Interpretation:**  
GPT-4o-mini and Llama-3.1-8B both significantly outperform Phi-3-mini-4k on soft agreement metrics (*p < 0.001*).  
Across domains, **literature** and **computer science** yield the highest grounding consistency, while **medicine** remains most challenging.

---

*Full raw results available in [`runs/`](runs/):*  
[`per_model_summary.csv`](runs/per_model_summary.csv) · [`per_domain_summary.csv`](runs/per_domain_summary.csv) · [`pairwise_mcnemar.csv`](runs/pairwise_mcnemar.csv)

---

*Full raw results available in [`runs/`](runs/):*  
[`per_model_summary.csv`](runs/per_model_summary.csv) · [`per_domain_summary.csv`](runs/per_domain_summary.csv) · [`pairwise_mcnemar.csv`](runs/pairwise_mcnemar.csv)

---

##  Research Context

Truth Layer builds upon and extends recent progress in factuality evaluation:

- **TruthfulQA** – Lin et al., 2022  
- **RARR: Retrieval-Augmented Response Rewriting** – Gao et al., 2023  
- **FactScore** – Min et al., 2023  

Truth Layer unifies these ideas into a practical, end-to-end framework for evaluating factual reliability.
It complements related efforts like ProbeEng (model interpretability) and *”OSCE Learning Analytics: Rubric-Guided Generation and Evaluation of LLM Feedback”*
(human-feedback calibration).

**Paper:** *“Evidence-Grounded Evaluation: Toward Infrastructure for Truthful AI.”*  
*(in preparation — Shah, 2025)*

---

## Installation

```bash
git clone https://github.com/amshah1022/truth-layer.git
cd truth-layer
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
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


## Roadmap

**Phase 1 — Core Reliability Infrastructure (Q4 2025)**
- [ ] Extend retrieval to multiple sources (Wikipedia, PubMed, ArXiv)  
- [ ] Add per-claim verification for finer-grained truth metrics  
- [ ] Release public evaluation scripts for multi-domain factual QA datasets  

**Phase 2 — Calibration & Comparative Analysis (Q1 2026)**
- [ ] Prototype verifier ensembles and uncertainty scoring  
- [ ] Introduce confidence-weighted metrics and reliability curves  
- [ ] Expand model comparison suite (McNemar tests, bootstrap CIs)  

**Phase 3 — Transparency & Collaboration (Q2 2026)**
- [ ] Define an open evaluation format to enable community submissions  
- [ ] Deploy an interactive Streamlit dashboard for audit visualization  
- [ ] Draft and publish an evaluation schema for reproducible truthfulness research  

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


















