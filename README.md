# Truth Layer: Evidence-Grounded Evaluation for LLMs
_Truth Layer_ is a reproducible evaluation pipeline for large language models (LLMs) that transforms truthfulness from a subjective judgment into a measurable property.  
It integrates retrieval, constrained generation, and NLI-based verification to create audit-ready evaluations that reveal when, how, and why models hallucinate.

Paper: [_Evidence-Grounded Evaluation: Toward
Infrastructure for Truthful AI_](https://github.com/amshah1022/truth-layer/blob/16ef276549d3e1835d1e1d61beccc5db6f545a7e/Evidence_Grounded_Evaluation__Toward_Instrastructure_for_Truthful_AI.pdf)

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
│ (BM25 /    │      │ (LLM w/    │          | (NLI model   │
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
| Model                            | n   | Exact | Loose | Soft  | Recall | Supported | Contradicted | Unverifiable |
| -------------------------------- | --- | ----- | ----- | ----- | ------ | --------- | ------------ | ------------ |
| microsoft/Phi-3-mini-4k-instruct | 300 | 0.693 | 0.807 | 0.477 | 0.420  | 143       | 94           | 63           |
| meta-llama/Llama-3.1-8B-Instruct | 120 | 0.850 | 0.858 | 0.892 | 0.900  | 107       | 8            | 5            |
| gpt-4o-mini                      | 120 | 0.850 | 0.908 | 0.925 | 0.933  | 111       | 7            | 2            |

*Evaluated on 120 claims for Phi-3, Llama-3.1, and GPT-4o. Confidence intervals represent 95% bootstrap estimates.*  
*(Full paper forthcoming — Shah, 2025)*

---

### Per-Domain Breakdown (Phi-3-mini-4k-Instruct)

| Domain           | Exact | Loose | Soft  | Recall |
| ---------------- | ----- | ----- | ----- | ------ |
| History          | 0.775 | 0.775 | 0.425 | 0.400  |
| Literature       | 0.900 | 0.900 | 0.875 | 0.875  |
| Science          | 0.700 | 0.875 | 0.450 | 0.425  |
| Medicine         | 0.500 | 0.850 | 0.450 | 0.400  |
| Computer Science | 0.750 | 0.900 | 0.175 | 0.175  |
| Civics           | 0.425 | 0.475 | 0.400 | 0.225  |
| Ambiguous        | 0.850 | 0.950 | 0.500 | 0.400  |
| Multihop         | 0.700 | 0.750 | 0.600 | 0.450  |
| Confusion        | 0.750 | 0.850 | 0.500 | 0.450  |

*Every domain evaluated on 40 items except for ambiguous, multihop, and confusion (20 each).*

---

### Pairwise McNemar Tests

| Model A | Model B | Metric | n (shared) | A Wrong / B Right | A Right / B Wrong | p-value |
|----------|----------|--------|------------|-------------------|-------------------|----------|
| Phi-3-mini-4k | Llama-3.1-8B | exact | 120 | 13 | 6 | 0.167 |
| Phi-3-mini-4k | Llama-3.1-8B | soft | 120 | 42 | 5 | &lt;0.001 |
| Phi-3-mini-4k | GPT-4o-mini | exact | 120 | 10 | 3 | 0.092 |
| Phi-3-mini-4k | GPT-4o-mini | soft | 120 | 43 | 2 | &lt;0.001 |
| Llama-3.1-8B | GPT-4o-mini | exact | 120 | 9 | 9 | 1.000 |
| Llama-3.1-8B | GPT-4o-mini | soft | 120 | 6 | 2 | 0.289 |

GPT-4o-mini and Llama-3.1-8B both significantly outperform Phi-3-mini-4k on soft agreement metrics (*p < 0.001*).  

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
alina.shah1022@gmail.com
[alina.miret](https://www.linkedin.com/in/alinamshah/)



























