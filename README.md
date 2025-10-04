# Truth Layer: Evidence-Grounded Evaluation for LLMs

Truth Layer is a reproducible evaluation pipeline for large language models (LLMs) that combines*retrieval, constrained generation, and NLI-based verification to create audit-ready truthfulness benchmarks. The goal is to expose systematic failure modes that are invisible to surface-level metrics (e.g., token overlap or BLEU) and make LLM evaluations more transparent, reproducible, and accountable.

---

## Features
- **Evidence-Grounded Evaluation**: Responses are checked against retrieved sources to prevent models from “hallucinating” plausible but false claims.  
- **Three-Layer Pipeline**:
  1. **Retrieval**: Collects relevant evidence from trusted corpora.  
  2. **Constrained Generation**: Forces models to ground answers in retrieved context.  
  3. **NLI Verification**: Uses natural language inference to test if model claims are supported, contradicted, or unverifiable.  
- **Audit-Ready Outputs**: Generates structured JSON caches that can be re-run, compared, and shared across experiments.  
- **Flexible Deployment**: Works with multiple LLM backends; modular design allows substitution of retrieval engines or verifiers.

---

## Installation
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt


