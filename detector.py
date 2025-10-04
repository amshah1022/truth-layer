# detector.py
from __future__ import annotations
from typing import List, Dict
from functools import lru_cache
import os, re, json, hashlib

import wikipedia
from transformers import pipeline

# =========================
# Config
# =========================
# Faster default; switch via env if you want stronger (but slower) NLI
#   - fast:    "typeform/distilbert-base-uncased-mnli"
#   - standard:"facebook/bart-large-mnli"
#   - slowest: "cross-encoder/nli-deberta-v3-base"
NLI_MODEL_ID = os.getenv("NLI_MODEL_ID", "typeform/distilbert-base-uncased-mnli")
WIKI_SENTENCES = int(os.getenv("WIKI_SENTENCES", "3"))
WIKI_RESULTS_PER_QUERY = int(os.getenv("WIKI_RESULTS_PER_QUERY", "2"))
USE_CLAIMIFY = True  # set False to behave like pre-change baseline

# =========================
# Small helpers
# =========================
def _clean_text(s: str) -> str:
    s = (s or "").strip()
    return re.sub(r"\s+", " ", s)

def _first_sentences(txt: str, n_sentences: int = 5) -> str:
    parts = re.split(r"(?<=[.!?])\s+", txt.strip())
    return " ".join(parts[:max(1, n_sentences)])

def claimify(question: str, answer: str) -> str:
    """Turn (Q, span-answer) into a declarative hypothesis for NLI."""
    q = (question or "").strip().lower()
    a = (answer or "").strip()
    if not a:
        return "The answer is unknown."
    if q.startswith("who wrote"):
        return f"{a} wrote the work."
    if q.startswith("who "):
        return f"{a} is the person in question."
    if "what year" in q or q.startswith("when "):
        return f"It happened in {a}."
    if q.startswith("where "):
        return f"It happened in {a}."
    if q.startswith(("what is", "what was")):
        return f"It is {a}."
    if q.startswith("which "):
        return f"It is {a}."
    # fallback
    return f"Answer: {a}"

def _span_support(answer: str, snippets: List[Dict]) -> bool:
    """Fast-path: if short span (<=3 tokens) appears in any snippet, count supported."""
    a = re.sub(r"\s+", " ", (answer or "").strip().lower())
    if not a or len(a.split()) > 3:
        return False
    for s in snippets or []:
        if a in (s.get("text", "").lower()):
            return True
    return False

# =========================
# Evidence: disk cache
# =========================
EVID_CACHE_PATH = "runs/evidence_cache.json"
os.makedirs("runs", exist_ok=True)
try:
    _EVID_CACHE = json.load(open(EVID_CACHE_PATH, "r", encoding="utf-8"))
except Exception:
    _EVID_CACHE = {}

def _evid_key(question: str, answer: str, k: int) -> str:
    return hashlib.sha1(f"{question}::{answer}::k={k}".encode()).hexdigest()

def _evid_get(q: str, a: str, k: int):
    return _EVID_CACHE.get(_evid_key(q, a, k))

def _evid_put(q: str, a: str, k: int, val):
    _EVID_CACHE[_evid_key(q, a, k)] = val
    try:
        with open(EVID_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_EVID_CACHE, f)
    except Exception:
        pass

# =========================
# Wikipedia retrieval (summaries only)
# =========================
@lru_cache(maxsize=512)
def _search_titles(query: str, n: int) -> List[str]:
    try:
        return wikipedia.search(query, results=n) or []
    except Exception:
        return []

@lru_cache(maxsize=512)
def _wiki_summary(title: str, sentences: int) -> str:
    try:
        summ = wikipedia.summary(title, sentences=sentences, auto_suggest=False)
        return _clean_text(summ)
    except Exception:
        return ""

def retrieve_evidence(question: str, answer: str, k: int = 3) -> List[Dict]:
    """Search Q and A surface forms, fetch short wiki summaries, dedupe, cache."""
    cached = _evid_get(question, answer, k)
    if cached:
        return cached

    queries = [question, answer]
    out: List[Dict] = []
    seen_txt = set()

    for q in queries:
        if len(out) >= k:
            break
        for title in _search_titles(q, n=WIKI_RESULTS_PER_QUERY):
            if len(out) >= k:
                break
            snip = _wiki_summary(title, sentences=WIKI_SENTENCES)
            if not snip:
                continue
            snip = _clean_text(snip)
            if snip in seen_txt:
                continue
            out.append({"source": "wikipedia", "title": title, "text": snip})
            seen_txt.add(snip)

    out = out[:k]
    _evid_put(question, answer, k, out)
    return out

# =========================
# NLI verifier 
# =========================
_nli = pipeline(
    "text-classification",
    model=NLI_MODEL_ID,
    return_all_scores=True,   # we keep both entail & contradict
    device_map="auto",
)

def _normalize_scores(res) -> Dict[str, float]:
    # Normalize dict | [dict] | [[dict,...]] -> {"entail","neutral","contradict"}
    if isinstance(res, dict):
        scores_list = [res]
    elif isinstance(res, list):
        scores_list = res[0] if (res and isinstance(res[0], list)) else res
    else:
        scores_list = []

    out = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
    for d in scores_list:
        if isinstance(d, dict) and "label" in d and "score" in d:
            lab = d["label"].lower()
            if lab in out:
                out[lab] = float(d["score"])
    return {"entail": out["entailment"], "neutral": out["neutral"], "contradict": out["contradiction"]}

def _nli_batch(premises: List[str], hypothesis: str) -> List[Dict]:
    inputs = [{"text": p, "text_pair": hypothesis} for p in premises]
    results = _nli(inputs)  # batched by HF
    return [_normalize_scores(r) for r in results]

def nli_pair(premise: str, hypothesis: str) -> Dict:
    # Kept for API compatibility; used in some UIs
    return _normalize_scores(_nli({"text": premise, "text_pair": hypothesis}))

def best_verdict(question: str, answer: str, snippets: List[Dict], tau: float = 0.60) -> Dict:
    """Verdict from max entail vs max contradict across snippets."""
    if not snippets:
        return {"label": "unverifiable", "confidence": 0.0, "max_entail": 0.0, "max_contradict": 0.0, "evidence": None}

    # Short-span fast path: if the span literally appears in evidence, treat as supported.
    if _span_support(answer, snippets):
        return {"label": "supported", "confidence": 0.7, "max_entail": 0.7, "max_contradict": 0.0, "evidence": snippets[0]}

    premises = [s.get("text", "") for s in snippets if s.get("text")]
    if not premises:
        return {"label": "unverifiable", "confidence": 0.0, "max_entail": 0.0, "max_contradict": 0.0, "evidence": None}

    hypothesis = claimify(question, answer) if USE_CLAIMIFY else answer
    results = _nli_batch(premises, hypothesis)

    max_ent, max_con, best_idx = 0.0, 0.0, -1
    for i, sc in enumerate(results):
        ent = sc.get("entail", 0.0)
        con = sc.get("contradict", 0.0)
        if ent > max_ent:
            max_ent, best_idx = ent, i
        if con > max_con:
            max_con = con

    conf = max_ent - max_con
    if max_con >= tau and max_con > max_ent:
        label = "contradicted"
    elif max_ent >= tau and max_ent > max_con:
        label = "supported"
    else:
        label = "unverifiable"

    return {
        "label": label,
        "confidence": round(abs(conf), 3),
        "max_entail": round(max_ent, 3),
        "max_contradict": round(max_con, 3),
        "evidence": snippets[best_idx] if best_idx != -1 else None,
    }
