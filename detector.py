# detector.py
from typing import List, Dict, Tuple
import wikipedia
from transformers import pipeline
from functools import lru_cache
import re

# --- NLI model (MNLI) ---
_nli = pipeline("text-classification", model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33", return_all_scores=True)

@lru_cache(maxsize=256)
def fetch_wikipedia_snippet(query: str, n_sentences: int = 5) -> str:
    try:
        page_title = wikipedia.search(query, results=1)[0]
        page = wikipedia.page(page_title, auto_suggest=False)
        text = re.sub(r"\s+", " ", page.content).strip()
        return " ".join(text.split(". ")[:n_sentences])
    except Exception:
        return ""

def retrieve_evidence(question: str, answer: str, k: int = 3) -> List[Dict]:
    # simple heuristic: query with key nouns from Q and A; fetch a few snippets
    queries = [question, answer]
    snippets = []
    seen = set()
    for q in queries:
        snip = fetch_wikipedia_snippet(q)
        if snip and snip not in seen:
            snippets.append({"source": "wikipedia", "title": q, "text": snip})
            seen.add(snip)
        if len(snippets) >= k:
            break
    return snippets

def nli_pair(premise: str, hypothesis: str) -> Dict:
    # premise = evidence, hypothesis = claim
    scores = _nli({"text": premise, "text_pair": hypothesis})[0]
    # normalize to dict
    out = {s["label"].lower(): s["score"] for s in scores}
    # map labels to entail/neutral/contradiction (depends on model labels)
    entail = out.get("entailment", out.get("entailed", 0.0))
    neutral = out.get("neutral", 0.0)
    contra = out.get("contradiction", out.get("contradictory", 0.0))
    return {"entail": entail, "neutral": neutral, "contradict": contra}

def best_verdict(answer: str, snippets: List[Dict], tau: float = 0.60) -> Dict:
    """
    Compare answer vs each snippet; take the max entail and max contradict.
    Confidence = max(entail) - max(contradict)
    """
    max_ent, max_con, best_evidence = 0.0, 0.0, None
    for s in snippets:
        scores = nli_pair(s["text"], answer)
        if scores["entail"] > max_ent:
            max_ent, best_evidence = scores["entail"], s
        if scores["contradict"] > max_con:
            max_con = scores["contradict"]
    conf = max_ent - max_con
    if max_con >= tau and max_con > max_ent:
        label = "contradicted"
    elif max_ent >= tau and max_ent > max_con:
        label = "supported"
    else:
        label = "unverifiable"
    return {
        "label": label,
        "confidence": round(float(abs(conf)), 3),
        "max_entail": round(float(max_ent), 3),
        "max_contradict": round(float(max_con), 3),
        "evidence": best_evidence
    }
