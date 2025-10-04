from typing import List, Dict
import os
from textwrap import shorten
from dotenv import load_dotenv

load_dotenv()

# OpenAI client (adjust if you use a different SDK)
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_source_block(snippets: List[Dict]) -> str:
    lines = []
    for i, s in enumerate(snippets, 1):
        txt = shorten(s.get("text", ""), width=500, placeholder="…")
        lines.append(f"[S{i}] {txt}")
    return "\n".join(lines)

def regenerate_with_sources(question: str, sources: List[Dict], n: int = 3) -> List[Dict]:
    """
    Regenerate an answer constrained to provided sources. Returns a list of dicts [{"text": "..."}].
    """
    if not sources:
        # no evidence: return a refusal-style answer
        return [{"text": "Insufficient evidence in the provided sources to answer reliably."}]

    src = build_source_block(sources)
    system = (
        "You are a factual assistant. Answer ONLY using the provided sources. "
        "Cite sources inline like [S1], [S2]. If information is not present in sources, say 'Insufficient evidence.'"
    )
    user = f"Question: {question}\n\nSources:\n{src}\n\nAnswer:"

    outs = []
    for _ in range(n):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2
        )
        outs.append({"text": resp.choices[0].message.content.strip()})
    return outs
