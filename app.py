# app.py
import os
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv

# Local modules
from models import generate_answer, MODEL_ID  # HF baseline model
from detector import retrieve_evidence, best_verdict

# Optional OpenAI mitigation (only if key present + mitigator import works)
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
HAS_OPENAI = bool(OPENAI_KEY)
try:
    if HAS_OPENAI:
        from mitigator import regenerate_with_sources  # OpenAI-based mitigation
    else:
        regenerate_with_sources = None
except Exception:
    regenerate_with_sources = None
    HAS_OPENAI = False

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Truth Layer", page_icon="✅", layout="centered")
st.title("Truth Layer – Detect ➜ Visualize ➜ Mitigate")
st.caption(f"Baseline model: **{MODEL_ID}**  |  Mitigation: {'OpenAI (if needed)' if HAS_OPENAI else 'HF grounded rewrite'}")

def grounded_rewrite_with_sources(question: str, sources: List[Dict]) -> str:
    """
    Simple, API-free mitigation:
    stitch a concise answer from top evidence with inline [S#] citations.
    """
    if not sources:
        return "Insufficient evidence in the provided sources."
    bits = []
    for i, s in enumerate(sources[:2], 1):
        txt = (s.get("text") or "").strip()
        if txt:
            # keep it short and readable
            snippet = txt[:240].rsplit(" ", 1)[0]
            bits.append(f"{snippet} [S{i}]")
    return " ".join(bits) if bits else "Insufficient evidence in the provided sources."

# ----------------------------
# App flow
# ----------------------------
q = st.text_input("Ask a question")
if st.button("Answer") and q:
    with st.spinner("Generating answer..."):
        ans = generate_answer(q)

    st.subheader("Model Answer")
    st.write(ans)

    st.subheader("Evidence")
    evid = retrieve_evidence(q, ans, k=3)
    for i, e in enumerate(evid, 1):
        title = e.get("title") or e.get("source") or f"S{i}"
        with st.expander(f"[S{i}] {title}"):
            st.write(e.get("text", ""))

    st.subheader("Verifier Verdict")
    verdict = best_verdict(ans, evid)
    badge = (
        "Supported" if verdict["label"] == "supported"
        else "Contradicted" if verdict["label"] == "contradicted"
        else "Unverifiable"
    )
    st.write(
        f"**{badge}** | confidence: {verdict['confidence']} "
        f"| max_entail: {verdict['max_entail']} | max_contradict: {verdict['max_contradict']}"
    )

    st.divider()
    st.subheader("Mitigation")
    if verdict["label"] != "supported":
        if st.button("Mitigate (regenerate with sources)"):
            with st.spinner("Mitigating..."):
                # Path A: OpenAI-based mitigation if available
                if regenerate_with_sources is not None:
                    cands = regenerate_with_sources(q, evid, n=3)
                    scored = []
                    for c in cands:
                        v2 = best_verdict(c["text"], evid)
                        scored.append((v2["confidence"], v2, c["text"]))
                    scored.sort(reverse=True, key=lambda x: x[0])
                    best_conf, best_v, best_txt = scored[0]
                    st.markdown("**Best Mitigated Answer:**")
                    st.write(best_txt)
                    st.write(
                        f"Post-mitigation verdict: **{best_v['label']}** "
                        f"(confidence {best_v['confidence']})"
                    )
                else:
                    # Path B: API-free grounded rewrite using retrieved evidence
                    best_txt = grounded_rewrite_with_sources(q, evid)
                    v2 = best_verdict(best_txt, evid)
                    st.markdown("**Best Mitigated Answer (HF grounded):**")
                    st.write(best_txt)
                    st.write(
                        f"Post-mitigation verdict: **{v2['label']}** "
                        f"(confidence {v2['confidence']})"
                    )
    else:
        st.info("Answer already supported by evidence. No mitigation needed.")
