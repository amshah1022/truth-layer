# app.py
import streamlit as st
from detector import retrieve_evidence, best_verdict
from mitigator import regenerate_with_sources
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Truth Layer", page_icon="✅")
st.title("Truth Layer – Detect ➜ Visualize ➜ Mitigate")

# model for baseline answer (can use same OpenAI client; or any other)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def baseline_answer(q: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":"Answer concisely."},{"role":"user","content":q}],
        temperature=0.7
    )
    return r.choices[0].message.content.strip()

q = st.text_input("Ask a question")
if st.button("Answer") and q:
    with st.spinner("Generating answer..."):
        ans = baseline_answer(q)
        st.subheader("Model Answer")
        st.write(ans)

        st.subheader("Evidence")
        evid = retrieve_evidence(q, ans, k=3)
        for i, e in enumerate(evid, 1):
            with st.expander(f"[S{i}] {e['title']} (wikipedia)"):
                st.write(e["text"])

        st.subheader("Verifier Verdict")
        verdict = best_verdict(ans, evid)
        badge = "✅ Supported" if verdict["label"]=="supported" else ("❌ Contradicted" if verdict["label"]=="contradicted" else "⚠️ Unverifiable")
        st.write(f"**{badge}** | confidence: {verdict['confidence']} | max_entail: {verdict['max_entail']} | max_contradict: {verdict['max_contradict']}")

        st.divider()
        st.subheader("Mitigation")
        if verdict["label"] != "supported":
            if st.button("Mitigate (regenerate with sources)"):
                with st.spinner("Mitigating..."):
                    cands = regenerate_with_sources(q, evid, n=3)
                    # score each candidate with the same verifier and pick best
                    scored = []
                    for c in cands:
                        v2 = best_verdict(c["text"], evid)
                        scored.append((v2["confidence"], v2, c["text"]))
                    scored.sort(reverse=True, key=lambda x: x[0])
                    best_conf, best_v, best_txt = scored[0]
                    st.markdown("**Best Mitigated Answer:**")
                    st.write(best_txt)
                    st.write(f"Post-mitigation verdict: **{best_v['label']}** (confidence {best_v['confidence']})")
        else:
            st.info("Answer already supported by evidence. No mitigation needed.")
