# eval.py
import json, pandas as pd
from tqdm import tqdm
from detector import retrieve_evidence, best_verdict
from mitigator import regenerate_with_sources
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def baseline_answer(q: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":"Answer concisely."},{"role":"user","content":q}],
        temperature=0.7
    )
    return r.choices[0].message.content.strip()

prompts = json.load(open("data/prompts.json"))
rows = []
for q in tqdm(prompts):
    ans = baseline_answer(q)
    evid = retrieve_evidence(q, ans, k=3)
    v_base = best_verdict(ans, evid)

    mit_ans, v_mit = None, None
    if v_base["label"] != "supported":
        cands = regenerate_with_sources(q, evid, n=3)
        # pick best by confidence
        scores = [(best_verdict(c["text"], evid), c["text"]) for c in cands]
        scores.sort(key=lambda t: t[0]["confidence"], reverse=True)
        v_mit, mit_ans = scores[0][0], scores[0][1]
    rows.append({
        "question": q,
        "answer": ans,
        "base_label": v_base["label"],
        "base_conf": v_base["confidence"],
        "mit_answer": mit_ans,
        "mit_label": v_mit["label"] if v_mit else None,
        "mit_conf": v_mit["confidence"] if v_mit else None
    })

df = pd.DataFrame(rows)
df.to_csv("results.csv", index=False)
print(df["base_label"].value_counts(normalize=True))
print(df["mit_label"].value_counts(normalize=True))
