# eval.py
import os, json, time, argparse, csv, re, hashlib
from collections import Counter
from pathlib import Path

from models import generate_answer, MODEL_ID
from detector import retrieve_evidence, best_verdict  # NOTE: new signature: best_verdict(question, answer, evid)

# ---------- Text utils ----------
def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    return re.sub(r"\s+", " ", s)

def token_f1(pred: str, gold: str) -> float:
    ps = normalize_text(pred).split()
    gs = normalize_text(gold).split()
    if not ps or not gs:
        return 0.0
    from collections import Counter
    pset, gset = Counter(ps), Counter(gs)
    overlap = sum((pset & gset).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, len(ps))
    recall    = overlap / max(1, len(gs))
    return 2 * precision * recall / max(precision + recall, 1e-9)

def loose_correct(pred: str, gold: str, thresh: float = 0.6) -> bool:
    # exact match, substring, or token-F1 above threshold
    p, g = normalize_text(pred), normalize_text(gold)
    if p == g:
        return True
    if g and g in p:
        return True
    return token_f1(p, g) >= thresh

# ---------- Generation cache ----------
CACHE_PATH = "runs/gen_cache.json"
def _load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f: 
            return json.load(f)
    return {}
def _save_cache(c):
    Path("runs").mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(c, f)
def _key(q: str) -> str:
    return hashlib.sha1(f"{MODEL_ID}::{q}".encode()).hexdigest()

# ---------- Runner ----------
def run_eval(bench_path: str, out_dir: str, k: int = 3, limit: int = 0):
    # Load JSONL
    bench = []
    with open(bench_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            bench.append(json.loads(line))
    if limit:
        bench = bench[:limit]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = Path(out_dir) / f"results_{run_tag}_{MODEL_ID.replace('/','_')}.jsonl"
    csv_path   = Path(out_dir) / f"metrics_{run_tag}_{MODEL_ID.replace('/','_')}.csv"

    rows = []
    label_counts = Counter()
    n_exact = n_loose = n_soft = n = 0
    n_support_hit = 0
    t_ret = t_ver = 0.0

    gen_cache = _load_cache()

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for idx, ex in enumerate(bench, 1):
            q     = ex["question"]
            gold  = ex.get("gold_answer", "")
            dom   = ex.get("domain", "")

            # Generate (cached)
            ck = _key(q)
            if ck in gen_cache:
                ans = gen_cache[ck]
            else:
                ans = generate_answer(q)
                gen_cache[ck] = ans
                _save_cache(gen_cache)

            # Retrieve + Verify
            t1 = time.time()
            evid = retrieve_evidence(q, ans, k=k)
            t2 = time.time()
            verdict = best_verdict(q, ans, evid)  # << pass question too
            t3 = time.time()

            exact = int(normalize_text(ans) == normalize_text(gold))
            loose = int(loose_correct(ans, gold))
            # Soft correctness: consider "supported" by evidence as soft-correct
            soft  = int(verdict["label"] == "supported")

            support_hit = 0
            gnorm = normalize_text(gold)
            for s in evid or []:
                if gnorm and gnorm in normalize_text(s.get("text","")):
                    support_hit = 1
                    break

            n += 1
            n_exact += exact
            n_loose += loose
            n_soft  += soft
            n_support_hit += support_hit
            label_counts[verdict["label"]] += 1

            t_ret += (t2 - t1)
            t_ver += (t3 - t2)

            rec = {
                "id": ex.get("id", idx),
                "domain": dom,
                "question": q,
                "gold_answer": gold,
                "model": MODEL_ID,
                "answer": ans,
                "label": verdict["label"],
                "confidence": verdict.get("confidence", 0.0),
                "max_entail": verdict.get("max_entail", 0.0),
                "max_contradict": verdict.get("max_contradict", 0.0),
                "supported_gold_in_evidence": support_hit,
                "retrieved_titles": [s.get("title") for s in (evid or [])][:5],
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if idx % 5 == 0:
                print(f"[{idx}/{len(bench)}] {dom or '—'} | exact={exact} loose={loose} soft={soft} | label={rec['label']}")

    # Aggregate metrics
    exact_acc    = n_exact / max(n, 1)
    loose_acc    = n_loose / max(n, 1)
    soft_acc     = n_soft  / max(n, 1)
    support_rate = n_support_hit / max(n, 1)
    avg_ret_ms   = (t_ret / max(n, 1)) * 1000
    avg_ver_ms   = (t_ver / max(n, 1)) * 1000

    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow([
            "model","n","exact_acc","loose_acc","soft_acc(evidence-supported)","support@k",
            "label_supported","label_contradicted","label_unverifiable",
            "avg_retrieval_ms","avg_verify_ms"
        ])
        w.writerow([
            MODEL_ID, n, f"{exact_acc:.3f}", f"{loose_acc:.3f}", f"{soft_acc:.3f}", f"{support_rate:.3f}",
            label_counts["supported"], label_counts["contradicted"], label_counts["unverifiable"],
            int(avg_ret_ms), int(avg_ver_ms)
        ])

    print("\n==== Summary ====")
    print(f"Model: {MODEL_ID}")
    print(f"Items: {n}")
    print(f"Exact Acc:   {exact_acc:.3f}")
    print(f"Loose Acc:   {loose_acc:.3f}")
    print(f"Soft Acc:    {soft_acc:.3f}   (NLI/evidence-supported)")
    print(f"Support@{k}: {support_rate:.3f}")
    print(f"Labels:      {dict(label_counts)}")
    print(f"Wrote per-item to: {jsonl_path}")
    print(f"Wrote summary CSV: {csv_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="bench/questions.jsonl", help="Path to JSONL benchmark")
    ap.add_argument("--out",   default="runs", help="Output directory")
    ap.add_argument("--k",     type=int, default=3, help="Top-k evidence to retrieve")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only first N items (debug)")
    args = ap.parse_args()
    run_eval(args.bench, args.out, k=args.k, limit=args.limit)

if __name__ == "__main__":
    main()





