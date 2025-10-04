#!/usr/bin/env python3
import argparse, json, os, math, random, csv
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

# --------------------------
# Helpers
# --------------------------

PUNCT = set(" .,:;!?\"'()[]{}")

def normalize_span(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    # lower + strip common punctuation at ends + collapse spaces
    s = s.lower().strip("".join(PUNCT))
    s = " ".join(s.split())
    return s

def exact_match(ans: str, gold: str) -> int:
    return int(ans == gold)

def loose_match(ans: str, gold: str) -> int:
    # "loose": case/space/punct normalized
    return int(normalize_span(ans) == normalize_span(gold))

def bootstrap_ci(binary_list: List[int], n_boot: int = 10000, alpha: float = 0.05) -> Tuple[float,float,float]:
    if not binary_list:
        return (float('nan'), float('nan'), float('nan'))
    n = len(binary_list)
    base = sum(binary_list)/n
    if n == 0:
        return (float('nan'), float('nan'), float('nan'))
    # fast vectorless bootstrap
    rng = random.Random(17)
    stats = []
    for _ in range(n_boot):
        s = 0
        for _i in range(n):
            s += binary_list[rng.randrange(n)]
        stats.append(s/n)
    stats.sort()
    lo = stats[int((alpha/2)*n_boot)]
    hi = stats[int((1-alpha/2)*n_boot)]
    return (base, lo, hi)

def mcnemar(a_correct: List[int], b_correct: List[int]) -> Tuple[int,int,float]:
    """
    Exact binomial McNemar (no continuity corr) on paired 0/1 lists.
    Returns (b01, b10, p_value) where:
      b01: A wrong, B right
      b10: A right, B wrong
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("Lists must be the same length for McNemar.")
    b01 = b10 = 0
    for ac, bc in zip(a_correct, b_correct):
        if ac == 0 and bc == 1: b01 += 1
        elif ac == 1 and bc == 0: b10 += 1
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    # exact binomial two-sided p = 2*min(Bin(k; n, 0.5), 1 - Bin(k-1; n, 0.5))
    # compute tail at min(b01,b10)
    k = min(b01, b10)
    from math import comb
    tail = sum(comb(n, i) for i in range(0, k+1)) / (2**n)
    p = 2*tail
    p = min(1.0, p)
    return b01, b10, p

# --------------------------
# Loading
# --------------------------

def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def summarize_run(rows: List[dict], label_field: str = "label") -> dict:
    exact = []
    loose = []
    soft  = []   # label == 'supported'
    recall_any = []  # proxy for retriever Recall@k: did any passage contain gold?
    domains = defaultdict(lambda: {"exact": [], "loose": [], "soft": [], "recall": []})

    for r in rows:
        gold = r.get("gold_answer","")
        ans  = r.get("answer","")
        lab  = (r.get(label_field) or "").lower()
        exact.append(exact_match(ans, gold))
        loose.append(loose_match(ans, gold))
        soft.append(int(lab == "supported"))
        recall_any.append(int(r.get("supported_gold_in_evidence", 0) == 1))

        d = r.get("domain", "unknown")
        domains[d]["exact"].append(exact[-1])
        domains[d]["loose"].append(loose[-1])
        domains[d]["soft"].append(soft[-1])
        domains[d]["recall"].append(recall_any[-1])

    def pack(xs):
        mean, lo, hi = bootstrap_ci(xs)
        return {"mean": round(mean,3), "ci95": [round(lo,3), round(hi,3)], "n": len(xs)}

    per_domain = {}
    for d, dd in domains.items():
        per_domain[d] = {
            "exact": pack(dd["exact"]),
            "loose": pack(dd["loose"]),
            "soft":  pack(dd["soft"]),
            "recall_any": pack(dd["recall"]),
            "n": len(dd["exact"]),
        }

    overall = {
        "exact": pack(exact),
        "loose": pack(loose),
        "soft":  pack(soft),
        "recall_any": pack(recall_any),
        "n": len(rows),
        "label_counts": dict(Counter([(r.get(label_field) or "").lower() for r in rows]))
    }
    return {"overall": overall, "by_domain": per_domain}

def index_by_id(rows: List[dict]) -> Dict[int, dict]:
    out = {}
    for r in rows:
        out[int(r["id"])] = r
    return out

# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="Paths to results_*.jsonl from different models (Core-100 or any matched cohort).")
    ap.add_argument("--outdir", default="tables", help="Directory to write CSV summaries.")
    ap.add_argument("--pairwise", action="store_true", help="Run McNemar pairwise across supplied models.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    runs = []
    for p in args.models:
        rows = load_jsonl(p)
        # name: try to get from first row's 'model' or filename
        name = rows[0].get("model") if rows and rows[0].get("model") else os.path.splitext(os.path.basename(p))[0]
        runs.append((name, rows, p))

    # Per-model summaries
    per_model_csv = os.path.join(args.outdir, "per_model_summary.csv")
    with open(per_model_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","n",
                    "exact_mean","exact_lo","exact_hi",
                    "loose_mean","loose_lo","loose_hi",
                    "soft_mean","soft_lo","soft_hi",
                    "recall_mean","recall_lo","recall_hi",
                    "labels_supported","labels_contradicted","labels_unverifiable"])
        for name, rows, _ in runs:
            summ = summarize_run(rows)
            ov = summ["overall"]
            lab = ov["label_counts"]
            w.writerow([
                name, ov["n"],
                ov["exact"]["mean"], ov["exact"]["ci95"][0], ov["exact"]["ci95"][1],
                ov["loose"]["mean"], ov["loose"]["ci95"][0], ov["loose"]["ci95"][1],
                ov["soft"]["mean"],  ov["soft"]["ci95"][0],  ov["soft"]["ci95"][1],
                ov["recall_any"]["mean"], ov["recall_any"]["ci95"][0], ov["recall_any"]["ci95"][1],
                lab.get("supported",0), lab.get("contradicted",0), lab.get("unverifiable",0)
            ])

    # Per-domain CSVs
    per_domain_csv = os.path.join(args.outdir, "per_domain_summary.csv")
    with open(per_domain_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","domain","n",
                    "exact_mean","exact_lo","exact_hi",
                    "loose_mean","loose_lo","loose_hi",
                    "soft_mean","soft_lo","soft_hi",
                    "recall_mean","recall_lo","recall_hi"])
        for name, rows, _ in runs:
            summ = summarize_run(rows)
            for dom, dv in summ["by_domain"].items():
                w.writerow([
                    name, dom, dv["n"],
                    dv["exact"]["mean"], dv["exact"]["ci95"][0], dv["exact"]["ci95"][1],
                    dv["loose"]["mean"], dv["loose"]["ci95"][0], dv["loose"]["ci95"][1],
                    dv["soft"]["mean"],  dv["soft"]["ci95"][0],  dv["soft"]["ci95"][1],
                    dv["recall_any"]["mean"], dv["recall_any"]["ci95"][0], dv["recall_any"]["ci95"][1],
                ])

    print(f"[ok] wrote: {per_model_csv}")
    print(f"[ok] wrote: {per_domain_csv}")

    # Pairwise McNemar on EXACT and SOFT
    if args.pairwise and len(runs) >= 2:
        pair_csv = os.path.join(args.outdir, "pairwise_mcnemar.csv")
        with open(pair_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model_A","model_B","metric","n_shared","A_wrong_B_right","A_right_B_wrong","p_value"])
            for i in range(len(runs)):
                for j in range(i+1, len(runs)):
                    nameA, rowsA, _ = runs[i]
                    nameB, rowsB, _ = runs[j]
                    idxA = index_by_id(rowsA)
                    idxB = index_by_id(rowsB)
                    shared_ids = sorted(set(idxA.keys()) & set(idxB.keys()))
                    if not shared_ids:
                        continue

                    # EXACT
                    a_exact = [exact_match(idxA[k]["answer"], idxA[k]["gold_answer"]) for k in shared_ids]
                    b_exact = [exact_match(idxB[k]["answer"], idxB[k]["gold_answer"]) for k in shared_ids]
                    b01, b10, p = mcnemar(a_exact, b_exact)
                    w.writerow([nameA, nameB, "exact", len(shared_ids), b01, b10, round(p,6)])

                    # SOFT (supported)
                    a_soft = [int((idxA[k].get("label","").lower()) == "supported") for k in shared_ids]
                    b_soft = [int((idxB[k].get("label","").lower()) == "supported") for k in shared_ids]
                    b01, b10, p = mcnemar(a_soft, b_soft)
                    w.writerow([nameA, nameB, "soft", len(shared_ids), b01, b10, round(p,6)])

        print(f"[ok] wrote: {pair_csv}")

if __name__ == "__main__":
    main()
