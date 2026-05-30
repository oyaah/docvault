"""Calibrate the confidence gate (settings.confidence_threshold).

The gate decides when DocVault refuses ("I don't know"). It currently defaults to
0.0 (effectively off), which is why refusal on unanswerable questions is weak.

This sweeps the raw top reranker score over the benchmark question set, labelling
each question answerable vs not, and picks the threshold that best separates them
(max Youden's J = TPR − FPR, where a "positive" = should-refuse). Uses the
generation-free `pipe.retrieve_scores()` path, so it costs only query embeddings —
no LLM generation.

Usage:
    python benchmark/calibrate_gate.py
    python benchmark/calibrate_gate.py --questions benchmark/data/questions.jsonl
Then set the printed value:  export DOCVAULT_CONFIDENCE_THRESHOLD=<value>
"""

import argparse
import json
from pathlib import Path

import numpy as np

from docvault.config import settings
from docvault.pipeline import DocVaultPipeline


def main():
    ap = argparse.ArgumentParser(description="Calibrate the confidence gate threshold.")
    ap.add_argument("--questions", default=str(Path(__file__).parent / "data" / "questions.jsonl"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "gate_scores.jsonl"))
    args = ap.parse_args()

    settings.ensure_dirs()
    questions = [json.loads(l) for l in Path(args.questions).read_text().splitlines() if l.strip()]
    pipe = DocVaultPipeline()

    rows = []
    out = Path(args.out).open("w")
    for i, q in enumerate(questions, 1):
        try:
            score = pipe.retrieve_scores(q["question"])
        except Exception as e:
            print(f"  skip {q['id']}: {type(e).__name__}")
            continue
        should_refuse = not q.get("answerable", True)
        rows.append((score, should_refuse))
        out.write(json.dumps({"id": q["id"], "top_score": score, "should_refuse": should_refuse}) + "\n")
        print(f"[{i}/{len(questions)}] {'REFUSE' if should_refuse else 'answer':>7}  score={score:.3f}")
    out.close()

    if not rows:
        raise SystemExit("no scores collected")

    scores = np.array([r[0] for r in rows])
    refuse = np.array([r[1] for r in rows])
    pos, neg = refuse.sum(), (~refuse).sum()

    best = None
    for thr in np.unique(scores):
        # gate: top_score < thr -> refuse
        tp = int(((scores < thr) & refuse).sum())
        fp = int(((scores < thr) & ~refuse).sum())
        tpr = tp / pos if pos else 0.0
        fpr = fp / neg if neg else 0.0
        j = tpr - fpr
        if best is None or j > best["j"]:
            best = {"threshold": float(thr), "j": j, "tpr": tpr, "fpr": fpr}

    print("\n" + "=" * 50)
    print(f"answerable={neg}  unanswerable={pos}")
    print(f"score range: {scores.min():.3f} … {scores.max():.3f}")
    print(f"\nBest threshold = {best['threshold']:.3f}")
    print(f"  refuses {best['tpr']*100:.0f}% of unanswerable (TPR)")
    print(f"  wrongly refuses {best['fpr']*100:.0f}% of answerable (FPR)")
    print(f"\n  export DOCVAULT_CONFIDENCE_THRESHOLD={best['threshold']:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
