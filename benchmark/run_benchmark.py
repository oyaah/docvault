"""Run the DocVault benchmark: query the live pipeline, judge with an LLM, and
report metrics with bootstrap confidence intervals and pass/fail gates.

Does NOT use exact-match. Retrieval + generation + behaviour are all scored by
an LLM judge (see judges.py). Results are written incrementally so --resume can
continue an interrupted run.

Usage:
    python benchmark/run_benchmark.py                       # full run
    python benchmark/run_benchmark.py --limit 20            # smoke test
    python benchmark/run_benchmark.py --judge-model gpt-4o  # pick judge
"""

import argparse
import json
import statistics
import time
import warnings
from pathlib import Path

# litellm serializes responses through pydantic, emitting harmless "Unexpected
# field" UserWarnings — silence them so run output stays readable.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")

import numpy as np

from docvault.config import settings
from docvault.pipeline import DocVaultPipeline

import judges

GROUNDED = {"factoid", "multi_hop", "comparative", "paraphrase", "table_lookup"}


def evaluate_one(q: dict, pipe: DocVaultPipeline, judge_model: str) -> dict:
    t0 = time.time()
    res = pipe.query(q["question"])
    latency_ms = (time.time() - t0) * 1000

    answer = res.get("answer", "")
    contexts = res.get("contexts", [])
    answerable = q.get("answerable", True)

    raw: dict[str, float | None] = {}
    raw["refusal_correctness"] = judges.judge_refusal_correctness(answer, answerable)["score"]

    refused = judges.is_refusal(answer)

    if answerable:
        raw["context_precision"] = judges.judge_context_precision(q["question"], contexts, judge_model)["score"]
        raw["context_recall"] = judges.judge_context_recall(q["reference_answer"], contexts, judge_model)["score"]
        raw["faithfulness"] = judges.judge_faithfulness(answer, contexts, judge_model)["score"]
        raw["answer_relevancy"] = judges.judge_answer_relevancy(q["question"], answer, judge_model)["score"]
        raw["answer_correctness"] = judges.judge_answer_correctness(
            q["question"], answer, q["reference_answer"], judge_model)["score"]
        if not refused and not judges.has_citation(answer):
            # an answerable answer that cites nothing is a citation failure (spec: cite every claim)
            raw["citation_accuracy"] = 0.0
        else:
            raw["citation_accuracy"] = judges.judge_citation_accuracy(answer, contexts, judge_model)["score"]
    elif not refused:
        # It answered something it shouldn't have — measure how ungrounded it is.
        raw["faithfulness"] = judges.judge_faithfulness(answer, contexts, judge_model)["score"]

    # Drop None (failed/N-A judge calls) so they don't pollute aggregates.
    scores = {k: v for k, v in raw.items() if v is not None}

    return {
        "id": q["id"],
        "category": q["category"],
        "answerable": answerable,
        "question": q["question"],
        "answer": answer,
        "confidence": res.get("confidence"),
        "n_contexts": len(contexts),
        "latency_ms": round(latency_ms, 1),
        "scores": scores,
        "n_judge_failures": len(raw) - len(scores),
    }


def bootstrap_ci(values: list[float], iters: int = 2000) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    arr = np.array(values, dtype=float)
    means = [arr[np.random.randint(0, len(arr), len(arr))].mean() for _ in range(iters)]
    return (round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3))


def aggregate(results: list[dict]) -> dict:
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy",
               "answer_correctness", "citation_accuracy", "refusal_correctness"]
    overall = {}
    for m in metrics:
        vals = [r["scores"][m] for r in results if m in r["scores"]]
        if vals:
            overall[m] = {"mean": round(statistics.mean(vals), 3), "n": len(vals), "ci95": bootstrap_ci(vals)}

    faith = [r["scores"]["faithfulness"] for r in results if "faithfulness" in r["scores"]]
    hallucination_rate = round(1 - statistics.mean(faith), 3) if faith else None

    refusal_unanswerable = [r["scores"]["refusal_correctness"] for r in results
                            if not r["answerable"]]
    by_cat: dict[str, dict] = {}
    cats = sorted({r["category"] for r in results})
    for cat in cats:
        sub = [r for r in results if r["category"] == cat]
        cat_metrics = {}
        for m in metrics:
            vals = [r["scores"][m] for r in sub if m in r["scores"]]
            if vals:
                cat_metrics[m] = round(statistics.mean(vals), 3)
        by_cat[cat] = {"n": len(sub), **cat_metrics}

    lat = [r["latency_ms"] for r in results]
    return {
        "n": len(results),
        "judge_failures": sum(r.get("n_judge_failures", 0) for r in results),
        "overall": overall,
        "hallucination_rate": hallucination_rate,
        "refusal_correctness_unanswerable": round(statistics.mean(refusal_unanswerable), 3) if refusal_unanswerable else None,
        "latency_ms": {
            "p50": round(float(np.percentile(lat, 50)), 1) if lat else None,
            "p95": round(float(np.percentile(lat, 95)), 1) if lat else None,
        },
        "by_category": by_cat,
    }


def check_gates(agg: dict, args) -> list[tuple[str, bool, str]]:
    gates = []
    faith = agg["overall"].get("faithfulness", {}).get("mean")
    if faith is not None:
        gates.append(("faithfulness >= %.2f" % args.min_faithfulness, faith >= args.min_faithfulness, f"{faith}"))
    if agg["hallucination_rate"] is not None:
        gates.append(("hallucination_rate <= %.2f" % args.max_hallucination,
                      agg["hallucination_rate"] <= args.max_hallucination, f"{agg['hallucination_rate']}"))
    ref = agg["refusal_correctness_unanswerable"]
    if ref is not None:
        gates.append(("refusal_correctness >= %.2f" % args.min_refusal, ref >= args.min_refusal, f"{ref}"))
    if args.max_p95_ms and agg["latency_ms"]["p95"]:
        gates.append(("p95_latency_ms <= %d" % args.max_p95_ms,
                      agg["latency_ms"]["p95"] <= args.max_p95_ms, f"{agg['latency_ms']['p95']}"))
    return gates


def print_report(agg: dict, gates: list):
    print("\n" + "=" * 64)
    print(f"DocVault Benchmark — {agg['n']} questions")
    print("=" * 64)
    print(f"\nHallucination rate: {agg['hallucination_rate']}   "
          f"Refusal(unanswerable): {agg['refusal_correctness_unanswerable']}")
    print(f"Latency p50/p95: {agg['latency_ms']['p50']} / {agg['latency_ms']['p95']} ms\n")
    print(f"{'metric':<22}{'mean':>7}{'  95% CI':>16}{'  n':>5}")
    for m, v in agg["overall"].items():
        print(f"{m:<22}{v['mean']:>7}{str(v['ci95']):>16}{v['n']:>5}")
    print("\nPer-category mean (faithfulness / correctness / ctx_recall):")
    for cat, v in agg["by_category"].items():
        print(f"  {cat:<16} n={v['n']:>3}  "
              f"faith={v.get('faithfulness','-')}  corr={v.get('answer_correctness','-')}  "
              f"recall={v.get('context_recall','-')}")
    print("\nGates:")
    for name, ok, val in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  (got {val})")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description="Run the DocVault LLM-as-judge benchmark.")
    ap.add_argument("--questions", default=str(Path(__file__).parent / "data" / "questions.jsonl"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "report.json"))
    ap.add_argument("--results", default=str(Path(__file__).parent / "data" / "results.jsonl"))
    ap.add_argument("--judge-model", default="gpt-4o", help="cross-family judge (≠ generator)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    ap.add_argument("--resume", action="store_true", help="skip questions already in --results")
    # gates
    ap.add_argument("--min-faithfulness", type=float, default=0.85)
    ap.add_argument("--max-hallucination", type=float, default=0.05)
    ap.add_argument("--min-refusal", type=float, default=0.80)
    ap.add_argument("--max-p95-ms", type=int, default=0, help="0 = no latency gate")
    args = ap.parse_args()

    settings.ensure_dirs()
    qpath = Path(args.questions)
    if not qpath.exists():
        raise SystemExit(f"No question set at {qpath}. Run generate_dataset.py first.")
    questions = [json.loads(l) for l in qpath.read_text().splitlines() if l.strip()]
    if args.limit:
        questions = questions[: args.limit]

    results_path = Path(args.results)
    done_ids = set()
    if args.resume and results_path.exists():
        existing = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
        done_ids = {r["id"] for r in existing}
        results = existing
        print(f"Resuming: {len(done_ids)} already evaluated.")
    else:
        results = []
        results_path.write_text("")

    pipe = DocVaultPipeline()
    with results_path.open("a", encoding="utf-8") as rf:
        for i, q in enumerate(questions, 1):
            if q["id"] in done_ids:
                continue
            print(f"[{i}/{len(questions)}] {q['category']:<14} {q['question'][:60]}")
            try:
                r = evaluate_one(q, pipe, args.judge_model)
            except Exception as e:
                # Transient API/TLS errors shouldn't kill the run; --resume retries this q.
                print(f"    SKIP {q['id']}: {type(e).__name__}: {str(e)[:80]}")
                continue
            results.append(r)
            rf.write(json.dumps(r, ensure_ascii=False) + "\n")
            rf.flush()

    agg = aggregate(results)
    gates = check_gates(agg, args)
    agg["gates"] = [{"gate": n, "pass": ok, "value": v} for n, ok, v in gates]
    Path(args.out).write_text(json.dumps(agg, indent=2))
    print_report(agg, gates)
    print(f"\nReport: {args.out}\nPer-question: {args.results}")


if __name__ == "__main__":
    main()
