"""Run the DocVault benchmark: query the live pipeline, judge with an LLM, and
report metrics with bootstrap confidence intervals and pass/fail gates.

Does NOT use exact-match. Retrieval + generation + behaviour are all scored by
an LLM judge (docvault.ragops.judges). Results are written incrementally so --resume can
continue an interrupted run.

Usage:
    python benchmark/run_benchmark.py                       # full run
    python benchmark/run_benchmark.py --limit 20            # smoke test
    python benchmark/run_benchmark.py --judge-model gpt-4o  # pick judge
"""

import argparse
import json
import warnings
from pathlib import Path

# litellm serializes responses through pydantic, emitting harmless "Unexpected
# field" UserWarnings — silence them so run output stays readable.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")

from docvault.config import settings
from docvault.pipeline import DocVaultPipeline
# Eval core lives in the package so the Celery worker can reuse it too.
from docvault.ragops import judges
from docvault.ragops.judge_eval import evaluate_one, aggregate


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
