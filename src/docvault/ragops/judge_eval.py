"""LLM-as-judge evaluation core — shared by the benchmark CLI and Celery tasks.

Scores a question set against the live pipeline with semantic judges (no exact
matching). Lives in the package so worker.py can import it; benchmark/run_benchmark.py
is a thin CLI wrapper over the same functions.
"""

import json
import random
import statistics
from pathlib import Path

import numpy as np

from docvault.config import settings
from docvault.ragops import judges

GROUNDED = {"factoid", "multi_hop", "comparative", "paraphrase", "table_lookup"}

METRICS = [
    "context_precision", "context_recall", "faithfulness", "answer_relevancy",
    "answer_correctness", "citation_accuracy", "refusal_correctness",
]


def evaluate_one(q: dict, pipe, judge_model: str) -> dict:
    """Run the pipeline on one question and score it with the judges."""
    import time

    t0 = time.time()
    res = pipe.query(q["question"])
    latency_ms = (time.time() - t0) * 1000

    answer = res.get("answer", "")
    contexts = res.get("contexts", [])
    answerable = q.get("answerable", True)

    raw: dict[str, float | None] = {
        "refusal_correctness": judges.judge_refusal_correctness(answer, answerable)["score"],
    }
    refused = judges.is_refusal(answer)

    if answerable:
        raw["context_precision"] = judges.judge_context_precision(q["question"], contexts, judge_model)["score"]
        raw["context_recall"] = judges.judge_context_recall(q["reference_answer"], contexts, judge_model)["score"]
        raw["faithfulness"] = judges.judge_faithfulness(answer, contexts, judge_model)["score"]
        raw["answer_relevancy"] = judges.judge_answer_relevancy(q["question"], answer, judge_model)["score"]
        raw["answer_correctness"] = judges.judge_answer_correctness(
            q["question"], answer, q["reference_answer"], judge_model)["score"]
        if not refused and not judges.has_citation(answer):
            raw["citation_accuracy"] = 0.0  # answerable answer that cites nothing = citation failure
        else:
            raw["citation_accuracy"] = judges.judge_citation_accuracy(answer, contexts, judge_model)["score"]
    elif not refused:
        raw["faithfulness"] = judges.judge_faithfulness(answer, contexts, judge_model)["score"]

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
    overall = {}
    for m in METRICS:
        vals = [r["scores"][m] for r in results if m in r["scores"]]
        if vals:
            overall[m] = {"mean": round(statistics.mean(vals), 3), "n": len(vals), "ci95": bootstrap_ci(vals)}

    faith = [r["scores"]["faithfulness"] for r in results if "faithfulness" in r["scores"]]
    hallucination_rate = round(1 - statistics.mean(faith), 3) if faith else None
    refusal_unanswerable = [r["scores"]["refusal_correctness"] for r in results if not r["answerable"]]

    by_cat: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in results}):
        sub = [r for r in results if r["category"] == cat]
        cat_metrics = {m: round(statistics.mean([r["scores"][m] for r in sub if m in r["scores"]]), 3)
                       for m in METRICS if any(m in r["scores"] for r in sub)}
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


def load_questions(path: Path | None = None) -> list[dict]:
    p = path or settings.benchmark_questions_path
    if not Path(p).exists():
        return []
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def run_judge_eval(
    pipe=None,
    questions_path: Path | None = None,
    judge_model: str | None = None,
    sample: int | None = None,
    seed: int = 13,
) -> dict:
    """Evaluate (a sample of) the question set against the pipeline.

    Returns the aggregate dict, plus `status`: "ok" | "no_questions". Used by the
    scheduled Celery eval tasks; `sample` caps cost.
    """
    questions = load_questions(questions_path)
    if not questions:
        return {"status": "no_questions", "n": 0}

    if sample and sample < len(questions):
        rng = random.Random(seed)
        questions = rng.sample(questions, sample)

    if pipe is None:
        from docvault.pipeline import DocVaultPipeline
        pipe = DocVaultPipeline()

    jm = judge_model or settings.judge_model
    results = []
    for q in questions:
        try:
            results.append(evaluate_one(q, pipe, jm))
        except Exception:
            continue  # transient API/TLS error on one question shouldn't abort the run

    agg = aggregate(results)
    agg["status"] = "ok"
    return agg
