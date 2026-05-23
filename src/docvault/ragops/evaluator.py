"""Evaluation suite — golden dataset evals for retrieval and generation quality."""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field

from docvault.config import settings


@dataclass
class EvalResult:
    query: str
    expected_answer: str | None
    expected_chunks: list[str]  # expected chunk section_paths
    actual_answer: str
    retrieved_chunks: list[str]  # actual section_paths retrieved
    retrieval_recall: float  # % of expected chunks found
    answer_contains_expected: bool
    latency_ms: float
    timestamp: str = ""


@dataclass
class EvalSuiteResult:
    total_queries: int = 0
    avg_retrieval_recall: float = 0.0
    avg_latency_ms: float = 0.0
    answer_accuracy: float = 0.0  # % of queries where answer contains expected info
    results: list[EvalResult] = field(default_factory=list)
    timestamp: str = ""


def load_golden_dataset(path: Path | None = None) -> list[dict]:
    """Load golden Q&A pairs from JSON file.

    Format: [{"question": "...", "expected_answer": "...", "expected_sections": ["..."]}]
    """
    golden_path = path or Path("eval/golden_dataset.json")
    if not golden_path.exists():
        return []
    return json.loads(golden_path.read_text())


def compute_retrieval_recall(expected_sections: list[str], retrieved_sections: list[str]) -> float:
    """What fraction of expected sections were retrieved?"""
    if not expected_sections:
        return 1.0
    found = sum(1 for es in expected_sections if any(es.lower() in rs.lower() for rs in retrieved_sections))
    return found / len(expected_sections)


def run_eval_suite(query_fn, golden_path: Path | None = None) -> EvalSuiteResult:
    """Run full eval suite against golden dataset.

    Args:
        query_fn: callable(question: str) -> dict with keys:
                  answer, retrieved_sections, latency_ms
        golden_path: path to golden dataset JSON
    """
    dataset = load_golden_dataset(golden_path)
    if not dataset:
        return EvalSuiteResult(timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"))

    results = []
    for item in dataset:
        question = item["question"]
        expected = item.get("expected_answer", "")
        expected_sections = item.get("expected_sections", [])

        result = query_fn(question)

        recall = compute_retrieval_recall(
            expected_sections, result.get("retrieved_sections", [])
        )
        contains = expected.lower() in result["answer"].lower() if expected else True

        results.append(
            EvalResult(
                query=question,
                expected_answer=expected,
                expected_chunks=expected_sections,
                actual_answer=result["answer"],
                retrieved_chunks=result.get("retrieved_sections", []),
                retrieval_recall=recall,
                answer_contains_expected=contains,
                latency_ms=result.get("latency_ms", 0),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        )

    avg_recall = sum(r.retrieval_recall for r in results) / len(results) if results else 0
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0
    accuracy = sum(1 for r in results if r.answer_contains_expected) / len(results) if results else 0

    return EvalSuiteResult(
        total_queries=len(results),
        avg_retrieval_recall=round(avg_recall, 4),
        avg_latency_ms=round(avg_latency, 1),
        answer_accuracy=round(accuracy, 4),
        results=results,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def save_eval_results(suite_result: EvalSuiteResult, path: Path | None = None):
    """Save eval results to JSON."""
    out_path = path or (settings.data_dir / "eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "total_queries": suite_result.total_queries,
        "avg_retrieval_recall": suite_result.avg_retrieval_recall,
        "avg_latency_ms": suite_result.avg_latency_ms,
        "answer_accuracy": suite_result.answer_accuracy,
        "timestamp": suite_result.timestamp,
        "results": [
            {
                "query": r.query,
                "retrieval_recall": r.retrieval_recall,
                "answer_contains_expected": r.answer_contains_expected,
                "latency_ms": r.latency_ms,
            }
            for r in suite_result.results
        ],
    }
    out_path.write_text(json.dumps(data, indent=2))
