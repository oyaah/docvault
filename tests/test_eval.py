"""Tests for the evaluation suite."""

from docvault.ragops.evaluator import compute_retrieval_recall, run_eval_suite


def test_retrieval_recall_full_match():
    expected = ["Benefits > PTO", "Benefits > 401k"]
    retrieved = ["Benefits > PTO > Accrual", "Benefits > 401k Match", "Other Section"]
    recall = compute_retrieval_recall(expected, retrieved)
    assert recall == 1.0


def test_retrieval_recall_partial():
    expected = ["Benefits > PTO", "Benefits > 401k"]
    retrieved = ["Benefits > PTO > Accrual", "Security > VPN"]
    recall = compute_retrieval_recall(expected, retrieved)
    assert recall == 0.5


def test_retrieval_recall_none():
    expected = ["Benefits > PTO"]
    retrieved = ["Security > VPN", "Other"]
    recall = compute_retrieval_recall(expected, retrieved)
    assert recall == 0.0


def test_retrieval_recall_empty_expected():
    """If no expected sections, recall should be 1.0 (vacuously true)."""
    recall = compute_retrieval_recall([], ["anything"])
    assert recall == 1.0


def test_eval_suite_no_dataset(tmp_path):
    """Nonexistent golden dataset should return empty result."""
    result = run_eval_suite(
        lambda q: {"answer": "", "latency_ms": 0},
        golden_path=tmp_path / "nonexistent.json",
    )
    assert result.total_queries == 0
