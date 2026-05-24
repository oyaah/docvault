"""Tests for claim extraction and NLI adversarial scenarios."""

from docvault.generation.verifier import _extract_claims


def test_extract_claims_basic():
    answer = (
        "Employees get 15 days of PTO in their first year. "
        "After 3 years, this increases to 20 days. "
        "[Source: Employee Handbook, Benefits > PTO]"
    )
    claims = _extract_claims(answer)
    assert len(claims) == 2
    assert "15 days" in claims[0]
    assert "20 days" in claims[1]


def test_extract_claims_skips_no_answer():
    answer = "I couldn't find this in our policies. Please check with HR."
    claims = _extract_claims(answer)
    assert len(claims) == 0


def test_extract_claims_removes_citations():
    answer = "The 401k match is 4% [Source: HR Policy, Benefits]. This is great for employees."
    claims = _extract_claims(answer)
    for claim in claims:
        assert "[Source:" not in claim


# ── NLI adversarial tests (integration-level, use real model) ──

def test_nli_verifier_catches_contradiction():
    """Verifier should strip claims that contradict the source context."""
    from docvault.generation.verifier import verify_answer

    context = [{"content": "Employees receive 15 days of PTO per year during their first two years of employment."}]
    # This answer contains a hallucinated claim (30 days instead of 15)
    hallucinated_answer = "Employees receive 30 days of PTO per year. This is one of the best PTO policies in the industry."

    result = verify_answer(hallucinated_answer, context, threshold=0.5)
    # Should detect at least one problematic claim
    assert result.claims_total > 0
    # The verified answer should differ from original (something was flagged)
    has_flagged = any(c["label"] != "entailment" for c in result.claims)
    assert has_flagged, f"NLI should flag hallucinated claims. Got: {result.claims}"


def test_nli_verifier_passes_grounded_claims():
    """Verifier should NOT strip claims that are grounded in context."""
    from docvault.generation.verifier import verify_answer

    context = [{"content": "The company matches 401k contributions up to 4% of salary. Full vesting after 4 years."}]
    grounded_answer = "The company matches 401k contributions up to 4% of salary."

    result = verify_answer(grounded_answer, context, threshold=0.7)
    assert result.claims_stripped == 0, f"Grounded claims should not be stripped. Got: {result.claims}"


def test_nli_verifier_handles_unrelated_context():
    """When context is completely unrelated, claims should be flagged."""
    from docvault.generation.verifier import verify_answer

    context = [{"content": "The cafeteria serves lunch from 11:30am to 1:30pm on weekdays."}]
    unrelated_answer = "Employees are entitled to 16 weeks of parental leave upon the birth or adoption of a child."

    result = verify_answer(unrelated_answer, context, threshold=0.5)
    # Should flag this as not supported by context
    has_flagged = any(c["label"] != "entailment" for c in result.claims)
    assert has_flagged, f"Unrelated claims should be flagged. Got: {result.claims}"
