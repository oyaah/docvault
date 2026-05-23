"""Tests for claim extraction (NLI model tests are integration-level)."""

from docvault.generation.verifier import _extract_claims


def test_extract_claims_basic():
    """Extracts sentences as claims, skips short and meta sentences."""
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
    """Should skip 'I couldn't find...' meta-responses."""
    answer = "I couldn't find this in our policies. Please check with HR."
    claims = _extract_claims(answer)
    assert len(claims) == 0


def test_extract_claims_removes_citations():
    """Citation markers should be stripped before claim extraction."""
    answer = "The 401k match is 4% [Source: HR Policy, Benefits]. This is great for employees."
    claims = _extract_claims(answer)
    for claim in claims:
        assert "[Source:" not in claim
