"""Tests for hierarchical chunking."""

import pytest
from pathlib import Path

from docvault.ingest.parser import parse_file
from docvault.ingest.chunker import chunk_document, count_tokens


SAMPLE_DIR = Path(__file__).parent / "fixtures" / "sample_docs"


def test_parse_and_chunk_markdown():
    """Parsing markdown produces sections; chunking produces chunks."""
    doc = parse_file(SAMPLE_DIR / "employee-handbook.md")
    assert doc.title == "Employee Handbook"
    assert len(doc.sections) > 5
    assert doc.format == "markdown"

    chunks = chunk_document(doc, "test-doc-id")
    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk["document_id"] == "test-doc-id"
        assert chunk["content"]
        assert chunk["chunk_hash"]
        assert chunk["chunk_index"] >= 0


def test_chunk_sizes_within_budget():
    """Leaf (retrievable) chunks stay within budget; parents may be larger by design.

    Atomic table blocks are exempt — a table is never split mid-row.
    """
    doc = parse_file(SAMPLE_DIR / "employee-handbook.md")
    chunks = chunk_document(doc, "test-doc-id")

    for chunk in chunks:
        if _role(chunk) == "parent":
            continue
        if "|" in chunk["content"]:  # atomic table block
            continue
        tokens = count_tokens(chunk["content"])
        # 400 budget + breadcrumb + tokenization slack
        assert tokens <= 500, f"Leaf too large: {tokens} tokens in {chunk['section_path']}"


def _role(chunk):
    import json
    return json.loads(chunk["metadata"]).get("role", "leaf")


def test_parent_chunks_built_and_linked():
    """Multi-leaf sections produce a parent chunk that children point to."""
    doc = parse_file(SAMPLE_DIR / "employee-handbook.md")
    chunks = chunk_document(doc, "test-doc-id")

    parents = [c for c in chunks if _role(c) == "parent"]
    linked = [c for c in chunks if c.get("_parent_index") is not None]
    assert parents, "expected at least one parent chunk"
    assert linked, "expected at least one child linked to a parent"
    for c in linked:
        assert _role(chunks[c["_parent_index"]]) == "parent"


def test_section_paths_populated():
    """Chunks should have section paths reflecting document hierarchy."""
    doc = parse_file(SAMPLE_DIR / "employee-handbook.md")
    chunks = chunk_document(doc, "test-doc-id")

    paths = [c["section_path"] for c in chunks]
    assert any("Benefits" in p for p in paths)
    assert any("Remote Work" in p for p in paths)


def test_chunk_hashes_unique_for_different_content():
    """Different content should produce different hashes."""
    doc = parse_file(SAMPLE_DIR / "employee-handbook.md")
    chunks = chunk_document(doc, "test-doc-id")

    hashes = [c["chunk_hash"] for c in chunks]
    assert len(set(hashes)) == len(hashes), "Duplicate hashes found"


def test_empty_section_produces_no_chunks():
    """Sections with no content should not produce chunks."""
    from docvault.ingest.parser import Section, ParsedDocument

    doc = ParsedDocument(
        title="Test",
        source_path="test.md",
        sections=[Section(title="Empty", content="", level=1, path="Empty")],
        raw_text="",
        format="markdown",
    )
    chunks = chunk_document(doc, "test-doc-id")
    assert len(chunks) == 0
