"""Tests for storage, BM25 retrieval, and fusion logic."""

import pytest
import tempfile
from pathlib import Path

from docvault.storage.documents import DocumentStore
from docvault.retrieval.fusion import reciprocal_rank_fusion


@pytest.fixture
def store(tmp_path):
    return DocumentStore(db_path=tmp_path / "test.db")


def test_document_crud(store):
    """Add document, verify retrieval."""
    doc = store.add_document(title="Test Policy", source_path="/tmp/test.md", version="1.0")
    assert doc.id
    assert doc.status == "active"

    fetched = store.get_document(doc.id)
    assert fetched.title == "Test Policy"


def test_document_versioning(store):
    """New version supersedes old one."""
    doc1 = store.add_document(title="Policy A", source_path="/tmp/a.md", version="1.0")
    doc2 = store.add_document(title="Policy A", source_path="/tmp/a.md", version="2.0")

    assert doc2.supersedes_id == doc1.id

    old = store.get_document(doc1.id)
    assert old.status == "superseded"

    active = store.get_active_documents()
    assert len(active) == 1
    assert active[0].version == "2.0"


def test_chunk_storage_and_bm25(store):
    """Store chunks, verify BM25 search works."""
    doc = store.add_document(title="HR Policy", source_path="/tmp/hr.md")

    chunks = [
        {
            "document_id": doc.id,
            "content": "Employees receive 15 days of paid time off per year during their first two years.",
            "section_path": "Benefits > PTO",
            "chunk_index": 0,
            "chunk_hash": "abc123",
            "token_count": 20,
        },
        {
            "document_id": doc.id,
            "content": "The company matches 401k contributions up to 4% of salary.",
            "section_path": "Benefits > 401k",
            "chunk_index": 1,
            "chunk_hash": "def456",
            "token_count": 15,
        },
    ]
    store.add_chunks(chunks)

    # BM25 search
    results = store.bm25_search("paid time off", top_k=5)
    assert len(results) > 0
    assert "paid time off" in results[0]["content"].lower()


def test_reciprocal_rank_fusion():
    """RRF merges dense and sparse correctly."""
    dense = [
        {"chunk_id": "a", "score": 0.9},
        {"chunk_id": "b", "score": 0.8},
        {"chunk_id": "c", "score": 0.7},
    ]
    sparse = [
        {"chunk_id": "b", "score": 0.95},
        {"chunk_id": "d", "score": 0.85},
        {"chunk_id": "a", "score": 0.75},
    ]

    fused = reciprocal_rank_fusion(dense, sparse, top_k=3)
    assert len(fused) == 3

    # "a" and "b" appear in both — should rank higher
    fused_ids = [r["chunk_id"] for r in fused]
    assert "a" in fused_ids[:2] or "b" in fused_ids[:2]

    # All have RRF scores
    for r in fused:
        assert "rrf_score" in r
        assert r["rrf_score"] > 0


def test_stats(store):
    """Stats reflect document and chunk counts."""
    stats = store.stats()
    assert stats["active_documents"] == 0
    assert stats["active_chunks"] == 0

    doc = store.add_document(title="Test", source_path="/tmp/t.md")
    store.add_chunks([{
        "document_id": doc.id,
        "content": "Test content",
        "section_path": "Test",
        "chunk_index": 0,
        "chunk_hash": "test123",
        "token_count": 5,
    }])

    stats = store.stats()
    assert stats["active_documents"] == 1
    assert stats["active_chunks"] == 1
