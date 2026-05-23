"""Sparse retrieval via SQLite FTS5 BM25 — delegates to DocumentStore."""

from docvault.storage.documents import DocumentStore
from docvault.config import settings


def search(query: str, store: DocumentStore, top_k: int | None = None) -> list[dict]:
    """BM25 search. Returns list of {chunk_id, score, content, section_path, document_id}."""
    k = top_k or settings.sparse_top_k
    results = store.bm25_search(query, top_k=k)

    return [
        {
            "chunk_id": r["id"],
            "score": -r["score"],  # FTS5 bm25() returns negative scores (lower = better)
            "content": r["content"],
            "section_path": r.get("section_path", ""),
            "document_id": r["document_id"],
        }
        for r in results
    ]
