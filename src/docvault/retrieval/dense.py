"""Dense retrieval via LanceDB ANN search."""

import lancedb
import pyarrow as pa
import numpy as np

from docvault.config import settings


_db = None
_table = None

TABLE_NAME = "chunks"


def _get_db():
    global _db
    if _db is None:
        settings.ensure_dirs()
        _db = lancedb.connect(str(settings.lance_path))
    return _db


def _get_table():
    global _table
    if _table is None:
        db = _get_db()
        try:
            _table = db.open_table(TABLE_NAME)
        except Exception:
            _table = None
    return _table


def index_chunks(chunk_ids: list[str], embeddings: np.ndarray, metadata: list[dict] | None = None):
    """Add chunks to dense index."""
    global _table

    db = _get_db()
    data = []
    for i, chunk_id in enumerate(chunk_ids):
        row = {
            "chunk_id": chunk_id,
            "vector": embeddings[i].tolist(),
        }
        if metadata:
            row["document_id"] = metadata[i].get("document_id", "")
            row["section_path"] = metadata[i].get("section_path", "")
        data.append(row)

    try:
        table = db.open_table(TABLE_NAME)
        table.add(data)
        _table = table
    except Exception:
        _table = db.create_table(TABLE_NAME, data)


def search(query_embedding: np.ndarray, top_k: int | None = None) -> list[dict]:
    """ANN search. Returns list of {chunk_id, score, document_id, section_path}."""
    table = _get_table()
    if table is None:
        return []

    k = top_k or settings.dense_top_k
    results = table.search(query_embedding.tolist()).limit(k).to_list()

    return [
        {
            "chunk_id": r["chunk_id"],
            "score": 1.0 - r.get("_distance", 0.0),  # cosine similarity
            "document_id": r.get("document_id", ""),
            "section_path": r.get("section_path", ""),
        }
        for r in results
    ]


def delete_by_document(document_id: str):
    """Remove all vectors for a document."""
    table = _get_table()
    if table is not None:
        try:
            table.delete(f"document_id = '{document_id}'")
        except Exception:
            pass


def reset():
    """Drop and recreate the table."""
    global _table
    db = _get_db()
    try:
        db.drop_table(TABLE_NAME)
    except Exception:
        pass
    _table = None
