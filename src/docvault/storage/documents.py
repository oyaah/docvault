"""Document version store — SQLite-backed metadata and versioning."""

import sqlite3
import uuid
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from docvault.config import settings


@dataclass
class Document:
    id: str
    title: str
    source_path: str
    version: str
    effective_date: str | None
    supersedes_id: str | None
    status: str  # 'active' | 'superseded'
    ingested_at: str
    doc_type: str | None = None
    total_chunks: int = 0


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_path TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '1.0',
    effective_date TEXT,
    supersedes_id TEXT REFERENCES documents(id),
    status TEXT NOT NULL DEFAULT 'active',
    doc_type TEXT,
    total_chunks INTEGER DEFAULT 0,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    parent_chunk_id TEXT REFERENCES chunks(id),
    content TEXT NOT NULL,
    section_path TEXT,
    chunk_index INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL,
    token_count INTEGER,
    metadata TEXT,  -- JSON
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(chunk_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- FTS5 for BM25 sparse retrieval
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    section_path,
    content=chunks,
    content_rowid=rowid,
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content, section_path)
    VALUES (new.rowid, new.content, new.section_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, section_path)
    VALUES ('delete', old.rowid, old.content, old.section_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, section_path)
    VALUES ('delete', old.rowid, old.content, old.section_path);
    INSERT INTO chunks_fts(rowid, content, section_path)
    VALUES (new.rowid, new.content, new.section_path);
END;
"""


class DocumentStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def add_document(
        self,
        title: str,
        source_path: str,
        version: str = "1.0",
        effective_date: str | None = None,
        doc_type: str | None = None,
    ) -> Document:
        doc_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        with self._conn() as conn:
            # Supersede previous versions of same title
            prev = conn.execute(
                "SELECT id FROM documents WHERE title = ? AND status = 'active'",
                (title,),
            ).fetchone()

            supersedes_id = None
            if prev:
                supersedes_id = prev["id"]
                conn.execute(
                    "UPDATE documents SET status = 'superseded' WHERE id = ?",
                    (supersedes_id,),
                )

            conn.execute(
                """INSERT INTO documents (id, title, source_path, version, effective_date,
                   supersedes_id, status, doc_type, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
                (doc_id, title, source_path, version, effective_date, supersedes_id, doc_type, now),
            )

        return Document(
            id=doc_id,
            title=title,
            source_path=source_path,
            version=version,
            effective_date=effective_date,
            supersedes_id=supersedes_id,
            status="active",
            doc_type=doc_type,
            ingested_at=now,
        )

    def add_chunks(self, chunks: list[dict]):
        """Insert chunks in batch with parent linking.

        Resolves _parent_index references to actual DB chunk IDs after insertion.
        """
        now = datetime.now(datetime.UTC).isoformat() if hasattr(datetime, 'UTC') else datetime.utcnow().isoformat()
        chunk_ids = []

        with self._conn() as conn:
            # First pass: insert all chunks without parent links
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                chunk_ids.append(chunk_id)
                conn.execute(
                    """INSERT INTO chunks (id, document_id, parent_chunk_id, content,
                       section_path, chunk_index, chunk_hash, token_count, metadata, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chunk_id,
                        chunk["document_id"],
                        None,  # parent set in second pass
                        chunk["content"],
                        chunk.get("section_path", ""),
                        chunk["chunk_index"],
                        chunk["chunk_hash"],
                        chunk.get("token_count", 0),
                        chunk.get("metadata", "{}"),
                        now,
                    ),
                )

            # Second pass: resolve parent links using _parent_index
            for i, chunk in enumerate(chunks):
                parent_idx = chunk.get("_parent_index")
                if parent_idx is not None and parent_idx < len(chunk_ids):
                    parent_id = chunk_ids[parent_idx]
                    conn.execute(
                        "UPDATE chunks SET parent_chunk_id = ? WHERE id = ?",
                        (parent_id, chunk_ids[i]),
                    )

            # Update document chunk count
            if chunks:
                doc_id = chunks[0]["document_id"]
                conn.execute(
                    "UPDATE documents SET total_chunks = ? WHERE id = ?",
                    (len(chunks), doc_id),
                )

        return chunk_ids

    def get_chunk_by_id(self, chunk_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
            return dict(row) if row else None

    def get_chunks_by_document(self, document_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (document_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_parent_chunk(self, chunk_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT p.* FROM chunks p
                   JOIN chunks c ON c.parent_chunk_id = p.id
                   WHERE c.id = ?""",
                (chunk_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_active_documents(self) -> list[Document]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE status = 'active' ORDER BY ingested_at DESC"
            ).fetchall()
            return [Document(**dict(r)) for r in rows]

    def get_document(self, doc_id: str) -> Document | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            return Document(**dict(row)) if row else None

    def get_chunk_hashes(self, document_id: str) -> set[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_hash FROM chunks WHERE document_id = ?", (document_id,)
            ).fetchall()
            return {r["chunk_hash"] for r in rows}

    def delete_document_chunks(self, document_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

    def bm25_search(self, query: str, top_k: int = 50) -> list[dict]:
        """BM25 search via FTS5. Returns chunks with bm25 rank scores."""
        import re
        tokens = re.findall(r'\w+', query)
        if not tokens:
            return []
        # Use OR so chunks matching any token are returned (BM25 ranks by relevance)
        fts_query = " OR ".join(f'"{t}"' for t in tokens)
        with self._conn() as conn:
            # Parent (context) chunks are excluded — only leaf chunks are retrievable.
            rows = conn.execute(
                """SELECT c.*, bm25(chunks_fts) AS score
                   FROM chunks_fts
                   JOIN chunks c ON chunks_fts.rowid = c.rowid
                   JOIN documents d ON c.document_id = d.id
                   WHERE chunks_fts MATCH ? AND d.status = 'active'
                     AND COALESCE(json_extract(c.metadata, '$.role'), 'leaf') != 'parent'
                   ORDER BY score
                   LIMIT ?""",
                (fts_query, top_k),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_active_leaf_chunks(self, limit: int | None = None) -> list[dict]:
        """Return retrievable (non-parent) chunks from active documents.

        Used by the benchmark dataset generator to ground synthetic questions.
        """
        sql = (
            """SELECT c.*, d.title AS doc_title, d.version AS doc_version
               FROM chunks c JOIN documents d ON c.document_id = d.id
               WHERE d.status = 'active'
                 AND COALESCE(json_extract(c.metadata, '$.role'), 'leaf') != 'parent'
               ORDER BY c.document_id, c.chunk_index"""
        )
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql).fetchall()]

    def stats(self) -> dict:
        with self._conn() as conn:
            docs = conn.execute("SELECT COUNT(*) as n FROM documents WHERE status='active'").fetchone()
            chunks = conn.execute(
                """SELECT COUNT(*) as n FROM chunks c
                   JOIN documents d ON c.document_id = d.id
                   WHERE d.status = 'active'"""
            ).fetchone()
            return {"active_documents": docs["n"], "active_chunks": chunks["n"]}
