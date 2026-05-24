"""Migration 001 — Initial schema.

This is the baseline migration. The schema is created by DocumentStore.__init__
on first run. This file exists to document the schema and enable future migrations.
"""

VERSION = 1
DESCRIPTION = "Initial schema: documents, chunks, chunks_fts"

UP = """
-- Documents table
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

-- Chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    parent_chunk_id TEXT REFERENCES chunks(id),
    content TEXT NOT NULL,
    section_path TEXT,
    chunk_index INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL,
    token_count INTEGER,
    metadata TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(chunk_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- FTS5 for BM25
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content, section_path,
    content=chunks, content_rowid=rowid,
    tokenize='porter unicode61'
);

-- FTS sync triggers
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

-- Migration tracking
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    description TEXT,
    applied_at TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_migrations (version, description, applied_at)
VALUES (1, 'Initial schema', datetime('now'));
"""

DOWN = """
DROP TABLE IF EXISTS schema_migrations;
DROP TRIGGER IF EXISTS chunks_au;
DROP TRIGGER IF EXISTS chunks_ad;
DROP TRIGGER IF EXISTS chunks_ai;
DROP TABLE IF EXISTS chunks_fts;
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS documents;
"""
