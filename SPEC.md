# DocVault — Specification

## What It Does

Employees ask questions about company policies. DocVault answers from the actual documents, cites its sources, and says "I don't know" when it doesn't.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INGEST PIPELINE                       │
│                                                              │
│  Documents → Parse → Chunk → Embed → Index → Version Store  │
│     (PDF/MD/HTML)  (hierarchical)  (async)   (hybrid)        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                       QUERY PIPELINE                         │
│                                                              │
│  Query → Expand → Hybrid Retrieve → Rerank → Filter         │
│            │         (BM25+Dense)    (CE)    (confidence)    │
│            │                                                 │
│            ▼                                                 │
│  Context Assembly → Generate → Verify → Cite → Respond      │
│  (parent chunks)    (LLM)     (NLI)    (links)              │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                        RAGOPS LAYER                          │
│                                                              │
│  Eval Suite │ Drift Detection │ Traces │ Metrics │ Alerts    │
└─────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Ingest Pipeline

**Document Parser**
- Input: PDF, Markdown, HTML, DOCX
- Extracts: raw text + structural metadata (headers, sections, lists, tables)
- Library: `docling` for PDFs, `markitdown` for others
- Output: structured document with section hierarchy preserved

**Chunker**
- Strategy: hierarchical — respect document structure
- Primary chunks: individual sections/clauses (~300-500 tokens)
- Each chunk stores: `parent_chunk_id`, `document_id`, `section_path`, `chunk_index`
- Overlap: 10% sliding window within sections only
- Metadata: `doc_title`, `doc_version`, `effective_date`, `section_path`, `chunk_hash`

**Embedder**
- Model: `BAAI/bge-m3` (local, 1024-dim)
- Batch processing with configurable batch size
- Cache: embeddings stored by `chunk_hash` — skip re-embedding unchanged chunks

**Indexer**
- Dense: LanceDB (embedded, ANN via IVF-PQ)
- Sparse: SQLite FTS5 with BM25
- Both updated atomically per document

**Version Store**
- SQLite: `documents(id, title, source_path, version, effective_date, supersedes_id, status, ingested_at)`
- New version → old `status='superseded'`, linked via `supersedes_id`
- Retrieval filters to `status='active'` by default

### 2. Query Pipeline

**Hybrid Retrieval**
- Dense: top-50 from LanceDB ANN
- Sparse: top-50 from FTS5 BM25
- Fusion: Reciprocal Rank Fusion (RRF), α=0.6
- Output: top-20 candidates

**Reranker**
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Input: top-20, Output: top-5

**Context Assembly**
- Fetch parent chunks for top-5
- Deduplicate, order by document > section > chunk_index
- Budget: 3000 tokens max

**Confidence Gate**
- Top reranker score < 0.3 → low confidence
- No active document chunks → no answer

### 3. Generation

**Constrained prompt**: answer only from context, cite every claim, say "I don't know" when unsure.

**NLI Verification**: `nli-deberta-v3-small` checks each claim against source chunks. Contradictions stripped.

**Citations**: inline with (doc_title, section_path, version).

### 4. Memory

- Session memory: last 5 turns, 30 min TTL
- Retrieval cache: LRU keyed on query embedding, 1 hour TTL, invalidated on re-ingest

### 5. RAGOps

**Evals**: retrieval recall@10, faithfulness, relevancy, citation accuracy, hallucination rate
**Drift**: embedding distribution, query clusters, retrieval quality trends
**Traces**: full per-query trace (retrieval scores, latency, NLI scores)
**Metrics**: Prometheus — latency, cache hits, confidence, hallucination rate
**Alerts**: hallucination >5%, P95 latency >500ms, faithfulness <0.85

### 6. API

```
POST /api/query     — ask a question
POST /api/ingest    — ingest documents
GET  /api/health    — health check
GET  /api/metrics   — prometheus metrics
POST /api/eval/run  — trigger eval suite
GET  /api/eval/results
```

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Framework | FastAPI |
| Dense Index | LanceDB |
| Sparse Index | SQLite FTS5 |
| Embeddings | BGE-M3 (local) |
| Reranker | ms-marco-MiniLM cross-encoder |
| NLI Verifier | nli-deberta-v3-small |
| LLM | LiteLLM → GPT-4o-mini |
| Metrics | Prometheus + Grafana |
| Tracing | JSONL → SQLite |
| Deployment | Docker Compose |

---

## Build Phases

| # | Phase | Verify |
|---|-------|--------|
| 1 | Ingest: parser + chunker + embedder | Chunk a PDF, inspect output |
| 2 | Retrieval: BM25 + dense + fusion | Query returns relevant chunks |
| 3 | Reranker | Top-5 quality improves |
| 4 | Generation + citations | Answers cite sources |
| 5 | NLI verifier | Hallucinated claims stripped |
| 6 | Confidence gating | Low-confidence → "I don't know" |
| 7 | Conversation memory | Follow-ups work |
| 8 | API + Docker | docker compose up works |
| 9 | Eval suite + golden dataset | Evals run, metrics reported |
| 10 | Observability: traces + metrics | Grafana shows pipeline health |
| 11 | Drift detection + alerts | Degradation detected |
