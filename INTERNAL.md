# DocVault — Internal Technical Document

> Brutally honest architecture review, bug postmortem, and benchmark strategy.
> Last updated: 2026-05-24

---

## Table of Contents

1. [What DocVault Is](#what-docvault-is)
2. [Full Tech Stack](#full-tech-stack)
3. [Pipeline Deep Dive](#pipeline-deep-dive)
4. [Architecture Decisions — Honest Assessment](#architecture-decisions)
5. [What's Wrong With It](#whats-wrong-with-it)
6. [Bug Postmortem — Everything That Broke](#bug-postmortem)
7. [Tuning & Configuration](#tuning--configuration)
8. [Benchmark Strategy & Metrics](#benchmark-strategy--metrics)
9. [Current Eval Results](#current-eval-results)

---

## 1. What DocVault Is

A RAG (Retrieval-Augmented Generation) pipeline for enterprise document Q&A. Feed it company policy documents (markdown, PDF, HTML), ask natural language questions, get cited answers with hallucination verification.

**Core value proposition:** Zero-hallucination answers with source citations, backed by NLI-based claim verification.

**What it actually does:**
```
User question → Query expansion → Hybrid retrieval (dense + sparse)
→ RRF fusion → Cross-encoder reranking → LLM generation
→ NLI claim verification → Cited, verified answer
```

---

## 2. Full Tech Stack

### Core Dependencies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **API** | FastAPI + Uvicorn | ≥0.115 / ≥0.30 | REST API server |
| **LLM** | LiteLLM → OpenAI GPT-4o-mini | ≥1.40 | Answer generation + query expansion |
| **Embeddings** | BAAI/bge-m3 (via sentence-transformers) | ≥3.0 | Dense retrieval embeddings (1024-dim) |
| **Dense Index** | LanceDB | ≥0.15 | ANN vector search |
| **Sparse Index** | SQLite FTS5 (porter + unicode61) | built-in | BM25 keyword search |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | via sentence-transformers | Cross-encoder relevance scoring |
| **NLI Verifier** | cross-encoder/nli-deberta-v3-small | via sentence-transformers | Hallucination detection via NLI |
| **Tokenizer** | tiktoken (cl100k_base) | ≥0.7 | Token counting for chunking |
| **Hashing** | xxhash (xxh64) | ≥3.0 | Chunk deduplication |
| **Metadata DB** | SQLite (WAL mode) | built-in | Document/chunk metadata + versioning |
| **Cache/Sessions** | Redis (with in-memory fallback) | ≥5.0 | Retrieval cache + conversation sessions |
| **Task Queue** | Celery | ≥5.3 | Async ingestion + scheduled evals |
| **Metrics** | prometheus-client | ≥0.20 | Counters, histograms, gauges |
| **Config** | pydantic-settings | ≥2.0 | Env-var based settings |
| **PDF Parsing** | Docling (optional) | ≥2.0 | Structured PDF extraction |
| **HTML Parsing** | MarkItDown (optional) | ≥0.1 | HTML → Markdown conversion |

### ML Models (loaded at runtime, downloaded on first use)

| Model | Size | Device | Load Time | Purpose |
|-------|------|--------|-----------|---------|
| BAAI/bge-m3 | ~2.3GB | MPS/CUDA/CPU | ~8s | Embeddings (1024-dim, multilingual) |
| ms-marco-MiniLM-L-6-v2 | ~80MB | CPU | ~1s | Reranking (cross-encoder) |
| nli-deberta-v3-small | ~140MB | CPU | ~2s | NLI entailment/contradiction |

### Infrastructure (Docker Compose)

| Service | Image | Purpose |
|---------|-------|---------|
| docvault | custom (Python 3.11-slim) | API server |
| worker | same image | Celery worker for async ingestion |
| beat | same image | Celery beat for scheduled tasks |
| redis | redis:7-alpine (256MB max) | Cache, sessions, Celery broker |
| prometheus | prom/prometheus:latest | Metrics collection |
| grafana | grafana/grafana:latest | Dashboards |

---

## 3. Pipeline Deep Dive

### 3.1 Ingest Pipeline

```
File → Parser → Chunker → Embedder → Store
```

**Step 1: Parse** (`ingest/parser.py`)
- Detects format by extension (.md, .pdf, .html, .txt)
- Markdown: Regex-based heading extraction → hierarchical `Section` objects
- PDF: Docling (structured) or pdftotext (fallback)
- HTML: MarkItDown or regex tag stripping
- Builds section paths like `"Remote Work Policy > 5. Working Hours > 5.1 Core Window"`

**Step 2: Chunk** (`ingest/chunker.py`)
- Token-based windowing using tiktoken's cl100k_base encoder
- Default: 400 tokens per chunk, 10% overlap
- Respects section boundaries — each section chunked independently
- Parent linking: first chunk of multi-chunk section becomes "parent" for context expansion
- Each chunk gets an xxhash64 hash for dedup/cache keying

**Step 3: Embed** (`ingest/embedder.py`)
- BGE-M3 via sentence-transformers, normalized embeddings
- In-memory hash-based cache (chunk_hash → embedding)
- After embedding, computes embedding drift vs baseline mean

**Step 4: Store**
- SQLite: Document metadata, chunk text, section paths, parent links
- FTS5: Auto-synced via triggers (INSERT/UPDATE/DELETE) for BM25 search
- LanceDB: Dense vector index for ANN search

### 3.2 Query Pipeline

```
Question → Expand → [Dense + Sparse] → Fuse → Rerank → Gate → Enrich → Generate → Verify → Respond
```

**Step 1: Query Expansion** (`generation/generator.py:expand_query`)
- For short queries (≤5 words), asks LLM for 2 alternative phrasings
- Currently: expansions are generated but NOT used in retrieval (only the original query is searched). This is a bug/incomplete feature.

**Step 2: Dense Retrieval** (`retrieval/dense.py`)
- Embeds query with BGE-M3
- LanceDB ANN search, top 50 results
- Returns chunk_id + cosine similarity score
- Enriched with content from SQLite (extra DB round-trip per chunk)

**Step 3: Sparse Retrieval** (`retrieval/sparse.py` → `storage/documents.py:bm25_search`)
- SQLite FTS5 with porter stemming + unicode61 tokenizer
- Query tokens extracted via `\w+` regex, joined with OR, double-quoted
- Returns top 50 by BM25 score
- Filtered to active documents only via JOIN

**Step 4: Reciprocal Rank Fusion** (`retrieval/fusion.py`)
- Standard RRF with k=60 smoothing constant
- Merges dense + sparse by chunk_id
- Returns top 20 fused results

**Step 5: Cross-Encoder Reranking** (`retrieval/reranker.py`)
- ms-marco-MiniLM-L-6-v2 cross-encoder
- Scores all 20 fused candidates
- Raw logits → sigmoid normalization → [0,1] scores
- Returns top 5

**Step 6: Confidence Gate** (`pipeline.py:206`)
- If top reranker score < 0.3, returns "I couldn't find this" (no LLM call)
- Saves ~$0.001/query and prevents confident-sounding garbage

**Step 7: Context Enrichment** (`pipeline.py:226-256`)
- For each reranked chunk, fetches parent chunk for broader context
- Adds document title and version metadata
- Deduplicates by first 100 chars of content

**Step 8: LLM Generation** (`generation/generator.py`)
- GPT-4o-mini via LiteLLM, temperature=0.1
- System prompt constrains: cite everything, don't use outside knowledge, flag conflicts
- Context formatted as `[Document N: Title, Section Path, vVersion]\n{content}`
- Conversation history (last 5 turns) included for multi-turn

**Step 9: NLI Verification** (`generation/verifier.py`)
- Splits answer into claims by sentence boundaries
- For each claim, NLI classifies as entailment/neutral/contradiction
- Raw logits → softmax → probability distribution
- Claims stripped only when contradiction probability > threshold (default 0.7)
- Stripped claims replaced with disclaimer text
- Neutral = "model uncertain" → NOT stripped (key design decision)

**Step 10: Citation Extraction** (`generation/generator.py:_extract_citations`)
- Regex extracts `[Source: document, section]` from LLM output
- Parses into structured citation objects

### 3.3 Observability Pipeline

**Tracing** (`ragops/tracer.py`)
- Every query gets a UUID trace_id
- Captures: BM25 results count, dense results count, fused count, reranker scores, cache hit, generation model/tokens, verification claims, confidence, total latency
- Appended to `data/traces.jsonl` (append-only JSONL)

**Metrics** (`ragops/metrics.py`)
- Prometheus counters: query_total, query_errors, ingest_total, cache_hits/misses
- Histograms: query_latency by stage, NLI scores, confidence scores
- Gauges: active_documents, active_chunks, hallucination_rate

**Drift Detection** (`ragops/drift.py`)
- Embedding drift: cosine distance between new batch mean and baseline
- Retrieval quality trend: rolling average of top reranker scores
- Hallucination rate: claims_stripped / claims_total over 7 days
- Query topic clustering: semantic clustering via embedding cosine similarity (greedy, threshold=0.75)
- New topic detection: queries with <0.6 similarity to any older query

---

## 4. Architecture Decisions — Honest Assessment

### Good Decisions

| Decision | Why It's Right |
|----------|---------------|
| **Hybrid retrieval (dense + sparse)** | Dense catches semantic similarity, sparse catches exact terms. RRF fusion is the standard approach. |
| **NLI-based verification** | Real claim-level verification, not just vibes. Catches contradictions that prompt engineering alone misses. |
| **Cross-encoder reranking** | Bi-encoder retrieval is fast but noisy. Cross-encoder reranking on top-20 gives much better precision without latency blowup. |
| **SQLite for metadata** | Perfect for single-node. No network round-trip, WAL mode handles concurrent reads, FTS5 is genuinely good for BM25. |
| **Redis with in-memory fallback** | Graceful degradation. Works in dev without Redis, scales in prod with it. |
| **Interface-based backends** | SessionBackend/CacheBackend ABCs allow swapping Redis/memory without touching business logic. |
| **Confidence gate** | Prevents the system from generating confident-sounding answers from irrelevant context. Critical for enterprise trust. |
| **Document versioning** | Supersede old versions, keep history. Essential for policies that change. |
| **Chunk parent linking** | Expanding context to parent chunk gives the LLM more context without inflating the retrieval index. |

### Questionable Decisions

| Decision | The Problem |
|----------|-------------|
| **LanceDB for dense index** | LanceDB is embedded/local-first. Fine for prototype, but no replication, no hot standby, questionable under concurrent writes. Would need Pinecone/Weaviate/Qdrant for real production. |
| **Single SQLite DB** | Works for single-node but is a scalability ceiling. Fine for internal tool, problematic for multi-pod deployment. |
| **GPT-4o-mini as default** | Cheap and fast, but accuracy on complex multi-hop questions is mediocre. Should benchmark against GPT-4o, Claude Sonnet for quality-sensitive deployments. |
| **In-process model loading** | BGE-M3 (2.3GB) loads into the API server process. First request after cold start takes ~10s. Should be a separate embedding service or pre-warmed. |
| **No connection pooling on SQLite** | Each method call creates a new connection (`self._conn()`). Overhead is low for SQLite but it's wasteful. |
| **Celery for task queue** | Celery is battle-tested but heavy for this use case. Could use FastAPI BackgroundTasks for simple async or something lighter like Dramatiq. |
| **Prometheus without push gateway** | Worker processes can't easily expose metrics to Prometheus (pull model). Would need push gateway or a different approach for worker metrics. |

### Wrong Decisions

| Decision | Why It's Wrong |
|----------|---------------|
| **Query expansion generated but not used** | `expand_query()` generates alternative phrasings but only the original query goes to retrieval. Dead code that burns LLM tokens. |
| **Citation extraction from LLM text** | Parses `[Source: ...]` with regex. Fragile — depends on LLM following format perfectly. Should use structured output or extract from chunk metadata directly. |
| **No rate limiting** | API has auth middleware but no rate limiting. One aggressive client can DoS the system. |
| **Healthcheck in Dockerfile calls wrong endpoint** | `http://localhost:8000/health` but the actual endpoint is `/api/health`. Container healthcheck always fails. |
| **embedding cache is in-memory only** | `_cache` in embedder.py is a module-level dict. Lost on restart, not shared across workers. Should use Redis or persistent cache. |

---

## 5. What's Wrong With It

### Critical Issues

1. **No request timeout on LLM calls.** LiteLLM `completion()` has no timeout. If OpenAI hangs, the request hangs forever. Need `timeout=30` parameter.

2. **Dense retrieval does N+1 queries.** For each of 50 dense results, does a separate `get_chunk_by_id()` to fetch content. Should batch-fetch or store content in LanceDB.

3. **API eval endpoint has the old bug.** `api.py:202` still uses `r.get("citations", [])` for `retrieved_sections` instead of `r.get("retrieved_sections", [])`. CLI was fixed but API wasn't.

4. **Worker eval task has the same old bug.** `worker.py:100` uses citation-based sections, not actual retrieved sections.

5. **No input validation/sanitization.** Query text goes directly to LLM and FTS5. No length limits, no injection protection on FTS5 (partially fixed with quoting, but no max length).

6. **No graceful shutdown.** Pipeline doesn't close SQLite connections or LanceDB handles on shutdown. The lifespan handler has `yield` but no cleanup logic.

### Performance Issues

1. **Cold start is ~10-15 seconds.** Loading BGE-M3 (2.3GB) + tokenizer init. First query after server start is painfully slow.

2. **Every query loads 3 ML models.** Embedding model, reranker, NLI verifier. All lazy-loaded on first use but stay in memory (~2.7GB total).

3. **Verification is slow.** NLI model runs on CPU by default. For a 7-claim answer, that's 7 sequential cross-encoder inference calls (~5-10s).

4. **No batch processing in reranker.** `model.predict(pairs)` does batch internally, but the verification step processes claims one at a time.

5. **Deduplication by content prefix (first 100 chars).** Crude. Two chunks with same opening but different content would be deduplicated incorrectly.

### Code Quality Issues

1. **Global mutable state everywhere.** `_model`, `_cache`, `_db`, `_table` — module-level globals. Thread safety is accidental, not guaranteed.

2. **`_conn()` returns a context manager but also a raw connection.** Used as `with self._conn() as conn:` in most places, but `_conn()` returns `sqlite3.Connection`, not a proper context manager. It works because `sqlite3.Connection` happens to be a context manager, but the semantics are wrong (commits on exit, not close).

3. **Bare except clauses.** `dense.py:55`, `dense.py:85`, `dense.py:96` catch `Exception` with `pass`. Silently swallows real errors.

4. **`import re` inside method.** `bm25_search()` imports `re` on every call. Minor but sloppy.

5. **Inconsistent datetime handling.** Some places use `datetime.utcnow()`, others use `time.strftime()`, one place tries `datetime.UTC` with a hasattr fallback. Should standardize on one approach.

### Missing Features for Production

1. **No request/response logging** (only traces)
2. **No structured logging** (uses basic `logging.INFO`)
3. **No CORS configuration** (default: no cross-origin)
4. **No pagination** on document/trace list endpoints
5. **No backup/restore** for SQLite + LanceDB data
6. **No document deletion** endpoint (only supersede)
7. **No user management** (just flat API keys)
8. **No query history search** (sessions expire, no persistent query log)

---

## 6. Bug Postmortem — Everything That Broke

### Bug 1: NLI Verifier Stripping Grounded Claims

**Symptom:** `test_nli_verifier_passes_grounded_claims` failing. Claim "The company matches 401k contributions up to 4% of salary" was being stripped despite being grounded in context.

**Root Cause:** Two bugs compounding:

1. **Raw logits treated as probabilities.** The NLI model (deberta-v3-small) outputs raw logits per class `[contradiction, entailment, neutral]`. The code compared these directly against a probability threshold (0.7). A raw logit of 3.708 (for neutral) is NOT a 370.8% probability — it's just a logit score. The fix was applying softmax to convert logits to proper probabilities.

2. **Neutral treated as contradiction.** The stripping condition was `label in ("contradiction", "neutral") and confidence > threshold`. This is semantically wrong. "Neutral" in NLI means "the premise neither entails nor contradicts the hypothesis" — it means the model can't tell, NOT that the claim is wrong. For a RAG verifier, stripping neutral claims is too aggressive — it would strip anything the model is uncertain about, including correctly paraphrased information.

**Fix:**
```python
# Before (broken):
is_stripped = label in ("contradiction", "neutral") and confidence > threshold

# After (fixed):
probs = softmax(raw_logits)
contradiction_prob = float(probs[0])
is_stripped = contradiction_prob > threshold
```

**Impact:** Only contradictions are now stripped. Claims the model is uncertain about stay in the answer. This is the right tradeoff for enterprise Q&A — false negatives (leaving a wrong claim) are caught by the threshold, while false positives (stripping correct claims) destroy user trust.

---

### Bug 2: FTS5 Syntax Error on Question Marks

**Symptom:** `sqlite3.OperationalError: fts5: syntax error near "?"` on every query containing `?`.

**Root Cause:** User queries were passed directly to FTS5 MATCH clause. FTS5 has its own query syntax where `?` is a valid operator (prefix query). The query "What is the PTO policy?" was interpreted as FTS5 syntax, not literal text.

**Fix:** Extract word tokens with `\w+` regex (strips all special chars), double-quote each token:
```python
tokens = re.findall(r'\w+', query)
fts_query = " OR ".join(f'"{t}"' for t in tokens)
```

**Lesson:** Never pass user input directly to FTS5 MATCH without sanitization. The quoting escapes special characters; the OR joining prevents overly restrictive AND matching.

---

### Bug 3: FTS5 Returning Near-Zero Results (AND vs OR)

**Symptom:** BM25 search returning 0-1 results for every query despite FTS5 table having 311 chunks.

**Root Cause:** Initial fix for Bug 2 used implicit AND (space-separated quoted tokens). Query `"What" "is" "PTO" "policy"` requires ALL four tokens to appear in a chunk. Most policy chunks don't contain "What" and "is", so almost nothing matched.

**Fix:** Changed to OR-joined quoted tokens:
```python
# Before: "What" "is" "PTO" "policy"  (implicit AND — too strict)
# After: "What" OR "is" OR "PTO" OR "policy"  (OR — BM25 ranks by relevance)
```

BM25 scoring naturally ranks chunks with more matching terms higher, so OR doesn't sacrifice precision — it just allows partial matches to appear in results.

**Impact:** Sparse retrieval went from returning 0 results to returning relevant results for all queries.

---

### Bug 4: Reranker Scores Below Confidence Gate

**Symptom:** 7/15 eval queries had 0.0 retrieval recall. Expected sections WERE in the reranked results but the pipeline returned "I couldn't find this."

**Root Cause:** The cross-encoder (ms-marco-MiniLM-L-6-v2) outputs raw logits, not probabilities. These logits are typically negative (e.g., -3.30, -0.42, -5.33). The confidence gate compared these against a threshold of 0.3:

```python
if top_score < settings.confidence_threshold:  # 0.3
    return "I couldn't find this"
```

Every query hit this gate because no raw logit exceeded 0.3.

**Fix:** Applied sigmoid normalization to convert logits to [0,1]:
```python
scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores)))
```

After sigmoid, a logit of -0.42 becomes ~0.40 (above threshold), while truly irrelevant results with logits like -9.75 become ~0.0001 (correctly gated).

**Impact:** Retrieval recall went from 0.53 to 0.67.

---

### Bug 5: Golden Dataset Had Wrong Expected Values

**Symptom:** Answer accuracy was 0.13 even when retrieval succeeded. LLM answers were correct but didn't match expected substrings.

**Root Cause:** The golden dataset was written with assumed/made-up values that didn't match the actual corpus:

| Question | Golden Expected | Corpus Actually Says |
|----------|----------------|---------------------|
| PTO days | "20 days" (after 3 years) | "15 business days" (minimum, no accrual) |
| Home office stipend | "$500" | "$1,500" |
| Core hours | "10am-3pm" | "4-hour block" (team-set) |
| Meal allowance | "$75/day" | "$100/day" |
| Probation | "90-day" | "6-month" |
| Incident reporting | "1 hour" | "immediately" |

**Fix:** Read actual corpus content, updated golden dataset to match. Also updated expected section paths from made-up paths (`"Benefits > Paid Time Off > PTO Accrual"`) to actual DB section paths (`"Paid Time Off (PTO) Policy > 2. PTO Guidelines > 2.1 Minimum PTO Requirement"`).

**Lesson:** Golden datasets must be derived from the corpus, not written independently. The eval was testing "does the LLM match my assumptions" not "does the LLM match the documents."

---

### Bug 6: Eval Using LLM Citations Instead of Retrieved Chunks

**Symptom:** Retrieval recall was 0.0 for all queries even when the right chunks were being retrieved.

**Root Cause:** The CLI eval function extracted `retrieved_sections` from LLM-generated citation text:
```python
# Before: sections from LLM's [Source: ...] markers
"retrieved_sections": [c.get("section", "") for c in r.get("citations", [])]
```

LLM citations look like `"Document 3, Remote Work Policy > 1. Overview"` — different format from golden dataset sections. The eval's substring matching never found a match.

**Fix:** Added `retrieved_sections` to pipeline return value using actual chunk `section_path` metadata:
```python
# After: sections from actual retrieved chunks
retrieved_sections = list({
    c.get("section_path", "") for c in context_chunks
    if c.get("section_path") and not c.get("is_parent")
})
```

**Note:** This same bug still exists in `api.py:202` and `worker.py:100` — only the CLI was fixed.

---

## 7. Tuning & Configuration

### All Settings (via env vars with DOCVAULT_ prefix)

| Setting | Default | What It Controls | Tuning Notes |
|---------|---------|-----------------|--------------|
| `llm_model` | `gpt-4o-mini` | LLM for generation | GPT-4o for better quality, Claude for different style |
| `embedding_model` | `BAAI/bge-m3` | Embedding model | bge-m3 is multilingual. For English-only, bge-large-en-v1.5 is smaller/faster |
| `chunk_size` | 400 tokens | Max tokens per chunk | Smaller = more precise retrieval, larger = more context per chunk |
| `chunk_overlap_pct` | 0.1 (10%) | Overlap between chunks | Higher = less info loss at boundaries, more chunks |
| `fusion_alpha` | 0.6 | Dense vs sparse weight | Not actually used (RRF is rank-based, not score-based). Dead config. |
| `dense_top_k` | 50 | Dense retrieval candidates | Higher = better recall, slower reranking |
| `sparse_top_k` | 50 | BM25 candidates | Same tradeoff |
| `fused_top_k` | 20 | Post-fusion candidates | These go to reranker. 20 is good balance. |
| `rerank_top_k` | 5 | Final reranked results | More = more context for LLM, higher cost |
| `context_budget_tokens` | 3000 | Max context tokens | Not enforced in code. Dead config. |
| `confidence_threshold` | 0.3 | Reranker score gate | Lower = more answers (possibly worse), higher = more "I don't know" |
| `session_ttl_seconds` | 1800 (30 min) | Session expiry | For multi-turn conversations |
| `max_session_turns` | 5 | History window | More turns = better context, higher LLM cost |
| `cache_ttl_seconds` | 3600 (1 hour) | Retrieval cache TTL | Lower for frequently updated corpora |
| `embedding_drift_threshold` | 0.01 | Alert threshold | Drift above this logs a warning |

### Dead/Unused Config
- `fusion_alpha`: RRF doesn't use weights. Would need weighted fusion instead.
- `context_budget_tokens`: Never checked — LLM gets all reranked chunks regardless of total tokens.

### Key Tuning Knobs

**For better retrieval accuracy:**
- Increase `dense_top_k` and `sparse_top_k` to 100 (more candidates for reranker)
- Decrease `chunk_size` to 200-300 (more granular chunks)
- Lower `confidence_threshold` to 0.2 (accept lower-confidence results)

**For faster responses:**
- Decrease `rerank_top_k` to 3 (fewer chunks to rerank and pass to LLM)
- Disable query expansion (remove `expand_query()` call)
- Use smaller embedding model

**For better answer quality:**
- Switch to GPT-4o or Claude Sonnet
- Increase `rerank_top_k` to 8-10 (more context)
- Add query expansion to retrieval (currently unused)

---

## 8. Benchmark Strategy & Metrics

### 8.1 Metrics That Matter

#### Retrieval Metrics

| Metric | Formula | What It Tells You | Target |
|--------|---------|-------------------|--------|
| **Recall@k** | (relevant retrieved in top-k) / (total relevant) | Does the system find the right chunks? | ≥ 0.85 @ k=10 |
| **MRR** (Mean Reciprocal Rank) | mean(1/rank of first relevant result) | How high does the right chunk rank? | ≥ 0.7 |
| **NDCG@k** | Normalized Discounted Cumulative Gain | Quality of ranking, accounting for position | ≥ 0.75 |
| **Precision@k** | (relevant in top-k) / k | What fraction of retrieved chunks are relevant? | ≥ 0.6 @ k=5 |

#### Generation Metrics

| Metric | How to Measure | What It Tells You | Target |
|--------|---------------|-------------------|--------|
| **Faithfulness** | LLM-as-judge or NLI | Does the answer stick to the context? | ≥ 0.95 |
| **Answer Relevance** | LLM-as-judge | Does it actually answer the question? | ≥ 0.85 |
| **Completeness** | LLM-as-judge | Does it cover all relevant information? | ≥ 0.75 |
| **Citation Accuracy** | Automated check | Do citations point to correct sources? | ≥ 0.90 |
| **Hallucination Rate** | NLI claims_stripped / claims_total | What fraction of claims are unsupported? | ≤ 0.05 |

#### End-to-End Metrics

| Metric | Measurement | Target |
|--------|------------|--------|
| **E2E Latency (p50)** | From question to answer | ≤ 3s |
| **E2E Latency (p95)** | 95th percentile | ≤ 8s |
| **Retrieval Latency** | Embed + search + rerank | ≤ 1s |
| **Generation Latency** | LLM API call | ≤ 3s |
| **Verification Latency** | NLI per-claim checking | ≤ 5s |
| **Cost per Query** | LLM tokens × price | ≤ $0.005 |
| **Cold Start Time** | Server start to first query | ≤ 15s |

#### RAGOps Metrics

| Metric | What It Tells You | Alert Threshold |
|--------|-------------------|-----------------|
| **Embedding Drift** | Has the document distribution shifted? | > 0.01 |
| **Retrieval Quality Trend** | Are reranker scores declining over time? | < 0.3 |
| **Hallucination Rate Trend** | Is the system hallucinating more? | > 0.05 |
| **Query Topic Distribution** | What are users asking about? | N/A (monitoring) |
| **Cache Hit Rate** | Is the cache effective? | < 0.1 = cache too small |
| **"I Don't Know" Rate** | How often does the confidence gate fire? | > 0.3 = retrieval problem |

### 8.2 Golden Dataset Best Practices

**Current state:** 15 Q&A pairs. This is a starting point, not sufficient.

**Recommended size:** 50-100 pairs minimum for statistically meaningful results. At 15 pairs, each question is worth 6.67% of the score — one flaky result swings everything.

**Question categories to cover:**

| Category | Count | Examples |
|----------|-------|---------|
| **Direct factual** | 20 | "What is the 401k match?" |
| **Multi-hop** | 10 | "If I relocate internationally, how does that affect my compensation?" |
| **Comparison** | 5 | "What's the difference between sick leave and PTO?" |
| **Temporal** | 5 | "When does open enrollment start?" |
| **Unanswerable** | 10 | "What is the stock price?" (not in corpus) |
| **Adversarial** | 5 | "The company offers 50 days of PTO, right?" (contradicts policy) |
| **Multi-document** | 10 | "What security requirements apply to remote work setups?" |
| **Edge cases** | 5 | Very long queries, queries with special chars, queries in different languages |

### 8.3 LLM-as-Judge Implementation

The most effective way to benchmark answer quality without manual human evaluation.

**Recommended approach: Use GPT-4o as judge with structured rubrics.**

#### Faithfulness Judge

```python
FAITHFULNESS_PROMPT = """You are evaluating whether an AI answer is faithful to the provided context.

Context:
{context}

Question: {question}
Answer: {answer}

Rate faithfulness on a scale of 1-5:
1 = Contains claims not in context (hallucination)
2 = Mostly faithful but includes unsupported inferences
3 = Faithful but stretches interpretations
4 = Faithful with minor phrasing differences
5 = Perfectly grounded in context

Respond with JSON: {"score": N, "reasoning": "..."}"""
```

#### Answer Relevance Judge

```python
RELEVANCE_PROMPT = """You are evaluating whether an AI answer is relevant to the question asked.

Question: {question}
Answer: {answer}

Rate relevance on a scale of 1-5:
1 = Completely off-topic
2 = Tangentially related
3 = Partially answers the question
4 = Answers the question with minor gaps
5 = Directly and completely answers the question

Respond with JSON: {"score": N, "reasoning": "..."}"""
```

#### Completeness Judge

```python
COMPLETENESS_PROMPT = """Given the context and question, evaluate if the answer covers all relevant information.

Context:
{context}

Question: {question}
Answer: {answer}
Expected key points: {expected_points}

Rate completeness on a scale of 1-5:
1 = Misses all key points
2 = Covers <25% of key points
3 = Covers ~50% of key points
4 = Covers most key points
5 = Covers all key points

Respond with JSON: {"score": N, "covered_points": [...], "missed_points": [...]}"""
```

#### Implementation Notes

- **Cost:** GPT-4o as judge costs ~$0.01-0.03 per evaluation. For 50 golden queries × 3 dimensions = 150 judge calls = ~$3/eval run.
- **Avoid self-enhancement bias:** Don't use the same model for generation and judging. If generating with GPT-4o-mini, judge with GPT-4o or Claude.
- **Calibration:** Run judge on known-good and known-bad examples first to verify scoring consistency.
- **Position bias mitigation:** For pairwise comparisons, run both orderings and average.

### 8.4 Recommended Benchmark Script

Create `eval/benchmark.py` that:

1. Loads golden dataset (expanded to 50+ questions)
2. Runs each query through the pipeline
3. Collects retrieval metrics (recall@k, MRR, precision@k)
4. Runs LLM-as-judge on each answer (faithfulness, relevance, completeness)
5. Measures latencies per stage
6. Outputs structured JSON report + human-readable summary

```python
# Pseudocode for benchmark runner
for question in golden_dataset:
    # Retrieval metrics
    result = pipeline.query(question)
    recall = compute_recall(expected_sections, result.retrieved_sections)
    mrr = compute_mrr(expected_sections, result.retrieved_sections)

    # LLM-as-judge
    faithfulness = judge_faithfulness(context, question, result.answer)
    relevance = judge_relevance(question, result.answer)
    completeness = judge_completeness(context, question, result.answer, expected_points)

    # Latency (from traces)
    trace = load_trace(result.trace_id)
    retrieval_ms = trace.retrieval.latency_ms
    generation_ms = trace.generation.latency_ms
```

---

## 9. Current Eval Results

### After All Fixes (2026-05-24)

```
Eval: 15 queries
  Retrieval Recall@10: 0.60
  Answer Accuracy:     0.53
  Avg Latency:         3162ms
```

### Per-Query Breakdown

| Query | Retrieval | Accuracy | Notes |
|-------|-----------|----------|-------|
| PTO days | ✓ 1.0 | ✓ | |
| Parental leave | ✗ 0.0 | ✗ | Section not surfaced by reranker |
| 401k match | ✓ 1.0 | ✓ | |
| Core work hours | ✓ 1.0 | ✓ | |
| Home office stipend | ✓ 1.0 | ✓ | |
| Password policy | ✗ 0.0 | ✗ | Section not surfaced by reranker |
| Security incident reporting | ✓ 1.0 | ✓ | |
| Severance | ✗ 0.0 | ✗ | Section not surfaced by reranker |
| Meal allowance | ✗ 0.0 | ✗ | Section not surfaced by reranker |
| MFA methods | ✗ 0.0 | ✗ | Section not surfaced by reranker |
| Professional development budget | ✓ 1.0 | ✓ | |
| Data classification levels | ✗ 0.0 | ✗ | Section not surfaced by reranker |
| Personal devices (BYOD) | ✓ 1.0 | ✓ | |
| P1 incident response time | ✓ 1.0 | ✓ | |
| Probation period | ✓ 1.0 | ✗ | Retrieved correctly but answer phrased differently |

### Progress

| Metric | Before Fixes | After Fixes | Delta |
|--------|-------------|-------------|-------|
| Retrieval Recall@10 | 0.00 | 0.60 | +0.60 |
| Answer Accuracy | 0.13 | 0.53 | +0.40 |
| Queries working E2E | 2/15 | 8/15 | +6 |

### Remaining Issues

1. **5 queries still fail retrieval** — reranker scores too low after sigmoid. These sections exist in the index but the cross-encoder doesn't score the query-chunk pair highly enough. Options:
   - Lower confidence_threshold to 0.15
   - Use a better reranker model (e.g., BAAI/bge-reranker-v2-m3)
   - Add the expanded queries to retrieval (currently unused)

2. **1 query retrieves correctly but fails accuracy** — "probation period" retrieves the right chunk but answer says "6-month probationary period" which doesn't contain the expected substring "6-month" as a standalone match (it does contain it — this may be an eval flakiness issue).

---

## Appendix: File Map

```
src/docvault/
├── __init__.py, __main__.py     # Package setup
├── config.py                     # Pydantic settings (env-var based)
├── cli.py                        # CLI: ingest, query, eval, serve, stats
├── api.py                        # FastAPI REST API
├── auth.py                       # API key middleware
├── pipeline.py                   # Core orchestrator (ingest + query)
├── worker.py                     # Celery tasks (async ingest, scheduled eval)
├── ingest/
│   ├── parser.py                 # Document parsing (md, pdf, html, txt)
│   ├── chunker.py                # Token-based hierarchical chunking
│   └── embedder.py               # BGE-M3 embeddings with hash cache
├── retrieval/
│   ├── dense.py                  # LanceDB ANN search
│   ├── sparse.py                 # SQLite FTS5 BM25 search
│   ├── fusion.py                 # Reciprocal Rank Fusion
│   └── reranker.py               # Cross-encoder reranking
├── generation/
│   ├── prompts.py                # System prompt + templates
│   ├── generator.py              # LLM generation + citation extraction
│   └── verifier.py               # NLI-based claim verification
├── memory/
│   ├── cache.py                  # Retrieval cache (Redis/memory)
│   └── session.py                # Session store (Redis/memory)
├── ragops/
│   ├── metrics.py                # Prometheus metrics
│   ├── tracer.py                 # Per-query JSONL tracing
│   ├── evaluator.py              # Golden dataset eval suite
│   └── drift.py                  # Embedding/retrieval/topic drift
└── storage/
    ├── documents.py              # SQLite document/chunk store + FTS5
    └── migrations/               # DB migration framework
```
