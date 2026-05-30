# DocVault

Production-grade RAG pipeline for company policy Q&A. Employees ask questions about internal policies — DocVault answers from the actual documents, cites its sources, and says "I don't know" when it can't find the answer.

Hallucination-resistant by design: each generated claim is checked against its source
chunks with an NLI model, and claims that are *confidently contradicted* are stripped
before the answer reaches the user. (This reduces hallucination — measured at ~0.5% on
the benchmark — it does not eliminate it: unsupported-but-not-contradicted claims can
still pass, which is why faithfulness is measured directly.)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INGEST PIPELINE                         │
│                                                                 │
│  Documents ──→ Parse ──→ Chunk ──→ Embed ──→ Index              │
│  (PDF/MD/HTML)    (hierarchical)    (OpenAI)   (Pinecone+FTS5)  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                         QUERY PIPELINE                          │
│                                                                 │
│  Query ──→ Expand ──→ Hybrid Retrieve ──→ Rerank ──→ Filter     │
│              │        (BM25 + Dense)     (ONNX CE)  (confidence)│
│              │                                                  │
│              ▼                                                  │
│  Context Assembly ──→ Generate ──→ Verify ──→ Cite ──→ Respond  │
│  (parent chunks)      (LLM)       (NLI)      (inline)          │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                          RAGOPS LAYER                           │
│                                                                 │
│  Eval Suite ── Drift Detection ── Traces ── Metrics ── Alerts   │
└─────────────────────────────────────────────────────────────────┘
```

### How a query flows

1. **Query expansion** — LLM rewrites the user question into search-optimized sub-queries
2. **Hybrid retrieval** — Dense search (Pinecone, OpenAI embeddings) + Sparse search (SQLite FTS5 BM25), fused via Reciprocal Rank Fusion (α=0.6), top-20 candidates
3. **Cross-encoder reranking** — ONNX-exported `ms-marco-MiniLM-L-6-v2` scores query-document relevance, selects top-5
4. **Confidence gating** — If top reranker score < threshold → returns "I don't know" instead of guessing
5. **Generation** — GPT-4o-mini generates answer constrained to retrieved context, with inline citations
6. **NLI verification** — `nli-deberta-v3-small` checks each claim against source chunks. Contradictions are stripped from the final answer
7. **Citation linking** — Every surviving claim is linked back to (document, section, version)

## Tech Stack

| Component | Technology |
|-----------|------------|
| API | FastAPI + Uvicorn |
| Dense retrieval | Pinecone (managed vector DB) |
| Sparse retrieval | SQLite FTS5 (BM25) |
| Embeddings | OpenAI `text-embedding-3-small` (1536d) |
| Reranker | `ms-marco-MiniLM-L-6-v2` via ONNX Runtime |
| NLI verifier | `nli-deberta-v3-small` via ONNX Runtime |
| LLM | GPT-4o-mini via LiteLLM |
| Task queue | Celery + Redis |
| Metrics | Prometheus |
| Tracing | OpenTelemetry + structured logging (structlog) |
| Eval | LLM-as-judge benchmark (faithfulness, recall/precision, correctness, citation, refusal) |

## Project Structure

```
src/docvault/
├── api.py                  # FastAPI app with health probes
├── cli.py                  # CLI: ingest, query, eval
├── config.py               # Pydantic settings (env-driven)
├── pipeline.py             # Main query pipeline orchestration
├── worker.py               # Celery tasks + agentic task chains
├── observability.py        # Structlog + OpenTelemetry setup
├── auth.py                 # API key middleware
├── ingest/
│   ├── parser.py           # PDF/MD/HTML document parsing
│   ├── chunker.py          # Hierarchical chunking with overlap
│   └── embedder.py         # Pluggable backends (OpenAI / local)
├── retrieval/
│   ├── dense.py            # Pinecone vector search
│   ├── sparse.py           # SQLite FTS5 BM25 search
│   ├── fusion.py           # Reciprocal Rank Fusion
│   └── reranker.py         # ONNX cross-encoder reranking
├── generation/
│   └── verifier.py         # NLI hallucination detection
├── storage/
│   └── documents.py        # Document versioning + metadata
├── memory/
│   ├── session.py          # Conversation memory (5 turns, 30min TTL)
│   └── cache.py            # Retrieval cache (LRU, 1hr TTL)
└── ragops/
    ├── evaluator.py        # Eval suite runner
    ├── tracer.py           # Per-query trace logging
    ├── drift.py            # Embedding + retrieval drift detection
    └── metrics.py          # Prometheus counters/histograms

models/                     # ONNX models (not in git — export locally)
scripts/export_onnx.py      # PyTorch → ONNX model export
corpus/                     # Source policy documents (Markdown)
eval/golden_dataset.json    # 15-query evaluation dataset
deploy/                     # AWS ECS + CloudFormation configs
```

## Setup

### Prerequisites

- Python 3.11+
- Redis (for Celery task queue and caching)
- OpenAI API key
- Pinecone API key

### Install

```bash
pip install -e .
```

### Export ONNX models

The reranker and NLI verifier run as ONNX models (no PyTorch needed at runtime). Export once:

```bash
pip install -e ".[export]"
python scripts/export_onnx.py
```

This creates `models/reranker.onnx` (~87MB) and `models/verifier.onnx` (~542MB).

### Configure

```bash
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI_API_KEY=sk-...
#   DOCVAULT_PINECONE_API_KEY=pcsk_...
```

### Ingest documents

```bash
python -m docvault.cli ingest corpus/
```

### Query

```bash
python -m docvault.cli query "What is the PTO policy?"
```

### Run evaluation

The real evaluation is the LLM-as-judge benchmark (see `benchmark/README.md`):

```bash
python benchmark/download_corpus.py --num-docs 40   # real policy corpus (US bills)
docvault ingest benchmark/corpus
python benchmark/generate_dataset.py -n 150         # questions grounded in ingested chunks
python benchmark/run_benchmark.py --judge-model gpt-4o
```

The legacy substring eval (`docvault eval` over `eval/golden_dataset.json`) is
**deprecated** and kept only for reference — it matched exact strings, which is
exactly the brittleness the benchmark replaces.

### Docker

```bash
docker compose up
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/query` | Ask a question |
| `POST` | `/api/ingest` | Ingest documents |
| `GET` | `/api/health/ready` | Readiness probe |
| `GET` | `/api/health/live` | Liveness probe |
| `GET` | `/api/metrics` | Prometheus metrics |
| `POST` | `/api/eval/run` | Trigger eval suite |
| `GET` | `/api/eval/results` | Get eval results |

## Benchmark Results

LLM-as-judge benchmark — **128 questions** across 6 categories, generated from
and grounded in a real downloaded corpus of **40 US congressional bills**.
Generator: `gpt-4o-mini`. Judge: `gpt-4o` (cross-family, to avoid self-grading
bias). No exact-string matching anywhere; every score is semantic.

| Metric | Mean | 95% CI | n |
|--------|------|--------|---|
| Faithfulness (groundedness) | **0.995** | 0.985–1.000 | 101 |
| → Hallucination rate (1 − faithfulness) | **0.005** | — | — |
| Answer relevancy | 0.833 | 0.763–0.896 | 95 |
| Answer correctness (vs reference) | 0.779 | 0.705–0.847 | 95 |
| Context recall | 0.840 | 0.776–0.901 | 95 |
| Context precision | 0.674 | 0.606–0.743 | 95 |
| Citation accuracy | 0.995 | 0.984–1.000 | 95 |
| Refusal correctness (overall) | 0.859 | 0.797–0.922 | 128 |
| Refusal on unanswerable | 0.818 | — | 33 |

Latency p50 / p95: **6.2 s / 15.3 s** (dominated by LLM generation + query expansion).

### Honest reading of these numbers

- **Faithfulness ≈ 1.0 is real but easy here.** Answers are short and extractive
  with a constrained, low-temperature prompt, so claims rarely drift from context.
  It is *not* evidence of a hard task — discrimination shows up elsewhere.
- **The pipeline's real weaknesses surface honestly:** `context_precision 0.67`
  (parent chunks + top-7 context dilute precision), `answer_correctness 0.78`,
  and especially **false-premise rejection at 0.60** — the system answers ~40% of
  questions built on a false assumption instead of correcting them.
- **`table_lookup` is absent:** the bills corpus has no tables, so that stress
  category was dropped (the generator warns and renormalises). Re-add it by
  pointing `download_corpus.py` at a table-bearing corpus.
- **Refusal on unanswerable = 0.82** with `confidence_threshold = 0.0` (effectively
  off). Tuning the threshold on the `unanswerable` category should raise this.

Reproduce with `benchmark/run_benchmark.py`; full methodology in `benchmark/README.md`.

## Production hardening

| Concern | Behaviour |
|---------|-----------|
| **Auth** | Fail-closed: with no API keys configured, protected endpoints return 503. Set `DOCVAULT_API_KEYS` (or `data/api_keys.txt`), or `DOCVAULT_REQUIRE_AUTH=false` for local dev. |
| **Hallucination SLO** | Counters `docvault_claims_total` / `docvault_claims_stripped_total`; alert on `rate(stripped)/rate(total)`. The old per-query gauge is deprecated. |
| **Latency SLOs** | End-to-end is LLM-bound (p50≈6s, p95≈15s); alerts target that, plus a separate `stage="retrieval"` p95<1s alert for the part we control. |
| **Drift baseline** | Accumulated over the first `DOCVAULT_DRIFT_BASELINE_MIN_SAMPLES` (200) chunks, then frozen — not pinned to the first document. |
| **Offline / no Pinecone** | `DOCVAULT_DENSE_BACKEND=local` (NumPy brute-force) + `DOCVAULT_EMBEDDING_BACKEND=local`. |
| **Confidence gate** | Default `0.0` is off; calibrate with `benchmark/calibrate_gate.py` and set `DOCVAULT_CONFIDENCE_THRESHOLD`. |

Still open (genuinely hard, not yet done): load-testing the RAGOps surface (Celery/Grafana/OTel under real traffic), provider failover, and authz beyond a flat API-key list.

---

## Cloud Deployment (In Progress)

The infrastructure for AWS deployment is scaffolded but not yet fully wired up. What exists:

- **CloudFormation template** (`deploy/cloudformation.yml`) — VPC, ALB, ECS Fargate cluster, ElastiCache Redis, ECR, IAM roles, security groups
- **ECS task definition** (`deploy/ecs/task-definition.json`) — Container config for API + Celery worker
- **ECS service config** (`deploy/ecs/service.json`) — Fargate service with ALB target group
- **CI/CD pipeline** (`.github/workflows/ci.yml`) — Build, test, push to ECR
- **Dockerfile** — Multi-stage build, ONNX-only (no PyTorch), ~800MB image

What remains: wiring up secrets management (AWS Secrets Manager for API keys), setting up the actual AWS account resources, configuring the domain/SSL, and running the first deployment. The application code is deployment-ready — all config is environment-driven, health probes are implemented, and the Docker image builds cleanly.
