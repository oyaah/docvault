# DocVault

Production-grade RAG pipeline for company policy Q&A. Employees ask questions about internal policies — DocVault answers from the actual documents, cites its sources, and says "I don't know" when it can't find the answer.

Built with zero-hallucination architecture: every claim is verified against source documents using NLI before reaching the user.

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
| Eval | Golden dataset + retrieval recall, answer accuracy |

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

```bash
python -m docvault.cli eval
```

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

## Current Eval Metrics

| Metric | Score |
|--------|-------|
| Retrieval Recall@10 | 53.3% |
| Answer Accuracy | 46.7% |
| NLI Verification | 5/5 claims verified on sample queries |

Evaluated on a 15-query golden dataset covering 10 HR policy documents.

---

## Cloud Deployment (In Progress)

The infrastructure for AWS deployment is scaffolded but not yet fully wired up. What exists:

- **CloudFormation template** (`deploy/cloudformation.yml`) — VPC, ALB, ECS Fargate cluster, ElastiCache Redis, ECR, IAM roles, security groups
- **ECS task definition** (`deploy/ecs/task-definition.json`) — Container config for API + Celery worker
- **ECS service config** (`deploy/ecs/service.json`) — Fargate service with ALB target group
- **CI/CD pipeline** (`.github/workflows/ci.yml`) — Build, test, push to ECR
- **Dockerfile** — Multi-stage build, ONNX-only (no PyTorch), ~800MB image

What remains: wiring up secrets management (AWS Secrets Manager for API keys), setting up the actual AWS account resources, configuring the domain/SSL, and running the first deployment. The application code is deployment-ready — all config is environment-driven, health probes are implemented, and the Docker image builds cleanly.
