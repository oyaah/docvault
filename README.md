# DocVault

**A production-style RAG system for company-policy Q&A — that proves it works instead of just claiming it.**

Employees ask questions like *"How many PTO days do I get?"* DocVault answers from the actual policy documents, **cites the exact source**, and says **"I don't know"** when the answer isn't there. Every generated claim is verified against its sources before it reaches the user.

The interesting part isn't the pipeline — it's that the whole thing is backed by a **rigorous LLM-as-judge benchmark** (no exact-string matching, no cherry-picked demos) that measures faithfulness, hallucination, retrieval quality, citation accuracy, and refusal behaviour — and reports the warts honestly.

```
Python · FastAPI · Pinecone + SQLite FTS5 · OpenAI · ONNX Runtime · Celery/Redis · Prometheus/Grafana · Docker
```

---

## About it

- **Hybrid retrieval that actually fuses signals** — dense (Pinecone, OpenAI embeddings) + sparse (SQLite FTS5 BM25), merged with Reciprocal Rank Fusion, then re-ranked by an ONNX cross-encoder.
- ** hierarchical chunking** — small leaf chunks for precise retrieval, larger parent chunks fetched for context; tables kept atomic; section breadcrumbs embedded into each chunk to boost recall.
- **Hallucination control you can measure** — an NLI model strips claims the sources contradict; the benchmark then *measures* the residual hallucination rate (≈0.5%) rather than asserting "zero hallucination."
- **A real evaluation harness** — LLM-as-judge with a cross-family judge model, bootstrap confidence intervals, per-category breakdowns, and pass/fail gates. This is the part most RAG demos skip.
- **Layered memory** — three distinct tiers: *short-term* conversation memory (last 5 turns, 30-min TTL) so follow-up questions keep context, a *query→results cache* (LRU, 1-hr TTL, auto-invalidated on re-ingest) to skip repeat retrieval, and a *long-term* versioned knowledge store (SQLite + the dense index) where re-ingesting a policy supersedes the old version instead of overwriting it. All Redis-backed in production, with a transparent in-memory fallback for local dev.
- **Production surface that does real work** — async ingest runs as a Celery *task chain* (`ingest → quality-check → drift-check`) with scheduled beat jobs (nightly judge-eval, session cleanup, periodic drift reports); health/readiness probes gate traffic; API-key auth is **fail-closed**; and observability is wired end-to-end — Prometheus counters/histograms, a Grafana dashboard, and alert rules on a *windowed* hallucination rate, retrieval latency, and error rate. Ships as Docker Compose and AWS ECS/CloudFormation.

---

## Benchmark results

LLM-as-judge over **128 questions** in 6 categories, generated from and grounded in a real downloaded corpus of **40 US congressional bills**. Generator: `gpt-4o-mini`. Judge: `gpt-4o` (different model family, to avoid self-grading bias). Nothing is string-matched — every score is semantic.

| Metric | Mean | 95% CI |
|--------|:----:|:------:|
| **Faithfulness** (answer grounded in sources) | **0.995** | 0.985–1.000 |
| **Hallucination rate** (1 − faithfulness) | **0.5%** | — |
| Citation accuracy | 0.995 | 0.984–1.000 |
| Context recall | 0.840 | 0.776–0.901 |
| Answer relevancy | 0.833 | 0.763–0.896 |
| Answer correctness | 0.779 | 0.705–0.847 |
| Context precision | 0.674 | 0.606–0.743 |
| Refusal on unanswerable | 0.818 | — |

**What I learned reading my own numbers** (the honest part):
- Faithfulness is high because answers are short and extractive — easy mode for the judge, not proof of a hard task.
- The real weak spot is **refusing unanswerable questions** (0.82): the confidence gate was barely tuned. I shipped a calibration tool and raised the threshold to push this up.
- Context precision (0.67) reflects the precision/recall trade-off of pulling parent chunks — a deliberate choice that helps groundedness.

Full methodology, metric definitions, and integrity notes: **[`benchmark/README.md`](benchmark/README.md)**.

---

## How a query flows

```
                         ┌──────────────── INGEST ────────────────┐
   policy docs  ──▶  parse ──▶ hierarchical chunk ──▶ embed ──▶ index
   (MD/PDF/HTML)       │         (leaf + parent,        (OpenAI)   (Pinecone
                       │          tables atomic)                   + FTS5/BM25)
                       ▼
   ┌───────────────────────────── QUERY ─────────────────────────────┐
   question ─▶ expand ─▶ hybrid retrieve ─▶ RRF fuse ─▶ rerank ─▶ confidence gate
                          (dense + BM25)              (ONNX CE)    │
                                                                   ▼
              cite ◀─ NLI verify ◀─ generate (LLM) ◀─ assemble parent-chunk context
```

1. **Query expansion** — short questions are rewritten into search-friendly variants.
2. **Hybrid retrieval** — dense + BM25, fused via RRF → top candidates.
3. **Cross-encoder rerank** — `ms-marco-MiniLM-L-6-v2` (ONNX) selects the best few.
4. **Confidence gate** — below threshold → *"I don't know"* instead of guessing.
5. **Generation** — `gpt-4o-mini`, constrained to retrieved context, with inline `[Source: …]` citations.
6. **NLI verification** — `nli-deberta-v3-small` strips claims the sources contradict.

---

## Quickstart

```bash
pip install -e .
cp .env.example .env          # add OPENAI_API_KEY + DOCVAULT_PINECONE_API_KEY
python scripts/export_onnx.py # one-time: export reranker + NLI models to ONNX

docvault ingest corpus/                          # ingest the demo policy corpus
docvault query "How many PTO days do I get?"     # ask a question
```

Run the whole stack (API + worker + Redis + Prometheus + Grafana):

```bash
docker compose up
```

Reproduce the benchmark end-to-end:

```bash
python benchmark/download_corpus.py --num-docs 40
docvault ingest benchmark/corpus
python benchmark/generate_dataset.py -n 150
python benchmark/run_benchmark.py --judge-model gpt-4o
```

No Pinecone account? Run fully local: `export DOCVAULT_DENSE_BACKEND=local DOCVAULT_EMBEDDING_BACKEND=local`.

---

## Selected design decisions

| Decision | Why |
|----------|-----|
| Hybrid retrieval + RRF | Dense captures meaning, BM25 catches exact terms (policy numbers, dollar amounts); fusion beats either alone. |
| Parent/child chunking | Retrieve on precise leaves, generate from broader parent context — fixes the "isolated fragment" failure mode. |
| ONNX for reranker + NLI | No PyTorch at runtime → container drops from ~6 GB to ~800 MB. |
| LLM-as-judge, cross-family | Semantic scoring without brittle string matching; a different judge model avoids self-preference bias. |
| Three memory tiers, not one | Conversation memory, retrieval cache, and the versioned knowledge store solve different problems (context vs. latency vs. durability) and shouldn't share a lifetime or TTL. |
| Document versioning | Re-ingesting supersedes the prior version and retrieval defaults to `active` — answers can't silently come from stale policy. |
| Celery task chain for ingest | `ingest → quality-check → drift-check` keeps the API responsive and runs eval/drift automatically on new content, mirroring an agent-style handoff. |
| Fail-closed auth | No API keys configured ⇒ refuse traffic, not silently open. |
| Counters for hallucination SLO | Windowed PromQL rate, not a misleading last-value gauge. |

---

## Tech stack

| Layer | Choice |
|-------|--------|
| API | FastAPI + Uvicorn |
| Dense retrieval | Pinecone (or local NumPy backend) |
| Sparse retrieval | SQLite FTS5 (BM25) |
| Embeddings | OpenAI `text-embedding-3-small` (1536d) |
| Reranker / NLI | `ms-marco-MiniLM-L-6-v2` / `nli-deberta-v3-small`, via ONNX Runtime |
| Generation | `gpt-4o-mini` via LiteLLM |
| Async tasks | Celery + Redis (ingest chain + beat-scheduled eval/cleanup/drift) |
| Memory | Redis-backed session memory + retrieval cache; SQLite versioned document store (in-memory fallback) |
| Observability | Prometheus (counters/histograms) + Grafana dashboard & alerts + OpenTelemetry + structlog |
| Deploy | Docker Compose · AWS ECS Fargate + CloudFormation |

---

## Project layout

```
src/docvault/
├── pipeline.py          # query orchestration (retrieve → rerank → gate → generate → verify)
├── api.py · cli.py      # FastAPI app · CLI (ingest, query, serve, stats)
├── ingest/              # parser, hierarchical chunker, pluggable embedder
├── retrieval/           # dense (pinecone/local), sparse BM25, RRF fusion, ONNX reranker
├── generation/          # constrained generation + NLI verifier
├── storage/             # SQLite document versioning + FTS5
├── memory/              # session memory + retrieval cache (Redis or in-memory)
└── ragops/              # judges, judge-eval core, drift detection, tracing, metrics

benchmark/               # corpus downloader, question generator, judge harness, gate calibration
dashboards/              # Prometheus config, alert rules, Grafana dashboard + provisioning
deploy/                  # AWS ECS task/service + CloudFormation
```

---

*Built as a deep dive into doing RAG **properly** — retrieval quality, grounded generation, and an evaluation methodology honest enough to show where it still falls short.*
