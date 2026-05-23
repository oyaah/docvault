"""FastAPI application — DocVault REST API."""

import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from docvault.config import settings
from docvault.pipeline import DocVaultPipeline
from docvault.ragops.metrics import get_metrics_text
from docvault.ragops.evaluator import run_eval_suite, save_eval_results, load_golden_dataset
from docvault.ragops.drift import generate_drift_report
from docvault.ragops.tracer import load_traces

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline: DocVaultPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    settings.ensure_dirs()
    pipeline = DocVaultPipeline()
    logger.info("DocVault pipeline initialized")
    yield
    logger.info("DocVault shutting down")


app = FastAPI(
    title="DocVault",
    description="Industry-grade RAG pipeline for company policy Q&A",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request/Response Models ──────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    filters: dict | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    confidence: str
    verification: dict | None = None
    trace_id: str


class IngestRequest(BaseModel):
    source_path: str
    version: str = "1.0"
    effective_date: str | None = None
    doc_type: str | None = None


class IngestResponse(BaseModel):
    results: list[dict]
    total_documents: int
    total_chunks: int
    time_ms: int


# ── Endpoints ────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")

    try:
        result = pipeline.query(
            question=req.question,
            session_id=req.session_id,
            filters=req.filters,
        )
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(500, f"Query failed: {str(e)}")


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")

    source = Path(req.source_path)
    if not source.exists():
        raise HTTPException(404, f"Path not found: {req.source_path}")

    t0 = time.time()

    if source.is_dir():
        results = pipeline.ingest_directory(
            source, version=req.version, doc_type=req.doc_type
        )
    else:
        result = pipeline.ingest_file(
            source,
            version=req.version,
            effective_date=req.effective_date,
            doc_type=req.doc_type,
        )
        results = [result]

    total_chunks = sum(r.get("chunks", 0) for r in results if "error" not in r)
    elapsed = int((time.time() - t0) * 1000)

    return IngestResponse(
        results=results,
        total_documents=len([r for r in results if "error" not in r]),
        total_chunks=total_chunks,
        time_ms=elapsed,
    )


@app.get("/api/health")
async def health():
    if not pipeline:
        return {"status": "initializing"}
    stats = pipeline.store.stats()
    return {
        "status": "healthy",
        "active_documents": stats["active_documents"],
        "active_chunks": stats["active_chunks"],
        "cache": pipeline.cache.stats(),
    }


@app.get("/api/metrics")
async def metrics_endpoint():
    content_type, text = get_metrics_text()
    return PlainTextResponse(text, media_type=content_type)


@app.post("/api/eval/run")
async def run_eval():
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")

    golden = load_golden_dataset()
    if not golden:
        return {"error": "No golden dataset found at eval/golden_dataset.json"}

    def query_fn(question: str) -> dict:
        t0 = time.time()
        result = pipeline.query(question)
        return {
            "answer": result["answer"],
            "retrieved_sections": [c.get("section", "") for c in result.get("citations", [])],
            "latency_ms": (time.time() - t0) * 1000,
        }

    suite_result = run_eval_suite(query_fn)
    save_eval_results(suite_result)

    return {
        "total_queries": suite_result.total_queries,
        "avg_retrieval_recall": suite_result.avg_retrieval_recall,
        "avg_latency_ms": suite_result.avg_latency_ms,
        "answer_accuracy": suite_result.answer_accuracy,
    }


@app.get("/api/eval/results")
async def eval_results():
    results_path = settings.data_dir / "eval_results.json"
    if not results_path.exists():
        return {"error": "No eval results. Run POST /api/eval/run first."}
    import json
    return json.loads(results_path.read_text())


@app.get("/api/drift")
async def drift_report():
    report = generate_drift_report()
    return {
        "retrieval_quality_trend": report.retrieval_quality_trend,
        "hallucination_rate": report.hallucination_rate,
        "query_volume_7d": report.query_volume_7d,
        "timestamp": report.timestamp,
    }


@app.get("/api/traces")
async def traces(limit: int = 50):
    return load_traces(limit=limit)


@app.get("/api/documents")
async def list_documents():
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    docs = pipeline.store.get_active_documents()
    return [
        {
            "id": d.id,
            "title": d.title,
            "version": d.version,
            "effective_date": d.effective_date,
            "total_chunks": d.total_chunks,
            "ingested_at": d.ingested_at,
        }
        for d in docs
    ]
