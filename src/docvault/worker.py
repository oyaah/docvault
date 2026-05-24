"""Celery worker — async document ingestion, scheduled eval, and agentic task chains.

Task chain pattern (agentic memory handoff):
  ingest_file → quality_check → drift_check
  Each task writes results to Redis, next task picks up from there.
  This is the Celery equivalent of agent-to-agent memory passing.
"""

import logging
from pathlib import Path

from celery import Celery, chain
from celery.schedules import crontab

from docvault.config import settings

logger = logging.getLogger(__name__)

app = Celery(
    "docvault",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # one task at a time per worker (heavy work)
    result_expires=3600,
)

# Scheduled tasks
app.conf.beat_schedule = {
    "nightly-eval": {
        "task": "docvault.worker.run_eval_suite_task",
        "schedule": crontab(hour=2, minute=0),  # 2 AM UTC
    },
    "cleanup-expired-sessions": {
        "task": "docvault.worker.cleanup_sessions",
        "schedule": crontab(minute="*/30"),
    },
    "drift-report": {
        "task": "docvault.worker.compute_drift_report",
        "schedule": crontab(hour="*/6"),
    },
}


def _get_pipeline():
    """Lazy pipeline init for worker processes."""
    from docvault.pipeline import DocVaultPipeline
    settings.ensure_dirs()
    return DocVaultPipeline()


# ── Ingest Tasks ────────────────────────────────────────

@app.task(bind=True, name="docvault.worker.ingest_file")
def ingest_file_task(self, file_path: str, version: str = "1.0",
                     effective_date: str | None = None, doc_type: str | None = None) -> dict:
    """Async file ingestion — offloaded from API request thread."""
    logger.info(f"[worker] Ingesting {file_path}")
    self.update_state(state="PROCESSING", meta={"file": file_path})

    pipe = _get_pipeline()
    try:
        result = pipe.ingest_file(
            Path(file_path), version=version,
            effective_date=effective_date, doc_type=doc_type,
        )
        return result
    except Exception as e:
        logger.error(f"[worker] Ingest failed for {file_path}: {e}")
        return {"file": file_path, "error": str(e)}


@app.task(bind=True, name="docvault.worker.ingest_directory")
def ingest_directory_task(self, dir_path: str, version: str = "1.0",
                          doc_type: str | None = None) -> list[dict]:
    """Async directory ingestion."""
    logger.info(f"[worker] Ingesting directory {dir_path}")
    self.update_state(state="PROCESSING", meta={"directory": dir_path})

    pipe = _get_pipeline()
    results = pipe.ingest_directory(Path(dir_path), version=version, doc_type=doc_type)
    return results


# ── Quality Check Task ──────────────────────────────────

@app.task(name="docvault.worker.quality_check")
def quality_check_task(ingest_result: dict | list) -> dict:
    """Post-ingestion quality check — runs eval on recently ingested documents.

    Part of the agentic pipeline chain: ingest → quality_check → drift_check.
    Receives ingest result from upstream task via Celery chain.
    """
    import time
    from docvault.ragops.evaluator import run_eval_suite, save_eval_results

    logger.info("[worker] Running post-ingest quality check")
    pipe = _get_pipeline()

    def query_fn(question: str) -> dict:
        t0 = time.time()
        r = pipe.query(question)
        return {
            "answer": r["answer"],
            "retrieved_sections": r.get("retrieved_sections", []),
            "latency_ms": (time.time() - t0) * 1000,
        }

    result = run_eval_suite(query_fn)
    save_eval_results(result)

    status = "pass" if result.avg_retrieval_recall >= 0.5 else "degraded"
    logger.info(
        f"[worker] Quality check: {status} "
        f"(recall={result.avg_retrieval_recall:.3f}, accuracy={result.answer_accuracy:.3f})"
    )

    return {
        "status": status,
        "recall": result.avg_retrieval_recall,
        "accuracy": result.answer_accuracy,
        "ingest_result": ingest_result,
    }


# ── Drift Check Task ───────────────────────────────────

@app.task(name="docvault.worker.drift_check")
def drift_check_task(quality_result: dict) -> dict:
    """Post-quality drift check — computes embedding and retrieval drift.

    Final task in the agentic chain: ingest → quality_check → drift_check.
    Triggers re-indexing alert if drift exceeds thresholds.
    """
    from docvault.ragops.drift import generate_drift_report
    import json

    logger.info("[worker] Running post-quality drift check")
    report = generate_drift_report()

    alerts = []
    if report.hallucination_rate is not None and report.hallucination_rate > 0.05:
        alerts.append(f"hallucination_rate={report.hallucination_rate:.2%}")
    if report.retrieval_quality_trend is not None and report.retrieval_quality_trend < 0.3:
        alerts.append(f"retrieval_quality={report.retrieval_quality_trend:.3f}")

    if alerts:
        logger.warning(f"[ALERT] Drift thresholds exceeded: {', '.join(alerts)}")

    # Save report
    report_path = settings.data_dir / "drift_report.json"
    report_path.write_text(json.dumps({
        "embedding_drift": report.embedding_drift,
        "retrieval_quality_trend": report.retrieval_quality_trend,
        "hallucination_rate": report.hallucination_rate,
        "query_volume_7d": report.query_volume_7d,
        "query_topics": report.query_topics,
        "timestamp": report.timestamp,
        "quality_check": quality_result,
        "alerts": alerts,
    }, indent=2))

    return {
        "status": "ok",
        "alerts": alerts,
        "timestamp": report.timestamp,
        "quality_status": quality_result.get("status", "unknown"),
    }


# ── Agentic Chain ───────────────────────────────────────

def ingest_with_pipeline(file_path: str, **kwargs) -> chain:
    """Create an agentic task chain: ingest → quality_check → drift_check.

    Each task passes its result to the next via Celery chain.
    This is the production equivalent of agent memory handoff:
    Agent A completes work → writes result → Agent B picks up from there.

    Usage:
        task_chain = ingest_with_pipeline("/path/to/doc.md", version="2.0")
        result = task_chain.apply_async()
    """
    return chain(
        ingest_file_task.s(file_path, **kwargs),
        quality_check_task.s(),
        drift_check_task.s(),
    )


# ── Standalone Scheduled Tasks ──────────────────────────

@app.task(name="docvault.worker.run_eval_suite_task")
def run_eval_suite_task() -> dict:
    """Scheduled eval suite execution (nightly)."""
    import time
    from docvault.ragops.evaluator import run_eval_suite, save_eval_results

    logger.info("[worker] Running scheduled eval suite")
    pipe = _get_pipeline()

    def query_fn(question: str) -> dict:
        t0 = time.time()
        r = pipe.query(question)
        return {
            "answer": r["answer"],
            "retrieved_sections": r.get("retrieved_sections", []),
            "latency_ms": (time.time() - t0) * 1000,
        }

    result = run_eval_suite(query_fn)
    save_eval_results(result)

    logger.info(
        f"[worker] Eval complete: recall={result.avg_retrieval_recall:.3f}, "
        f"accuracy={result.answer_accuracy:.3f}"
    )
    return {
        "total_queries": result.total_queries,
        "avg_retrieval_recall": result.avg_retrieval_recall,
        "answer_accuracy": result.answer_accuracy,
    }


@app.task(name="docvault.worker.cleanup_sessions")
def cleanup_sessions() -> dict:
    """Periodic session cleanup (for in-memory fallback only; Redis handles TTL natively)."""
    from docvault.memory.session import SessionStore
    store = SessionStore()
    cleaned = store.cleanup_expired()
    return {"cleaned": cleaned}


@app.task(name="docvault.worker.compute_drift_report")
def compute_drift_report() -> dict:
    """Periodic drift detection (standalone, not part of chain)."""
    from docvault.ragops.drift import generate_drift_report
    import json

    report = generate_drift_report()

    if report.hallucination_rate is not None and report.hallucination_rate > 0.05:
        logger.warning(f"[ALERT] Hallucination rate {report.hallucination_rate:.2%} exceeds 5% threshold")

    if report.retrieval_quality_trend is not None and report.retrieval_quality_trend < 0.3:
        logger.warning(f"[ALERT] Retrieval quality trend {report.retrieval_quality_trend:.3f} is below threshold")

    report_path = settings.data_dir / "drift_report.json"
    report_path.write_text(json.dumps({
        "embedding_drift": report.embedding_drift,
        "retrieval_quality_trend": report.retrieval_quality_trend,
        "hallucination_rate": report.hallucination_rate,
        "query_volume_7d": report.query_volume_7d,
        "query_topics": report.query_topics,
        "timestamp": report.timestamp,
    }, indent=2))

    return {"status": "ok", "timestamp": report.timestamp}
