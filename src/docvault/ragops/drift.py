"""Drift detection — embedding distribution, retrieval quality, query patterns."""

import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from docvault.config import settings
from docvault.ragops.tracer import load_traces


@dataclass
class DriftReport:
    embedding_drift: float | None  # cosine sim shift from baseline
    retrieval_quality_trend: float | None  # 7-day avg reranker score change
    avg_confidence: float | None
    hallucination_rate: float | None
    query_volume_7d: int
    timestamp: str = ""


def compute_embedding_drift(
    new_embeddings: np.ndarray,
    baseline_path: Path | None = None,
) -> float | None:
    """Compare new embedding distribution against saved baseline.

    Returns magnitude of mean-vector shift (0 = no drift, higher = more drift).
    """
    bpath = baseline_path or (settings.data_dir / "embedding_baseline.npy")

    if not bpath.exists():
        # Save current as baseline
        np.save(str(bpath), np.mean(new_embeddings, axis=0))
        return None

    baseline_mean = np.load(str(bpath))
    new_mean = np.mean(new_embeddings, axis=0)

    # Cosine similarity between baseline and new mean
    cos_sim = np.dot(baseline_mean, new_mean) / (
        np.linalg.norm(baseline_mean) * np.linalg.norm(new_mean) + 1e-8
    )
    drift = 1.0 - float(cos_sim)
    return round(drift, 6)


def compute_retrieval_quality_trend(days: int = 7) -> float | None:
    """Compute average reranker top score over last N days from traces."""
    traces = load_traces(limit=1000)
    if not traces:
        return None

    cutoff = time.time() - (days * 86400)
    recent_scores = []

    for trace in traces:
        ts = trace.get("timestamp", "")
        try:
            trace_time = time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))
        except (ValueError, OverflowError):
            continue

        if trace_time >= cutoff:
            top_scores = trace.get("retrieval", {}).get("reranked_top_scores", [])
            if top_scores:
                recent_scores.append(max(top_scores))

    if not recent_scores:
        return None
    return round(float(np.mean(recent_scores)), 4)


def compute_hallucination_rate(days: int = 7) -> float | None:
    """Compute rolling hallucination rate from traces."""
    traces = load_traces(limit=1000)
    if not traces:
        return None

    cutoff = time.time() - (days * 86400)
    total_claims = 0
    stripped_claims = 0

    for trace in traces:
        ts = trace.get("timestamp", "")
        try:
            trace_time = time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))
        except (ValueError, OverflowError):
            continue

        if trace_time >= cutoff:
            v = trace.get("verification", {})
            total_claims += v.get("claims_total", 0)
            stripped_claims += v.get("claims_stripped", 0)

    if total_claims == 0:
        return None
    return round(stripped_claims / total_claims, 4)


def generate_drift_report() -> DriftReport:
    """Generate a comprehensive drift report."""
    traces = load_traces(limit=1000)
    cutoff = time.time() - (7 * 86400)

    query_volume = sum(
        1 for t in traces
        if _parse_time(t.get("timestamp", "")) and _parse_time(t["timestamp"]) >= cutoff
    )

    return DriftReport(
        embedding_drift=None,  # computed separately during ingest
        retrieval_quality_trend=compute_retrieval_quality_trend(),
        avg_confidence=None,
        hallucination_rate=compute_hallucination_rate(),
        query_volume_7d=query_volume,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def _parse_time(ts: str) -> float | None:
    try:
        return time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))
    except (ValueError, OverflowError):
        return None
