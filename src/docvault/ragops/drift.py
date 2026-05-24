"""Drift detection — embedding distribution, retrieval quality, query topic clustering."""

import json
import time
import logging
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from docvault.config import settings
from docvault.ragops.tracer import load_traces

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    embedding_drift: float | None  # cosine sim shift from baseline
    retrieval_quality_trend: float | None  # 7-day avg reranker score change
    avg_confidence: float | None
    hallucination_rate: float | None
    query_volume_7d: int
    query_topics: list[dict] = field(default_factory=list)  # [{topic, count, example}]
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

    # Handle dimension mismatch (model changed)
    if baseline_mean.shape != new_mean.shape:
        np.save(str(bpath), new_mean)
        logger.warning("Embedding dimension changed — reset baseline")
        return None

    cos_sim = np.dot(baseline_mean, new_mean) / (
        np.linalg.norm(baseline_mean) * np.linalg.norm(new_mean) + 1e-8
    )
    drift = 1.0 - float(cos_sim)

    if drift > settings.embedding_drift_threshold:
        logger.warning(f"Embedding drift detected: {drift:.6f} (threshold: {settings.embedding_drift_threshold})")

    return round(drift, 6)


def compute_retrieval_quality_trend(days: int = 7) -> float | None:
    """Compute average reranker top score over last N days from traces."""
    traces = load_traces(limit=1000)
    if not traces:
        return None

    cutoff = time.time() - (days * 86400)
    recent_scores = []

    for trace in traces:
        trace_time = _parse_time(trace.get("timestamp", ""))
        if trace_time and trace_time >= cutoff:
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
        trace_time = _parse_time(trace.get("timestamp", ""))
        if trace_time and trace_time >= cutoff:
            v = trace.get("verification", {})
            total_claims += v.get("claims_total", 0)
            stripped_claims += v.get("claims_stripped", 0)

    if total_claims == 0:
        return None
    return round(stripped_claims / total_claims, 4)


def compute_query_topics(days: int = 7, max_topics: int = 10) -> list[dict]:
    """Cluster recent queries into topic groups using keyword extraction.

    Uses TF-based keyword extraction (no extra ML models) to identify
    recurring query themes. Detects new topic emergence for coverage gaps.
    """
    traces = load_traces(limit=1000)
    if not traces:
        return []

    cutoff = time.time() - (days * 86400)
    queries = []

    for trace in traces:
        trace_time = _parse_time(trace.get("timestamp", ""))
        if trace_time and trace_time >= cutoff:
            q = trace.get("query", "").strip()
            if q:
                queries.append(q)

    if not queries:
        return []

    # Extract keywords from queries
    stop_words = {
        "what", "is", "the", "how", "do", "does", "can", "i", "my", "a", "an",
        "for", "to", "in", "of", "and", "or", "are", "we", "our", "get", "about",
        "much", "many", "long", "when", "where", "who", "which", "that", "this",
        "with", "have", "has", "it", "be", "on", "at", "by", "if", "me", "there",
    }

    # Build keyword frequency
    keyword_freq: Counter = Counter()
    keyword_queries: dict[str, list[str]] = {}

    for q in queries:
        words = [w.lower().strip("?.,!") for w in q.split()]
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        for kw in keywords:
            keyword_freq[kw] += 1
            if kw not in keyword_queries:
                keyword_queries[kw] = []
            keyword_queries[kw].append(q)

    # Group queries by dominant keyword into topics
    topics = []
    assigned_queries: set[str] = set()

    for keyword, count in keyword_freq.most_common(max_topics * 2):
        if count < 2:
            break

        # Find unassigned queries matching this keyword
        matching = [q for q in keyword_queries[keyword] if q not in assigned_queries]
        if not matching:
            continue

        for q in matching:
            assigned_queries.add(q)

        topics.append({
            "topic": keyword,
            "count": len(matching),
            "example": matching[0],
            "total_mentions": count,
        })

        if len(topics) >= max_topics:
            break

    # Detect new topics (appeared only in last 24h)
    recent_cutoff = time.time() - 86400
    recent_queries = []
    older_queries = []

    for trace in traces:
        trace_time = _parse_time(trace.get("timestamp", ""))
        if not trace_time or trace_time < cutoff:
            continue
        q = trace.get("query", "").strip()
        if not q:
            continue
        if trace_time >= recent_cutoff:
            recent_queries.append(q)
        else:
            older_queries.append(q)

    if recent_queries and older_queries:
        recent_keywords = set()
        for q in recent_queries:
            words = [w.lower().strip("?.,!") for w in q.split()]
            recent_keywords.update(w for w in words if w not in stop_words and len(w) > 2)

        older_keywords = set()
        for q in older_queries:
            words = [w.lower().strip("?.,!") for w in q.split()]
            older_keywords.update(w for w in words if w not in stop_words and len(w) > 2)

        new_keywords = recent_keywords - older_keywords
        if new_keywords:
            for kw in list(new_keywords)[:3]:
                topics.append({
                    "topic": kw,
                    "count": 1,
                    "example": next((q for q in recent_queries if kw in q.lower()), ""),
                    "is_new": True,
                })

    return topics


def generate_drift_report() -> DriftReport:
    """Generate a comprehensive drift report."""
    traces = load_traces(limit=1000)
    cutoff = time.time() - (7 * 86400)

    query_volume = sum(
        1 for t in traces
        if _parse_time(t.get("timestamp", "")) and _parse_time(t["timestamp"]) >= cutoff
    )

    # Compute average confidence
    confidences = []
    for t in traces:
        trace_time = _parse_time(t.get("timestamp", ""))
        if trace_time and trace_time >= cutoff:
            top_scores = t.get("retrieval", {}).get("reranked_top_scores", [])
            if top_scores:
                confidences.append(max(top_scores))

    avg_conf = round(float(np.mean(confidences)), 4) if confidences else None

    return DriftReport(
        embedding_drift=None,  # computed separately during ingest
        retrieval_quality_trend=compute_retrieval_quality_trend(),
        avg_confidence=avg_conf,
        hallucination_rate=compute_hallucination_rate(),
        query_volume_7d=query_volume,
        query_topics=compute_query_topics(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def _parse_time(ts: str) -> float | None:
    try:
        return time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))
    except (ValueError, OverflowError):
        return None
