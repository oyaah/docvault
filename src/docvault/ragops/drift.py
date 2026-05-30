"""Drift detection — embedding distribution, retrieval quality, semantic topic clustering."""

import time
import logging
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from docvault.config import settings
from docvault.ragops.tracer import load_traces

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DriftReport:
    embedding_drift: float | None
    retrieval_quality_trend: float | None
    avg_confidence: float | None
    hallucination_rate: float | None
    query_volume_7d: int
    query_topics: list[dict] = field(default_factory=list)
    timestamp: str = ""


def compute_embedding_drift(
    new_embeddings: np.ndarray,
    baseline_path: Path | None = None,
) -> float | None:
    """Cosine drift of a new batch's mean vs a frozen reference baseline.

    The baseline is accumulated as a running mean over the first
    `drift_baseline_min_samples` chunks, then frozen — so it reflects the initial
    corpus distribution rather than whatever single document was ingested first.
    Returns None while the baseline is still warming up.
    """
    bpath = baseline_path or (settings.data_dir / "embedding_baseline.npz")
    new_mean = np.mean(new_embeddings, axis=0)
    n_new = len(new_embeddings)
    min_samples = settings.drift_baseline_min_samples

    if not bpath.exists():
        np.savez(str(bpath), mean=new_mean, count=n_new, frozen=(n_new >= min_samples))
        return None

    data = np.load(str(bpath))
    base_mean, count, frozen = data["mean"], int(data["count"]), bool(data["frozen"])

    if base_mean.shape != new_mean.shape:
        np.savez(str(bpath), mean=new_mean, count=n_new, frozen=False)
        logger.warning("Embedding dimension changed — reset baseline")
        return None

    if not frozen:
        total = count + n_new
        merged = (base_mean * count + new_mean * n_new) / total
        np.savez(str(bpath), mean=merged, count=total, frozen=(total >= min_samples))
        return None

    cos_sim = float(np.dot(base_mean, new_mean) / (
        np.linalg.norm(base_mean) * np.linalg.norm(new_mean) + 1e-8
    ))
    drift = round(1.0 - cos_sim, 6)
    if drift > settings.embedding_drift_threshold:
        logger.warning(f"Embedding drift detected: {drift:.6f}")
    return drift


def compute_retrieval_quality_trend(days: int = 7) -> float | None:
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
    """Cluster recent queries semantically using embedding cosine similarity.

    Groups queries into topic clusters by:
    1. Embedding all queries
    2. Greedy clustering: assign each query to nearest existing cluster
       or create new cluster if similarity < threshold
    3. Return clusters with representative example and count
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

    if len(queries) < 2:
        return []

    # Embed all queries
    try:
        from docvault.ingest.embedder import embed_texts
        embeddings = embed_texts(queries)
    except Exception as e:
        logger.warning(f"Failed to embed queries for topic clustering: {e}")
        return _keyword_fallback(queries, max_topics)

    # Greedy clustering with cosine similarity
    similarity_threshold = 0.75
    clusters: list[dict] = []  # {centroid: np.array, queries: [str], indices: [int]}

    for i, emb in enumerate(embeddings):
        best_cluster = None
        best_sim = -1.0

        for cluster in clusters:
            sim = float(np.dot(emb, cluster["centroid"]) / (
                np.linalg.norm(emb) * np.linalg.norm(cluster["centroid"]) + 1e-8
            ))
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        if best_cluster and best_sim >= similarity_threshold:
            best_cluster["queries"].append(queries[i])
            best_cluster["indices"].append(i)
            # Update centroid (running mean)
            n = len(best_cluster["indices"])
            best_cluster["centroid"] = (
                best_cluster["centroid"] * (n - 1) + emb
            ) / n
        else:
            clusters.append({
                "centroid": emb.copy(),
                "queries": [queries[i]],
                "indices": [i],
            })

    # Sort by cluster size, return top N
    clusters.sort(key=lambda c: len(c["queries"]), reverse=True)

    topics = []
    for cluster in clusters[:max_topics]:
        if len(cluster["queries"]) < 1:
            continue
        topics.append({
            "topic": _extract_topic_label(cluster["queries"]),
            "count": len(cluster["queries"]),
            "example": cluster["queries"][0],
        })

    # Detect new topics (only in last 24h, not in older traces)
    recent_cutoff = time.time() - 86400
    recent_queries = set()
    older_queries = set()

    for trace in traces:
        trace_time = _parse_time(trace.get("timestamp", ""))
        if not trace_time or trace_time < cutoff:
            continue
        q = trace.get("query", "").strip()
        if not q:
            continue
        if trace_time >= recent_cutoff:
            recent_queries.add(q)
        else:
            older_queries.add(q)

    if recent_queries and older_queries:
        # Find queries in recent that don't cluster with any older query
        try:
            recent_embs = embed_texts(list(recent_queries))
            older_embs = embed_texts(list(older_queries))

            # For each recent query, check max similarity to any older query
            for i, r_emb in enumerate(recent_embs):
                sims = np.dot(older_embs, r_emb) / (
                    np.linalg.norm(older_embs, axis=1) * np.linalg.norm(r_emb) + 1e-8
                )
                if np.max(sims) < 0.6:
                    topics.append({
                        "topic": list(recent_queries)[i],
                        "count": 1,
                        "example": list(recent_queries)[i],
                        "is_new": True,
                    })
        except Exception:
            pass

    return topics


def _extract_topic_label(queries: list[str]) -> str:
    """Extract a representative label from a cluster of queries."""
    # Use the most common non-stop words across all queries
    stop_words = {
        "what", "is", "the", "how", "do", "does", "can", "i", "my", "a", "an",
        "for", "to", "in", "of", "and", "or", "are", "we", "our", "get", "about",
        "much", "many", "long", "when", "where", "who", "which", "that", "this",
        "with", "have", "has", "it", "be", "on", "at", "by", "if", "me", "there",
    }
    words: Counter = Counter()
    for q in queries:
        for w in q.lower().split():
            w = w.strip("?.,!")
            if w not in stop_words and len(w) > 2:
                words[w] += 1

    top = words.most_common(3)
    return " ".join(w for w, _ in top) if top else queries[0][:50]


def _keyword_fallback(queries: list[str], max_topics: int) -> list[dict]:
    """Fallback keyword-based clustering when embeddings unavailable."""
    stop_words = {
        "what", "is", "the", "how", "do", "does", "can", "i", "my", "a", "an",
        "for", "to", "in", "of", "and", "or", "are", "we", "our", "get", "about",
    }
    keyword_freq: Counter = Counter()
    keyword_queries: dict[str, list[str]] = {}

    for q in queries:
        for w in q.lower().split():
            w = w.strip("?.,!")
            if w not in stop_words and len(w) > 2:
                keyword_freq[w] += 1
                keyword_queries.setdefault(w, []).append(q)

    topics = []
    for keyword, count in keyword_freq.most_common(max_topics):
        if count < 2:
            break
        topics.append({"topic": keyword, "count": count, "example": keyword_queries[keyword][0]})

    return topics


def generate_drift_report() -> DriftReport:
    traces = load_traces(limit=1000)
    cutoff = time.time() - (7 * 86400)

    query_volume = sum(
        1 for t in traces
        if _parse_time(t.get("timestamp", "")) and _parse_time(t["timestamp"]) >= cutoff
    )

    confidences = []
    for t in traces:
        trace_time = _parse_time(t.get("timestamp", ""))
        if trace_time and trace_time >= cutoff:
            top_scores = t.get("retrieval", {}).get("reranked_top_scores", [])
            if top_scores:
                confidences.append(max(top_scores))

    avg_conf = round(float(np.mean(confidences)), 4) if confidences else None

    return DriftReport(
        embedding_drift=None,
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
