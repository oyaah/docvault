"""Prometheus metrics for DocVault."""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


# Counters
QUERY_TOTAL = Counter("docvault_query_total", "Total queries processed")
QUERY_ERRORS = Counter("docvault_query_errors_total", "Total query errors")
INGEST_TOTAL = Counter("docvault_ingest_total", "Total documents ingested")
CACHE_HITS = Counter("docvault_cache_hits_total", "Retrieval cache hits")
CACHE_MISSES = Counter("docvault_cache_misses_total", "Retrieval cache misses")

# Histograms
QUERY_LATENCY = Histogram(
    "docvault_query_latency_seconds",
    "Query latency by stage",
    labelnames=["stage"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
NLI_SCORES = Histogram(
    "docvault_nli_score",
    "NLI verification scores",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
CONFIDENCE_DIST = Histogram(
    "docvault_confidence_score",
    "Reranker confidence scores",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# Gauges
ACTIVE_DOCUMENTS = Gauge("docvault_active_documents", "Number of active documents")
ACTIVE_CHUNKS = Gauge("docvault_active_chunks", "Number of active chunks")
HALLUCINATION_RATE = Gauge(
    "docvault_hallucination_rate", "Rolling hallucination rate (claims stripped / total)"
)


def get_metrics_text() -> tuple[str, str]:
    """Return (content_type, metrics_text) for Prometheus scraping."""
    return CONTENT_TYPE_LATEST, generate_latest().decode("utf-8")
