"""Per-query trace logging — append-only JSONL."""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict

from docvault.config import settings


@dataclass
class RetrievalTrace:
    bm25_results: int = 0
    dense_results: int = 0
    fused_results: int = 0
    reranked_top_scores: list[float] = field(default_factory=list)
    cache_hit: bool = False
    latency_ms: float = 0


@dataclass
class GenerationTrace:
    model: str = ""
    context_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0


@dataclass
class VerificationTrace:
    claims_total: int = 0
    claims_verified: int = 0
    claims_stripped: int = 0
    nli_scores: list[float] = field(default_factory=list)


@dataclass
class QueryTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = ""
    query: str = ""
    expanded_queries: list[str] = field(default_factory=list)
    session_id: str | None = None
    retrieval: RetrievalTrace = field(default_factory=RetrievalTrace)
    generation: GenerationTrace = field(default_factory=GenerationTrace)
    verification: VerificationTrace = field(default_factory=VerificationTrace)
    confidence: str = "unknown"
    total_latency_ms: float = 0

    def save(self):
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        log_path = settings.trace_log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(asdict(self)) + "\n")


def load_traces(limit: int = 100) -> list[dict]:
    """Load recent traces from the log file."""
    log_path = settings.trace_log_path
    if not log_path.exists():
        return []

    lines = log_path.read_text().strip().split("\n")
    traces = []
    for line in lines[-limit:]:
        try:
            traces.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return traces
