"""Cross-encoder reranker — deep query-chunk relevance scoring."""

import numpy as np
from sentence_transformers import CrossEncoder

from docvault.config import settings


_model: CrossEncoder | None = None
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(
    query: str,
    chunks: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """Rerank chunks by cross-encoder score.

    Args:
        query: user query
        chunks: list of dicts with 'content' key
        top_k: how many to return

    Returns:
        Top-k chunks with 'rerank_score' added, sorted descending.
    """
    if not chunks:
        return []

    k = top_k or settings.rerank_top_k
    model = get_model()

    pairs = [(query, chunk["content"]) for chunk in chunks]
    raw_scores = model.predict(pairs)
    # Normalize raw logits to [0,1] via sigmoid
    scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores)))

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:k]
