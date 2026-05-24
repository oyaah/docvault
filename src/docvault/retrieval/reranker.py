"""Cross-encoder reranker — ONNX Runtime inference (no torch needed).

Uses pre-exported ONNX model for ms-marco-MiniLM-L-6-v2.
Export with: python scripts/export_onnx.py
"""

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from docvault.config import settings

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ONNX_FILE = "reranker.onnx"

_session: ort.InferenceSession | None = None
_tokenizer = None


def _get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        model_path = settings.onnx_models_dir / ONNX_FILE
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX reranker model not found at {model_path}. "
                f"Run: python scripts/export_onnx.py"
            )
        _session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        logger.info(f"Loaded ONNX reranker from {model_path}")
    return _session


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def rerank(
    query: str,
    chunks: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """Rerank chunks by cross-encoder score.

    Returns top-k chunks with 'rerank_score' added, sorted descending.
    """
    if not chunks:
        return []

    k = top_k or settings.rerank_top_k
    session = _get_session()
    tokenizer = _get_tokenizer()

    # Tokenize query-chunk pairs
    pairs_a = [query] * len(chunks)
    pairs_b = [chunk["content"] for chunk in chunks]
    inputs = tokenizer(
        pairs_a, pairs_b,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=512,
    )

    # Run inference — filter to only inputs the model expects
    input_names = {inp.name for inp in session.get_inputs()}
    feed = {k: v for k, v in inputs.items() if k in input_names}
    outputs = session.run(None, feed)

    # ms-marco model outputs unbounded relevance scores (not logits for sigmoid).
    # Higher = more relevant. Typical range: [-12, +12].
    scores = outputs[0].flatten()

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:k]


def warmup():
    """Pre-load model and tokenizer."""
    _get_session()
    _get_tokenizer()
    logger.info("ONNX reranker warmed up")
