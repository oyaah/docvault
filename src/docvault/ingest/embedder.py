"""Embedding engine — OpenAI API (primary) or local sentence-transformers (fallback).

Production default: OpenAI text-embedding-3-small via API.
  - No torch/GPU needed in container
  - Docker image drops from 6GB to ~800MB
  - ~100ms latency per batch, $0.02/1M tokens

Local fallback: Set DOCVAULT_EMBEDDING_BACKEND=local
  - Requires torch + sentence-transformers (install with `pip install docvault[local]`)
  - BGE-M3 model loaded from disk/volume mount
"""

import logging

import numpy as np

from docvault.config import settings

logger = logging.getLogger(__name__)

_cache: dict[str, np.ndarray] = {}  # chunk_hash -> embedding


def embed_texts(texts: list[str], chunk_hashes: list[str] | None = None) -> np.ndarray:
    """Embed texts using configured backend, with hash-based caching for ingestion."""
    if chunk_hashes is None:
        return _get_backend().embed(texts)

    embeddings = [None] * len(texts)
    texts_to_embed = []
    indices_to_embed = []

    for i, h in enumerate(chunk_hashes):
        if h in _cache:
            embeddings[i] = _cache[h]
        else:
            texts_to_embed.append(texts[i])
            indices_to_embed.append(i)

    if texts_to_embed:
        new_embeddings = _get_backend().embed(texts_to_embed)
        for j, idx in enumerate(indices_to_embed):
            emb = new_embeddings[j]
            embeddings[idx] = emb
            _cache[chunk_hashes[idx]] = emb

    return np.array(embeddings)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query. No caching (queries are ephemeral)."""
    return _get_backend().embed([query])[0]


def embedding_dim() -> int:
    return settings.embedding_dimensions


def clear_cache():
    _cache.clear()


def warmup():
    """Pre-warm the embedding backend (validates connectivity)."""
    _get_backend().warmup()


# ── Backend interface ───────────────────────────────────

class _EmbeddingBackend:
    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def warmup(self):
        pass


class _OpenAIBackend(_EmbeddingBackend):
    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI()
        self._model = settings.openai_embedding_model
        self._dims = settings.embedding_dimensions
        logger.info(f"Embedding backend: OpenAI {self._model} ({self._dims}d)")

    def embed(self, texts: list[str]) -> np.ndarray:
        # OpenAI API supports batches up to 2048 texts
        all_embeddings = []
        batch_size = 512
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dims,
            )
            batch_embs = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embs)

        result = np.array(all_embeddings, dtype=np.float32)
        # Normalize to unit vectors (consistent with sentence-transformers behavior)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return result / norms

    def warmup(self):
        self.embed(["warmup"])
        logger.info("OpenAI embedding backend warmed up")


class _LocalBackend(_EmbeddingBackend):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(settings.local_embedding_model)
        logger.info(f"Embedding backend: local {settings.local_embedding_model}")

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
        )

    def warmup(self):
        self.embed(["warmup"])
        logger.info("Local embedding backend warmed up")


# ── Singleton ───────────────────────────────────────────

_backend: _EmbeddingBackend | None = None


def _get_backend() -> _EmbeddingBackend:
    global _backend
    if _backend is None:
        if settings.embedding_backend == "local":
            _backend = _LocalBackend()
        else:
            _backend = _OpenAIBackend()
    return _backend
