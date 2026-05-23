"""Embedding engine — BGE-M3 with caching by chunk hash."""

import numpy as np
from sentence_transformers import SentenceTransformer

from docvault.config import settings


_model: SentenceTransformer | None = None
_cache: dict[str, np.ndarray] = {}  # chunk_hash -> embedding


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str], chunk_hashes: list[str] | None = None) -> np.ndarray:
    """Embed texts, using cache where possible.

    Args:
        texts: texts to embed
        chunk_hashes: if provided, cache lookup/store by hash

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    model = get_model()

    if chunk_hashes is None:
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

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
        new_embeddings = model.encode(
            texts_to_embed, normalize_embeddings=True, show_progress_bar=len(texts_to_embed) > 10
        )
        for j, idx in enumerate(indices_to_embed):
            emb = new_embeddings[j]
            embeddings[idx] = emb
            _cache[chunk_hashes[idx]] = emb

    return np.array(embeddings)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query. No caching (queries are ephemeral)."""
    model = get_model()
    return model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]


def embedding_dim() -> int:
    return get_model().get_sentence_embedding_dimension()


def clear_cache():
    _cache.clear()
