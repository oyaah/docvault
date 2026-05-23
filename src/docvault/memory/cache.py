"""Retrieval cache — LRU cache keyed on query embedding."""

import time
import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from docvault.config import settings


@dataclass
class CacheEntry:
    results: list[dict]
    timestamp: float


class RetrievalCache:
    def __init__(self, max_size: int = 256):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size

    def _make_key(self, query_embedding: np.ndarray, top_k: int) -> str:
        # Quantize embedding to reduce key space
        quantized = (query_embedding * 1000).astype(np.int16).tobytes()
        h = hashlib.md5(quantized + str(top_k).encode()).hexdigest()
        return h

    def get(self, query_embedding: np.ndarray, top_k: int) -> list[dict] | None:
        key = self._make_key(query_embedding, top_k)
        entry = self._cache.get(key)
        if entry is None:
            return None

        # Check TTL
        if (time.time() - entry.timestamp) > settings.cache_ttl_seconds:
            del self._cache[key]
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return entry.results

    def put(self, query_embedding: np.ndarray, top_k: int, results: list[dict]):
        key = self._make_key(query_embedding, top_k)
        self._cache[key] = CacheEntry(results=results, timestamp=time.time())
        self._cache.move_to_end(key)

        # Evict oldest if over capacity
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def invalidate(self):
        """Clear entire cache (e.g., after document re-ingest)."""
        self._cache.clear()

    def stats(self) -> dict:
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
        }
