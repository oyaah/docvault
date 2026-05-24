"""Retrieval cache — Redis-backed with in-memory LRU fallback."""

import json
import time
import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from docvault.config import settings

logger = logging.getLogger(__name__)

CACHE_PREFIX = "docvault:cache:"


def _get_redis():
    try:
        import redis
        r = redis.from_url(settings.redis_url, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


class RetrievalCache:
    """Redis-backed retrieval cache. Falls back to in-memory LRU if Redis unavailable."""

    def __init__(self, max_size: int = 256):
        self._redis = _get_redis()
        # In-memory fallback
        self._mem_cache: OrderedDict[str, dict] = OrderedDict()
        self._max_size = max_size

        if self._redis:
            logger.info("Retrieval cache: Redis connected")
        else:
            logger.warning("Retrieval cache: Redis unavailable, using in-memory LRU")

    def _make_key(self, query_embedding: np.ndarray, top_k: int) -> str:
        quantized = (query_embedding * 1000).astype(np.int16).tobytes()
        h = hashlib.md5(quantized + str(top_k).encode()).hexdigest()
        return h

    def get(self, query_embedding: np.ndarray, top_k: int) -> list[dict] | None:
        key = self._make_key(query_embedding, top_k)

        if self._redis:
            raw = self._redis.get(CACHE_PREFIX + key)
            if raw is None:
                return None
            return json.loads(raw)
        else:
            entry = self._mem_cache.get(key)
            if entry is None:
                return None
            if (time.time() - entry["timestamp"]) > settings.cache_ttl_seconds:
                del self._mem_cache[key]
                return None
            self._mem_cache.move_to_end(key)
            return entry["results"]

    def put(self, query_embedding: np.ndarray, top_k: int, results: list[dict]):
        key = self._make_key(query_embedding, top_k)

        # Serialize results — strip non-serializable fields
        serializable = []
        for r in results:
            clean = {}
            for k, v in r.items():
                if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    clean[k] = v
            serializable.append(clean)

        if self._redis:
            self._redis.setex(
                CACHE_PREFIX + key,
                settings.cache_ttl_seconds,
                json.dumps(serializable),
            )
        else:
            self._mem_cache[key] = {"results": serializable, "timestamp": time.time()}
            self._mem_cache.move_to_end(key)
            while len(self._mem_cache) > self._max_size:
                self._mem_cache.popitem(last=False)

    def invalidate(self):
        """Clear entire cache."""
        if self._redis:
            keys = self._redis.keys(CACHE_PREFIX + "*")
            if keys:
                self._redis.delete(*keys)
        else:
            self._mem_cache.clear()

    def stats(self) -> dict:
        if self._redis:
            keys = self._redis.keys(CACHE_PREFIX + "*")
            return {"backend": "redis", "size": len(keys)}
        return {"backend": "memory", "size": len(self._mem_cache), "max_size": self._max_size}
