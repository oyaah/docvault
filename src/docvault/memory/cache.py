"""Retrieval cache — backend interface with Redis and in-memory implementations."""

import json
import time
import hashlib
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np

from docvault.config import settings

logger = logging.getLogger(__name__)

CACHE_PREFIX = "docvault:cache:"


class _JSONEncoder(json.JSONEncoder):
    """Handles numpy types and other non-standard JSON objects."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return super().default(obj)


def _serialize(results: list[dict]) -> str:
    return json.dumps(results, cls=_JSONEncoder)


def _make_key(query_embedding: np.ndarray, top_k: int) -> str:
    quantized = (query_embedding * 1000).astype(np.int16).tobytes()
    return hashlib.md5(quantized + str(top_k).encode()).hexdigest()


# ── Interface ────────────────────────────────────────────

class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> list[dict] | None: ...

    @abstractmethod
    def put(self, key: str, results: list[dict], ttl: int) -> None: ...

    @abstractmethod
    def invalidate(self) -> None: ...

    @abstractmethod
    def stats(self) -> dict: ...


# ── Redis Backend ────────────────────────────────────────

class RedisCacheBackend(CacheBackend):
    def __init__(self, redis_client):
        self._r = redis_client

    def get(self, key: str) -> list[dict] | None:
        raw = self._r.get(CACHE_PREFIX + key)
        return json.loads(raw) if raw else None

    def put(self, key: str, results: list[dict], ttl: int) -> None:
        self._r.setex(CACHE_PREFIX + key, ttl, _serialize(results))

    def invalidate(self) -> None:
        keys = self._r.keys(CACHE_PREFIX + "*")
        if keys:
            self._r.delete(*keys)

    def stats(self) -> dict:
        return {"backend": "redis", "size": len(self._r.keys(CACHE_PREFIX + "*"))}


# ── In-Memory Backend ────────────────────────────────────

class MemoryCacheBackend(CacheBackend):
    def __init__(self, max_size: int = 256):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> list[dict] | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if (time.time() - entry["timestamp"]) > settings.cache_ttl_seconds:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return entry["results"]

    def put(self, key: str, results: list[dict], ttl: int) -> None:
        # Re-serialize through JSON to strip non-serializable types consistently
        clean = json.loads(_serialize(results))
        self._cache[key] = {"results": clean, "timestamp": time.time()}
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def invalidate(self) -> None:
        self._cache.clear()

    def stats(self) -> dict:
        return {"backend": "memory", "size": len(self._cache), "max_size": self._max_size}


# ── Public API ───────────────────────────────────────────

class RetrievalCache:
    """Retrieval cache with automatic backend selection."""

    def __init__(self, max_size: int = 256):
        self._backend = self._select_backend(max_size)
        logger.info(f"Retrieval cache: {self._backend.stats()['backend']}")

    @staticmethod
    def _select_backend(max_size: int) -> CacheBackend:
        try:
            import redis
            r = redis.from_url(settings.redis_url, decode_responses=True)
            r.ping()
            return RedisCacheBackend(r)
        except Exception:
            return MemoryCacheBackend(max_size)

    def get(self, query_embedding: np.ndarray, top_k: int) -> list[dict] | None:
        return self._backend.get(_make_key(query_embedding, top_k))

    def put(self, query_embedding: np.ndarray, top_k: int, results: list[dict]):
        self._backend.put(_make_key(query_embedding, top_k), results, settings.cache_ttl_seconds)

    def invalidate(self):
        self._backend.invalidate()

    def stats(self) -> dict:
        return self._backend.stats()
