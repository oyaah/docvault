"""Tests for session memory and retrieval cache."""

import time
import numpy as np

from docvault.memory.session import SessionStore
from docvault.memory.cache import RetrievalCache


def test_session_management():
    store = SessionStore()
    store.get_or_create_session("sess-1")
    store.add_turn("sess-1", "What is PTO?", "15 days for new employees.")
    history = store.get_history("sess-1")
    assert len(history) == 1
    assert history[0]["question"] == "What is PTO?"


def test_session_max_turns():
    store = SessionStore()
    store.get_or_create_session("sess-max")
    for i in range(10):
        store.add_turn("sess-max", f"Q{i}", f"A{i}")
    history = store.get_history("sess-max")
    # Should only keep last max_session_turns (default 5)
    assert len(history) == 5
    assert history[0]["question"] == "Q5"


def test_session_history_format():
    store = SessionStore()
    store.get_or_create_session("sess-fmt")
    store.add_turn("sess-fmt", "What is PTO?", "15 days.")
    history = store.get_history("sess-fmt")
    assert history[0]["question"] == "What is PTO?"
    assert history[0]["answer"] == "15 days."


def test_cache_put_get():
    cache = RetrievalCache(max_size=10)
    emb = np.random.rand(128).astype(np.float32)
    results = [{"chunk_id": "c1", "score": 0.9}]

    cache.put(emb, 5, results)
    cached = cache.get(emb, 5)
    assert cached is not None
    assert cached[0]["chunk_id"] == "c1"


def test_cache_miss():
    cache = RetrievalCache()
    emb = np.random.rand(128).astype(np.float32)
    assert cache.get(emb, 5) is None


def test_cache_invalidate():
    cache = RetrievalCache()
    emb = np.random.rand(128).astype(np.float32)
    cache.put(emb, 5, [{"chunk_id": "c1"}])
    cache.invalidate()
    assert cache.get(emb, 5) is None


def test_cache_eviction():
    cache = RetrievalCache(max_size=2)
    for i in range(3):
        emb = np.random.rand(128).astype(np.float32)
        cache.put(emb, 5, [{"i": i}])
    assert cache.stats()["size"] == 2
