"""Tests for session memory and retrieval cache."""

import time
import numpy as np

from docvault.memory.session import SessionManager, Session
from docvault.memory.cache import RetrievalCache


def test_session_management():
    mgr = SessionManager()
    session = mgr.get_or_create("sess-1")
    assert session.id == "sess-1"
    assert len(session.turns) == 0

    session.add_turn("What is PTO?", "15 days for new employees.")
    assert len(session.turns) == 1

    # Same session returned
    same = mgr.get_or_create("sess-1")
    assert len(same.turns) == 1


def test_session_max_turns():
    session = Session(id="test")
    for i in range(10):
        session.add_turn(f"Q{i}", f"A{i}")
    # Should only keep last 5 (default max_session_turns)
    assert len(session.turns) == 5
    assert session.turns[0].question == "Q5"


def test_session_history_format():
    session = Session(id="test")
    session.add_turn("What is PTO?", "15 days.")
    history = session.get_history()
    assert history == [{"question": "What is PTO?", "answer": "15 days."}]


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
