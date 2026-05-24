"""Conversation session memory — backend interface with Redis and in-memory implementations."""

import json
import time
import logging
from abc import ABC, abstractmethod

from docvault.config import settings

logger = logging.getLogger(__name__)


# ── Interface ────────────────────────────────────────────

class SessionBackend(ABC):
    @abstractmethod
    def get_or_create(self, session_id: str, user_id: str | None = None) -> str: ...

    @abstractmethod
    def add_turn(self, session_id: str, question: str, answer: str,
                 citations: list[dict] | None = None, confidence: str = "unknown",
                 trace_id: str | None = None) -> None: ...

    @abstractmethod
    def get_history(self, session_id: str, limit: int = 5) -> list[dict]: ...

    @abstractmethod
    def get_all_questions(self, session_id: str) -> list[str]: ...

    @abstractmethod
    def cleanup_expired(self) -> int: ...

    @abstractmethod
    def session_count(self) -> int: ...


# ── Redis Backend ────────────────────────────────────────

class RedisSessionBackend(SessionBackend):
    def __init__(self, redis_client):
        self._r = redis_client

    def _sk(self, sid: str) -> str:
        return f"docvault:session:{sid}"

    def _tk(self, sid: str) -> str:
        return f"docvault:turns:{sid}"

    def get_or_create(self, session_id: str, user_id: str | None = None) -> str:
        key = self._sk(session_id)
        now = str(time.time())
        if not self._r.exists(key):
            self._r.hset(key, mapping={"user_id": user_id or "", "created_at": now, "last_active": now})
        else:
            self._r.hset(key, "last_active", now)
        self._r.expire(key, settings.session_ttl_seconds)
        self._r.expire(self._tk(session_id), settings.session_ttl_seconds)
        return session_id

    def add_turn(self, session_id: str, question: str, answer: str,
                 citations: list[dict] | None = None, confidence: str = "unknown",
                 trace_id: str | None = None) -> None:
        turn = {"question": question, "answer": answer, "citations": citations or [],
                "confidence": confidence, "trace_id": trace_id, "timestamp": time.time()}
        tk = self._tk(session_id)
        self._r.rpush(tk, json.dumps(turn))
        self._r.ltrim(tk, -settings.max_session_turns, -1)
        self._r.hset(self._sk(session_id), "last_active", str(time.time()))
        self._r.expire(self._sk(session_id), settings.session_ttl_seconds)
        self._r.expire(tk, settings.session_ttl_seconds)

    def get_history(self, session_id: str, limit: int = 5) -> list[dict]:
        raw = self._r.lrange(self._tk(session_id), -limit, -1)
        return [
            {"question": t["question"], "answer": t["answer"],
             "citations": t.get("citations", []), "confidence": t.get("confidence", "unknown")}
            for t in (json.loads(r) for r in raw)
        ]

    def get_all_questions(self, session_id: str) -> list[str]:
        return [json.loads(r)["question"] for r in self._r.lrange(self._tk(session_id), 0, -1)]

    def cleanup_expired(self) -> int:
        return 0  # Redis TTL handles this

    def session_count(self) -> int:
        return len(self._r.keys("docvault:session:*"))


# ── In-Memory Backend ────────────────────────────────────

class MemorySessionBackend(SessionBackend):
    def __init__(self):
        self._sessions: dict[str, dict] = {}

    def get_or_create(self, session_id: str, user_id: str | None = None) -> str:
        now = time.time()
        if session_id not in self._sessions:
            self._sessions[session_id] = {"turns": [], "created_at": now, "last_active": now}
        else:
            self._sessions[session_id]["last_active"] = now
        return session_id

    def add_turn(self, session_id: str, question: str, answer: str,
                 citations: list[dict] | None = None, confidence: str = "unknown",
                 trace_id: str | None = None) -> None:
        if session_id not in self._sessions:
            self.get_or_create(session_id)
        turn = {"question": question, "answer": answer, "citations": citations or [],
                "confidence": confidence, "trace_id": trace_id, "timestamp": time.time()}
        turns = self._sessions[session_id]["turns"]
        turns.append(turn)
        if len(turns) > settings.max_session_turns:
            self._sessions[session_id]["turns"] = turns[-settings.max_session_turns:]
        self._sessions[session_id]["last_active"] = time.time()

    def get_history(self, session_id: str, limit: int = 5) -> list[dict]:
        turns = self._sessions.get(session_id, {}).get("turns", [])
        return [
            {"question": t["question"], "answer": t["answer"],
             "citations": t.get("citations", []), "confidence": t.get("confidence", "unknown")}
            for t in turns[-limit:]
        ]

    def get_all_questions(self, session_id: str) -> list[str]:
        return [t["question"] for t in self._sessions.get(session_id, {}).get("turns", [])]

    def cleanup_expired(self) -> int:
        cutoff = time.time() - settings.session_ttl_seconds
        expired = [sid for sid, s in self._sessions.items() if s["last_active"] < cutoff]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    def session_count(self) -> int:
        return len(self._sessions)


# ── Public API ───────────────────────────────────────────

class SessionStore:
    """Session store with automatic backend selection."""

    def __init__(self):
        self._backend = self._select_backend()
        backend_name = "redis" if isinstance(self._backend, RedisSessionBackend) else "memory"
        logger.info(f"Session store: {backend_name}")

    @staticmethod
    def _select_backend() -> SessionBackend:
        try:
            import redis
            r = redis.from_url(settings.redis_url, decode_responses=True)
            r.ping()
            return RedisSessionBackend(r)
        except Exception:
            return MemorySessionBackend()

    def get_or_create_session(self, session_id: str, user_id: str | None = None) -> str:
        return self._backend.get_or_create(session_id, user_id)

    def add_turn(self, session_id: str, question: str, answer: str,
                 citations: list[dict] | None = None, confidence: str = "unknown",
                 trace_id: str | None = None):
        self._backend.add_turn(session_id, question, answer, citations, confidence, trace_id)

    def get_history(self, session_id: str, limit: int | None = None) -> list[dict]:
        return self._backend.get_history(session_id, limit or settings.max_session_turns)

    def get_all_session_questions(self, session_id: str) -> list[str]:
        return self._backend.get_all_questions(session_id)

    def cleanup_expired(self) -> int:
        return self._backend.cleanup_expired()

    def session_count(self) -> int:
        return self._backend.session_count()

    def stats(self) -> dict:
        backend_name = "redis" if isinstance(self._backend, RedisSessionBackend) else "memory"
        return {"backend": backend_name, "active_sessions": self.session_count()}
