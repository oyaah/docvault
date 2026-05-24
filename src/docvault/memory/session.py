"""Conversation session memory — Redis-backed with SQLite fallback."""

import json
import time
import logging
from dataclasses import dataclass, field

from docvault.config import settings

logger = logging.getLogger(__name__)


def _get_redis():
    """Lazy Redis connection. Returns None if Redis unavailable."""
    try:
        import redis
        r = redis.from_url(settings.redis_url, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


class SessionStore:
    """Redis-backed session storage. Falls back to in-memory if Redis unavailable."""

    def __init__(self):
        self._redis = _get_redis()
        self._fallback: dict[str, dict] = {}  # session_id -> {turns, created_at, last_active}
        if self._redis:
            logger.info("Session store: Redis connected")
        else:
            logger.warning("Session store: Redis unavailable, using in-memory fallback")

    def _session_key(self, session_id: str) -> str:
        return f"docvault:session:{session_id}"

    def _turns_key(self, session_id: str) -> str:
        return f"docvault:turns:{session_id}"

    def get_or_create_session(self, session_id: str, user_id: str | None = None) -> str:
        """Ensure session exists. Returns session_id."""
        if self._redis:
            key = self._session_key(session_id)
            if not self._redis.exists(key):
                self._redis.hset(key, mapping={
                    "user_id": user_id or "",
                    "created_at": str(time.time()),
                    "last_active": str(time.time()),
                })
            else:
                self._redis.hset(key, "last_active", str(time.time()))
            self._redis.expire(key, settings.session_ttl_seconds)
            self._redis.expire(self._turns_key(session_id), settings.session_ttl_seconds)
        else:
            if session_id not in self._fallback:
                self._fallback[session_id] = {
                    "turns": [],
                    "created_at": time.time(),
                    "last_active": time.time(),
                }
            else:
                self._fallback[session_id]["last_active"] = time.time()
        return session_id

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        citations: list[dict] | None = None,
        confidence: str = "unknown",
        trace_id: str | None = None,
    ):
        """Record a conversation turn."""
        turn = {
            "question": question,
            "answer": answer,
            "citations": citations or [],
            "confidence": confidence,
            "trace_id": trace_id,
            "timestamp": time.time(),
        }

        if self._redis:
            self._redis.rpush(self._turns_key(session_id), json.dumps(turn))
            # Trim to max turns
            self._redis.ltrim(self._turns_key(session_id), -settings.max_session_turns, -1)
            self._redis.hset(self._session_key(session_id), "last_active", str(time.time()))
            # Refresh TTL
            self._redis.expire(self._session_key(session_id), settings.session_ttl_seconds)
            self._redis.expire(self._turns_key(session_id), settings.session_ttl_seconds)
        else:
            if session_id not in self._fallback:
                self.get_or_create_session(session_id)
            turns = self._fallback[session_id]["turns"]
            turns.append(turn)
            if len(turns) > settings.max_session_turns:
                self._fallback[session_id]["turns"] = turns[-settings.max_session_turns:]
            self._fallback[session_id]["last_active"] = time.time()

    def get_history(self, session_id: str, limit: int | None = None) -> list[dict]:
        """Get recent turns for a session."""
        k = limit or settings.max_session_turns

        if self._redis:
            raw = self._redis.lrange(self._turns_key(session_id), -k, -1)
            return [
                {
                    "question": t["question"],
                    "answer": t["answer"],
                    "citations": t.get("citations", []),
                    "confidence": t.get("confidence", "unknown"),
                }
                for t in (json.loads(r) for r in raw)
            ]
        else:
            turns = self._fallback.get(session_id, {}).get("turns", [])
            return [
                {
                    "question": t["question"],
                    "answer": t["answer"],
                    "citations": t.get("citations", []),
                    "confidence": t.get("confidence", "unknown"),
                }
                for t in turns[-k:]
            ]

    def get_all_session_questions(self, session_id: str) -> list[str]:
        """Get all questions from a session."""
        if self._redis:
            raw = self._redis.lrange(self._turns_key(session_id), 0, -1)
            return [json.loads(r)["question"] for r in raw]
        else:
            turns = self._fallback.get(session_id, {}).get("turns", [])
            return [t["question"] for t in turns]

    def cleanup_expired(self) -> int:
        """Remove expired sessions (only needed for in-memory fallback; Redis handles TTL)."""
        if self._redis:
            return 0
        cutoff = time.time() - settings.session_ttl_seconds
        expired = [sid for sid, s in self._fallback.items() if s["last_active"] < cutoff]
        for sid in expired:
            del self._fallback[sid]
        return len(expired)

    def session_count(self) -> int:
        if self._redis:
            keys = self._redis.keys("docvault:session:*")
            return len(keys)
        return len(self._fallback)

    def stats(self) -> dict:
        return {
            "backend": "redis" if self._redis else "memory",
            "active_sessions": self.session_count(),
        }
