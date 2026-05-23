"""Conversation session memory — tracks recent turns for coreference resolution."""

import time
from dataclasses import dataclass, field

from docvault.config import settings


@dataclass
class Turn:
    question: str
    answer: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    id: str
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def add_turn(self, question: str, answer: str):
        self.turns.append(Turn(question=question, answer=answer))
        if len(self.turns) > settings.max_session_turns:
            self.turns = self.turns[-settings.max_session_turns:]
        self.last_active = time.time()

    def get_history(self) -> list[dict]:
        return [{"question": t.question, "answer": t.answer} for t in self.turns]

    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > settings.session_ttl_seconds


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        self._cleanup_expired()
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(id=session_id)
        return self._sessions[session_id]

    def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session and session.is_expired():
            del self._sessions[session_id]
            return None
        return session

    def _cleanup_expired(self):
        expired = [sid for sid, s in self._sessions.items() if s.is_expired()]
        for sid in expired:
            del self._sessions[sid]
