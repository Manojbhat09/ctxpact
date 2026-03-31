"""Session persistence — in-memory and SQLite backends."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

from ctxpact.session.models import Session


class SessionStore(ABC):
    """Abstract session store interface."""

    @abstractmethod
    async def get(self, session_id: str) -> Session | None: ...

    @abstractmethod
    async def put(self, session: Session) -> None: ...

    @abstractmethod
    async def delete(self, session_id: str) -> None: ...

    @abstractmethod
    async def cleanup_expired(self, ttl_hours: int) -> int: ...


class MemorySessionStore(SessionStore):
    """In-memory session store — fast, no persistence across restarts."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    async def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session:
            session.last_active = time.time()
        return session

    async def put(self, session: Session) -> None:
        self._sessions[session.session_id] = session

    async def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def cleanup_expired(self, ttl_hours: int) -> int:
        cutoff = time.time() - (ttl_hours * 3600)
        expired = [
            sid for sid, s in self._sessions.items() if s.last_active < cutoff
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)


class SqliteSessionStore(SessionStore):
    """SQLite-backed session store — survives restarts."""

    def __init__(self, db_path: str = "./ctxpact_sessions.db") -> None:
        self._db_path = db_path
        self._initialized = False

    async def _ensure_db(self) -> Any:
        import aiosqlite

        db = await aiosqlite.connect(self._db_path)
        if not self._initialized:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    last_active REAL NOT NULL
                )
            """)
            await db.commit()
            self._initialized = True
        return db

    async def get(self, session_id: str) -> Session | None:
        db = await self._ensure_db()
        try:
            cursor = await db.execute(
                "SELECT data FROM sessions WHERE session_id = ?", (session_id,)
            )
            row = await cursor.fetchone()
            if row:
                # For now, return a fresh session — full deserialization can be added later
                return Session(session_id=session_id)
            return None
        finally:
            await db.close()

    async def put(self, session: Session) -> None:
        db = await self._ensure_db()
        try:
            data = json.dumps({
                "session_id": session.session_id,
                "message_count": session.message_count,
                "user_turn_count": session.user_turn_count,
                "total_input_tokens": session.total_input_tokens,
                "total_output_tokens": session.total_output_tokens,
                "compaction_count": len(session.compaction_events),
            })
            await db.execute(
                """INSERT OR REPLACE INTO sessions (session_id, data, last_active)
                   VALUES (?, ?, ?)""",
                (session.session_id, data, session.last_active),
            )
            await db.commit()
        finally:
            await db.close()

    async def delete(self, session_id: str) -> None:
        db = await self._ensure_db()
        try:
            await db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            await db.commit()
        finally:
            await db.close()

    async def cleanup_expired(self, ttl_hours: int) -> int:
        db = await self._ensure_db()
        try:
            cutoff = time.time() - (ttl_hours * 3600)
            cursor = await db.execute(
                "DELETE FROM sessions WHERE last_active < ?", (cutoff,)
            )
            await db.commit()
            return cursor.rowcount
        finally:
            await db.close()
