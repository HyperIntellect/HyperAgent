"""Scratchpad storage service for context offloading."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

SCRATCHPAD_MEMORY_TYPE = "procedural"
SCRATCHPAD_NAMESPACE = "scratchpad"
SCRATCHPAD_MAX_READ_CHARS = 4000
SCRATCHPAD_SESSION_TTL_SECONDS = 3600  # 1 hour
SCRATCHPAD_MAX_SESSION_ENTRIES = 500


@dataclass
class ScratchpadPayload:
    """Scratchpad read payload."""

    notes: str
    scope: str
    namespace: str
    updated_at: float


class ScratchpadService:
    """Stores short-term and persistent scratchpad notes."""

    def __init__(self) -> None:
        # Session-scope notes: (user_id, task_id, namespace) -> payload
        self._session_notes: dict[tuple[str, str, str], ScratchpadPayload] = {}
        self._persistent_fallback: dict[tuple[str, str], ScratchpadPayload] = {}
        self._lock = asyncio.Lock()

    def _effective_namespace(self, namespace: str | None, task_id: str | None) -> str:
        if namespace and namespace.strip():
            return namespace.strip()
        if task_id:
            return f"task:{task_id}"
        return "default"

    def _evict_stale_sessions(self) -> None:
        """Remove expired session entries. Must be called inside self._lock."""
        if not self._session_notes:
            return
        now = time.time()
        cutoff = now - SCRATCHPAD_SESSION_TTL_SECONDS
        stale_keys = [
            k for k, v in self._session_notes.items() if v.updated_at < cutoff
        ]
        for k in stale_keys:
            del self._session_notes[k]
        # Hard cap on remaining entries
        if len(self._session_notes) > SCRATCHPAD_MAX_SESSION_ENTRIES:
            sorted_keys = sorted(
                self._session_notes, key=lambda k: self._session_notes[k].updated_at,
            )
            for k in sorted_keys[: len(self._session_notes) - SCRATCHPAD_MAX_SESSION_ENTRIES]:
                del self._session_notes[k]

    def _clip_notes(self, notes: str, max_chars: int = SCRATCHPAD_MAX_READ_CHARS) -> str:
        if len(notes) <= max_chars:
            return notes
        clipped = notes[:max_chars]
        return f"{clipped}\n...[truncated]"

    async def write(
        self,
        *,
        notes: str,
        user_id: str | None,
        task_id: str | None,
        scope: str,
        namespace: str | None = None,
    ) -> ScratchpadPayload:
        """Write notes to session or persistent scratchpad storage."""
        normalized_scope = "persistent" if scope == "persistent" else "session"
        effective_namespace = self._effective_namespace(namespace, task_id)
        entry = ScratchpadPayload(
            notes=self._clip_notes(notes, max_chars=12000),
            scope=normalized_scope,
            namespace=effective_namespace,
            updated_at=time.time(),
        )

        if normalized_scope == "session":
            if not user_id:
                return entry
            async with self._lock:
                self._evict_stale_sessions()
                self._session_notes[(user_id, task_id or "", effective_namespace)] = entry
            return entry

        # persistent scope
        if not settings.context_offloading_persistent_enabled or not user_id:
            async with self._lock:
                self._persistent_fallback[(user_id or "anonymous", effective_namespace)] = entry
            return entry

        try:
            from sqlalchemy import select

            from app.db.base import async_session_maker
            from app.db.models import Memory

            async with async_session_maker() as session:
                stmt = (
                    select(Memory)
                    .where(Memory.user_id == user_id)
                    .where(Memory.memory_type == SCRATCHPAD_MEMORY_TYPE)
                    .where(Memory.metadata_json.contains(f'"key": "{effective_namespace}"'))
                    .where(Memory.metadata_json.contains(f'"namespace": "{SCRATCHPAD_NAMESPACE}"'))
                )
                result = await session.execute(stmt)
                target = result.scalar_one_or_none()

                metadata = {
                    "namespace": SCRATCHPAD_NAMESPACE,
                    "scratchpad_scope": "persistent",
                    "key": effective_namespace,
                    "task_id": task_id,
                }
                if target is None:
                    target = Memory(
                        id=str(uuid.uuid4()),
                        user_id=user_id,
                        memory_type=SCRATCHPAD_MEMORY_TYPE,
                        content=entry.notes,
                        metadata_json=json.dumps(metadata),
                        source_conversation_id=task_id,
                    )
                    session.add(target)
                else:
                    target.content = entry.notes
                    target.metadata_json = json.dumps(metadata)
                    target.source_conversation_id = task_id

                await session.commit()
            return entry
        except Exception as e:
            logger.warning("scratchpad_persistent_write_fallback", error=str(e))
            async with self._lock:
                self._persistent_fallback[(user_id, effective_namespace)] = entry
            return entry

    async def read(
        self,
        *,
        user_id: str | None,
        task_id: str | None,
        scope: str,
        namespace: str | None = None,
    ) -> ScratchpadPayload | None:
        """Read notes from session or persistent scratchpad storage."""
        normalized_scope = "persistent" if scope == "persistent" else "session"
        effective_namespace = self._effective_namespace(namespace, task_id)

        if normalized_scope == "session":
            if not user_id:
                return None
            async with self._lock:
                return self._session_notes.get((user_id, task_id or "", effective_namespace))

        if not user_id:
            return None

        if settings.context_offloading_persistent_enabled:
            try:
                from sqlalchemy import select

                from app.db.base import async_session_maker
                from app.db.models import Memory

                async with async_session_maker() as session:
                    ns_filter = f'"namespace": "{SCRATCHPAD_NAMESPACE}"'
                    key_filter = f'"key": "{effective_namespace}"'
                    stmt = (
                        select(Memory)
                        .where(Memory.user_id == user_id)
                        .where(Memory.memory_type == SCRATCHPAD_MEMORY_TYPE)
                        .where(Memory.metadata_json.contains(key_filter))
                        .where(Memory.metadata_json.contains(ns_filter))
                    )
                    result = await session.execute(stmt)
                    row = result.scalar_one_or_none()
                    if row:
                        return ScratchpadPayload(
                            notes=self._clip_notes(row.content),
                            scope="persistent",
                            namespace=effective_namespace,
                            updated_at=time.time(),
                        )
            except Exception as e:
                logger.warning("scratchpad_persistent_read_fallback", error=str(e))

        async with self._lock:
            return self._persistent_fallback.get((user_id, effective_namespace))

    async def get_compact_context(
        self,
        *,
        user_id: str | None,
        task_id: str | None,
        max_chars: int = 1200,
    ) -> str | None:
        """Build a compact scratchpad context snippet for prompt injection."""
        if not settings.context_offloading_enabled or not user_id:
            return None

        snippets: list[str] = []
        session_entry = await self.read(
            user_id=user_id,
            task_id=task_id,
            scope="session",
            namespace=None,
        )
        if session_entry and session_entry.notes:
            snippets.append(f"Session scratchpad:\n{session_entry.notes}")

        if settings.context_offloading_persistent_enabled:
            persistent_entry = await self.read(
                user_id=user_id,
                task_id=task_id,
                scope="persistent",
                namespace=None,
            )
            if persistent_entry and persistent_entry.notes:
                snippets.append(f"Persistent scratchpad:\n{persistent_entry.notes}")

        if not snippets:
            return None

        combined = "\n\n".join(snippets)
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n...[scratchpad context truncated]"
        return f"[Scratchpad Context]\n{combined}"


_scratchpad_service = ScratchpadService()


def get_scratchpad_service() -> ScratchpadService:
    """Get singleton scratchpad service."""
    return _scratchpad_service

