"""Persistent cross-session memory service.

Extracts and stores user preferences, facts, and learned patterns
from conversations for reuse in future sessions.

Supports PostgreSQL persistence with an in-memory fallback for
testing and development without a database.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Memory types
# ---------------------------------------------------------------------------


class MemoryType(str, Enum):
    """Types of memories the system can store."""

    PREFERENCE = "preference"  # User preferences (language, style, tools)
    FACT = "fact"  # Facts about the user (role, company, expertise)
    EPISODIC = "episodic"  # Notable past interactions / outcomes
    PROCEDURAL = "procedural"  # Learned procedures / workflows


ALL_MEMORY_TYPES = [t.value for t in MemoryType]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """A single memory entry."""

    id: str = ""
    user_id: str = ""
    memory_type: str = "fact"
    content: str = ""
    metadata: dict = field(default_factory=dict)
    source_conversation_id: str | None = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0


# ---------------------------------------------------------------------------
# In-memory fallback store
# ---------------------------------------------------------------------------


class InMemoryStore:
    """Pure in-memory store used when the database is unavailable."""

    def __init__(self):
        self._memories: dict[str, list[MemoryEntry]] = {}

    def add_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        metadata: dict | None = None,
        source_conversation_id: str | None = None,
    ) -> MemoryEntry:
        if user_id not in self._memories:
            self._memories[user_id] = []

        for existing in self._memories[user_id]:
            if existing.content.lower() == content.lower():
                existing.last_accessed = time.time()
                existing.access_count += 1
                return existing

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
            source_conversation_id=source_conversation_id,
        )
        self._memories[user_id].append(entry)
        return entry

    def get_memories(
        self, user_id: str, memory_type: str | None = None, limit: int = 20
    ) -> list[MemoryEntry]:
        entries = self._memories.get(user_id, [])
        if memory_type:
            entries = [e for e in entries if e.memory_type == memory_type]
        sorted_entries = sorted(
            entries,
            key=lambda e: (e.last_accessed, e.access_count),
            reverse=True,
        )
        for e in sorted_entries[:limit]:
            e.last_accessed = time.time()
            e.access_count += 1
        return sorted_entries[:limit]

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        entries = self._memories.get(user_id, [])
        original_len = len(entries)
        self._memories[user_id] = [e for e in entries if e.id != memory_id]
        return len(self._memories[user_id]) < original_len

    def clear_memories(self, user_id: str) -> None:
        self._memories.pop(user_id, None)


# ---------------------------------------------------------------------------
# Persistent (PostgreSQL) memory store
# ---------------------------------------------------------------------------


class PersistentMemoryStore:
    """Database-backed store with in-memory fallback.

    Attempts to use PostgreSQL via async sessions. If the database is
    unavailable (connection errors, missing tables, etc.) it falls back
    to a pure in-memory store transparently.
    """

    def __init__(self):
        self._fallback = InMemoryStore()
        self._use_db: bool | None = None  # None = not yet probed

    # -- helpers --------------------------------------------------------------

    async def _get_session(self):
        """Return a new async DB session, or None on import / connection error."""
        try:
            from app.db.base import async_session_maker

            return async_session_maker()
        except Exception:
            return None

    def _row_to_entry(self, row) -> MemoryEntry:
        metadata = {}
        if row.metadata_json:
            try:
                metadata = json.loads(row.metadata_json)
            except (json.JSONDecodeError, TypeError):
                pass
        return MemoryEntry(
            id=row.id,
            user_id=row.user_id,
            memory_type=row.memory_type,
            content=row.content,
            metadata=metadata,
            source_conversation_id=row.source_conversation_id,
            created_at=row.created_at.timestamp() if row.created_at else time.time(),
            last_accessed=row.last_accessed.timestamp() if row.last_accessed else time.time(),
            access_count=row.access_count or 0,
        )

    # -- public API -----------------------------------------------------------

    def add_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        metadata: dict | None = None,
        source_conversation_id: str | None = None,
    ) -> MemoryEntry:
        """Synchronous add - delegates to in-memory fallback.

        For DB persistence, use ``add_memory_async``.
        """
        return self._fallback.add_memory(
            user_id, memory_type, content, metadata, source_conversation_id
        )

    async def add_memory_async(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        metadata: dict | None = None,
        source_conversation_id: str | None = None,
    ) -> MemoryEntry:
        """Add a memory, persisting to DB if available."""
        session = await self._get_session()
        if session is None:
            return self._fallback.add_memory(
                user_id, memory_type, content, metadata, source_conversation_id
            )

        try:
            from datetime import datetime, timezone

            from sqlalchemy import select

            from app.db.models import Memory

            async with session:
                # Deduplication: look for identical content (case-insensitive)
                from sqlalchemy import func

                stmt = (
                    select(Memory)
                    .where(Memory.user_id == user_id)
                    .where(func.lower(Memory.content) == content.lower())
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    existing.last_accessed = datetime.now(timezone.utc)
                    existing.access_count = (existing.access_count or 0) + 1
                    await session.commit()
                    logger.info("memory_deduped", user_id=user_id, memory_id=existing.id)
                    return self._row_to_entry(existing)

                new_memory = Memory(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    memory_type=memory_type,
                    content=content,
                    metadata_json=json.dumps(metadata or {}),
                    source_conversation_id=source_conversation_id,
                )
                session.add(new_memory)
                await session.commit()
                await session.refresh(new_memory)
                logger.info("memory_added", user_id=user_id, memory_type=memory_type)
                return self._row_to_entry(new_memory)

        except Exception as e:
            logger.warning("memory_add_db_failed_using_fallback", error=str(e))
            return self._fallback.add_memory(
                user_id, memory_type, content, metadata, source_conversation_id
            )

    def get_memories(
        self, user_id: str, memory_type: str | None = None, limit: int = 20
    ) -> list[MemoryEntry]:
        """Synchronous get - uses in-memory fallback."""
        return self._fallback.get_memories(user_id, memory_type=memory_type, limit=limit)

    async def get_memories_async(
        self, user_id: str, memory_type: str | None = None, limit: int = 20
    ) -> list[MemoryEntry]:
        """Get memories from DB, falling back to in-memory."""
        session = await self._get_session()
        if session is None:
            return self._fallback.get_memories(user_id, memory_type=memory_type, limit=limit)

        try:
            from datetime import datetime, timezone

            from sqlalchemy import select

            from app.db.models import Memory

            async with session:
                stmt = (
                    select(Memory)
                    .where(Memory.user_id == user_id)
                )
                if memory_type:
                    stmt = stmt.where(Memory.memory_type == memory_type)
                stmt = stmt.order_by(
                    Memory.last_accessed.desc(),
                    Memory.access_count.desc(),
                ).limit(limit)

                result = await session.execute(stmt)
                rows = result.scalars().all()

                # Update last_accessed for returned memories
                now = datetime.now(timezone.utc)
                for row in rows:
                    row.last_accessed = now
                    row.access_count = (row.access_count or 0) + 1
                await session.commit()

                return [self._row_to_entry(r) for r in rows]

        except Exception as e:
            logger.warning("memory_get_db_failed_using_fallback", error=str(e))
            return self._fallback.get_memories(user_id, memory_type=memory_type, limit=limit)

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Synchronous delete - delegates to fallback."""
        return self._fallback.delete_memory(user_id, memory_id)

    async def delete_memory_async(self, user_id: str, memory_id: str) -> bool:
        """Delete a memory from DB."""
        session = await self._get_session()
        if session is None:
            return self._fallback.delete_memory(user_id, memory_id)

        try:
            from sqlalchemy import delete

            from app.db.models import Memory

            async with session:
                stmt = (
                    delete(Memory)
                    .where(Memory.id == memory_id)
                    .where(Memory.user_id == user_id)
                )
                result = await session.execute(stmt)
                await session.commit()
                deleted = result.rowcount > 0
                if deleted:
                    # Also remove from fallback if present
                    self._fallback.delete_memory(user_id, memory_id)
                return deleted

        except Exception as e:
            logger.warning("memory_delete_db_failed_using_fallback", error=str(e))
            return self._fallback.delete_memory(user_id, memory_id)

    def clear_memories(self, user_id: str) -> None:
        """Synchronous clear - delegates to fallback."""
        self._fallback.clear_memories(user_id)

    async def clear_memories_async(self, user_id: str) -> None:
        """Clear all memories for a user from DB."""
        session = await self._get_session()
        if session is None:
            self._fallback.clear_memories(user_id)
            return

        try:
            from sqlalchemy import delete

            from app.db.models import Memory

            async with session:
                stmt = delete(Memory).where(Memory.user_id == user_id)
                await session.execute(stmt)
                await session.commit()
                self._fallback.clear_memories(user_id)

        except Exception as e:
            logger.warning("memory_clear_db_failed_using_fallback", error=str(e))
            self._fallback.clear_memories(user_id)

    def format_memories_for_prompt(self, user_id: str, limit: int = 10) -> str:
        """Format user memories as a prompt section, grouped by type.

        Uses the synchronous fallback store. For DB-backed formatting,
        call ``format_memories_for_prompt_async``.
        """
        memories = self.get_memories(user_id, limit=limit)
        return _format_memories(memories)

    async def format_memories_for_prompt_async(self, user_id: str, limit: int = 10) -> str:
        """Format user memories as a prompt section, grouped by type (async/DB)."""
        memories = await self.get_memories_async(user_id, limit=limit)
        return _format_memories(memories)


# ---------------------------------------------------------------------------
# Shared formatting helper
# ---------------------------------------------------------------------------

_TYPE_CONFIG = {
    MemoryType.PREFERENCE.value: {
        "tag": "preferences",
        "guidance": "Apply these preferences to tailor your responses (language, style, tools).",
    },
    MemoryType.FACT.value: {
        "tag": "facts",
        "guidance": "Use these facts as context. Do not re-ask questions already answered here.",
    },
    MemoryType.EPISODIC.value: {
        "tag": "past_experiences",
        "guidance": "Reference relevant past experiences. Reuse successful approaches; avoid repeating failures.",
    },
    MemoryType.PROCEDURAL.value: {
        "tag": "procedures",
        "guidance": "Follow these known procedures/tool sequences when the task matches.",
    },
}


def _format_memory_item(m: MemoryEntry) -> str:
    """Format a single memory entry, including relevant metadata."""
    line = f"- {m.content}"
    meta_parts = []
    if m.metadata:
        if m.metadata.get("tools_used"):
            meta_parts.append(f"tools: {', '.join(m.metadata['tools_used'])}")
        if m.metadata.get("outcome"):
            meta_parts.append(f"outcome: {m.metadata['outcome']}")
        if m.metadata.get("duration_seconds") is not None:
            meta_parts.append(f"took {m.metadata['duration_seconds']}s")
    if meta_parts:
        line += f" ({'; '.join(meta_parts)})"
    return line


def _format_memories(memories: list[MemoryEntry]) -> str:
    """Format a list of memory entries into XML blocks grouped by type."""
    if not memories:
        return ""

    # Group by type, keeping full MemoryEntry for metadata access
    grouped: dict[str, list[MemoryEntry]] = {}
    for m in memories:
        grouped.setdefault(m.memory_type, []).append(m)

    lines = ["<user_memories>"]
    lines.append("Remembered from previous conversations:")

    for mem_type, cfg in _TYPE_CONFIG.items():
        items = grouped.get(mem_type, [])
        if items:
            tag = cfg["tag"]
            lines.append(f"<{tag}>")
            lines.append(f"<!-- {cfg['guidance']} -->")
            for m in items:
                lines.append(_format_memory_item(m))
            lines.append(f"</{tag}>")

    # Handle any types not in the config map
    for mem_type, items in grouped.items():
        if mem_type not in _TYPE_CONFIG:
            lines.append(f"<{mem_type}>")
            for m in items:
                lines.append(_format_memory_item(m))
            lines.append(f"</{mem_type}>")

    lines.append("Use these to personalize responses. Do not mention them unless asked.")
    lines.append("</user_memories>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_memory_store = PersistentMemoryStore()


def get_memory_store() -> PersistentMemoryStore:
    """Get the global memory store singleton."""
    return _memory_store


# ---------------------------------------------------------------------------
# Memory extraction from conversations
# ---------------------------------------------------------------------------


async def extract_memories_from_conversation(
    messages: list[dict],
    user_id: str,
    conversation_id: str,
    provider: str | None = None,
    episodic_context: dict | None = None,
) -> list[MemoryEntry]:
    """Extract memorable facts/preferences from a conversation using FLASH LLM.

    Extracts all four memory types: preference, fact, episodic, procedural.

    Args:
        messages: Conversation message dicts with 'role' and 'content'.
        user_id: The user's ID.
        conversation_id: Conversation or task ID for provenance.
        provider: Optional LLM provider override.
        episodic_context: Optional dict with run metadata for richer episodic
            memories. Keys: task_description, tools_used, outcome,
            duration_seconds, mode.
    """
    from langchain_core.messages import HumanMessage

    from app.ai.llm import extract_text_from_content, llm_service
    from app.models.schemas import ModelTier

    conv_text = "\n".join(
        f"{m.get('role', 'unknown')}: {m.get('content', '')[:500]}"
        for m in messages[-20:]
    )

    if len(conv_text.strip()) < 50:
        return []

    # Build episodic context block if available
    episodic_block = ""
    if episodic_context:
        parts = []
        if episodic_context.get("task_description"):
            parts.append(f"Task: {episodic_context['task_description'][:300]}")
        if episodic_context.get("tools_used"):
            parts.append(f"Tools used: {', '.join(episodic_context['tools_used'])}")
        if episodic_context.get("outcome"):
            parts.append(f"Outcome: {episodic_context['outcome']}")
        if episodic_context.get("duration_seconds") is not None:
            parts.append(f"Duration: {episodic_context['duration_seconds']}s")
        if episodic_context.get("mode"):
            parts.append(f"Mode: {episodic_context['mode']}")
        if parts:
            episodic_block = (
                "\n\nTask execution context (use this to create episodic memories):\n"
                + "\n".join(f"- {p}" for p in parts)
                + "\n"
            )

    extraction_prompt = (
        "Analyze this conversation and extract key information about the user "
        "that would be useful in future conversations. Return a JSON array of objects, "
        "each with 'type' and 'content' fields.\n\n"
        "Types to extract:\n"
        "- 'preference': User preferences (coding style, language, tools, communication style)\n"
        "- 'fact': Facts about the user (role, company, expertise, projects)\n"
        "- 'episodic': Notable outcomes or experiences from this conversation "
        "(include what was done, which tools/skills were used, and whether it succeeded)\n"
        "- 'procedural': Workflows or procedures the user follows or prefers\n\n"
        "Only extract genuinely useful, specific information. Avoid generic observations. "
        "If nothing notable, return [].\n\n"
        f"Conversation:\n{conv_text}"
        f"{episodic_block}"
        "\n\nReturn ONLY valid JSON array:"
    )

    try:
        llm = llm_service.get_llm_for_tier(ModelTier.LITE, provider=provider)
        result = await llm.ainvoke([HumanMessage(content=extraction_prompt)])
        text = extract_text_from_content(result.content).strip()

        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        extracted = json.loads(text)
        if not isinstance(extracted, list):
            return []

        store = get_memory_store()
        entries = []
        for item in extracted:
            if isinstance(item, dict) and "content" in item:
                mem_type = item.get("type", "fact")
                if mem_type not in ALL_MEMORY_TYPES:
                    mem_type = "fact"

                # Merge episodic metadata for episodic memories
                item_metadata = item.get("metadata") or {}
                if mem_type == "episodic" and episodic_context:
                    item_metadata.setdefault("tools_used", episodic_context.get("tools_used", []))
                    item_metadata.setdefault("outcome", episodic_context.get("outcome"))
                    item_metadata.setdefault("duration_seconds", episodic_context.get("duration_seconds"))

                entry = await store.add_memory_async(
                    user_id=user_id,
                    memory_type=mem_type,
                    content=item["content"],
                    metadata=item_metadata if item_metadata else None,
                    source_conversation_id=conversation_id,
                )
                entries.append(entry)

        logger.info("memories_extracted", user_id=user_id, count=len(entries))
        return entries

    except Exception as e:
        logger.warning("memory_extraction_failed", error=str(e))
        return []
