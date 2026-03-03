"""Helper functions for the query API endpoints.

Extracted from query.py to keep the main endpoint file focused on
routing and streaming logic.
"""

import asyncio
import base64
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import Conversation, ConversationMessage
from app.db.models import File as FileModel
from app.services.file_storage import file_storage_service

logger = get_logger(__name__)


TASK_SYSTEM_PROMPT = """You are HyperAgent, a versatile AI assistant and general-purpose task handler. You help users accomplish a wide range of tasks including coding, research, analysis, and general questions.

You have access to a web search tool that you can use to find current information when needed. Use it when:
- The user asks about recent events or news
- You need to verify facts or find up-to-date information
- The question requires knowledge beyond your training data

When you decide to search, refine the query to improve quality:
- Include specific entities, versions, dates, and locations
- Add the most likely authoritative source (e.g. official docs/site:example.com)
- Use short, focused queries rather than a single broad query
- Avoid vague terms; include exact product or feature names

Be concise, accurate, and helpful. When providing code, use proper formatting with markdown code blocks and specify the language.

If you're unsure about something, say so rather than making things up."""

MAX_CHAT_HISTORY_MESSAGES = 20


async def get_conversation_history(
    db: AsyncSession,
    conversation_id: str | None,
    user_id: str,
    limit: int = MAX_CHAT_HISTORY_MESSAGES,
) -> list[dict]:
    """Fetch recent conversation history for short-term memory."""
    if not conversation_id:
        return []

    result = await db.execute(
        select(Conversation.id).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )
    if not result.scalar_one_or_none():
        return []

    message_result = await db.execute(
        select(ConversationMessage)
        .where(ConversationMessage.conversation_id == conversation_id)
        .order_by(ConversationMessage.created_at.desc())
        .limit(limit)
    )
    messages = list(reversed(message_result.scalars().all()))
    history = [
        {
            "role": message.role,
            "content": message.content,
            "metadata": message.message_metadata,
        }
        for message in messages
        if message.role in ("user", "assistant")
    ]
    return history


def trim_duplicate_user_message(history: list[dict], query: str) -> list[dict]:
    """Remove duplicate trailing user message when it matches the current query."""
    if not history:
        return history
    last = history[-1]
    if last.get("role") == "user" and last.get("content", "").strip() == query.strip():
        return history[:-1]
    return history


async def get_file_context(
    db: AsyncSession,
    attachment_ids: list[str],
    user_id: str,
) -> str:
    """Get extracted text from attached files for LLM context.

    Includes sandbox paths so the LLM knows where files are available
    for code execution (uploaded to /home/user/ in the sandbox).
    """
    if not attachment_ids:
        return ""

    result = await db.execute(
        select(FileModel).where(
            FileModel.id.in_(attachment_ids),
            FileModel.user_id == user_id,
        )
    )
    files = result.scalars().all()

    context_parts = []
    for file in files:
        # Build sandbox path matching _safe_filename in data_analysis_skill
        safe_name = f"{file.id}_{file.original_filename.rsplit('/', 1)[-1] or 'file'}"
        sandbox_path = f"/home/user/{safe_name}"

        if file.extracted_text:
            context_parts.append(
                f"[Attached file: {file.original_filename}]\n"
                f"Sandbox path: {sandbox_path}\n"
                f"{file.extracted_text}\n"
            )
        else:
            context_parts.append(
                f"[Attached file: {file.original_filename}"
                f" - binary content, use sandbox path for analysis]\n"
                f"Sandbox path: {sandbox_path}\n"
            )

    if context_parts:
        return "\n---\n".join(context_parts)
    return ""


# Image MIME types that can be processed by vision tools
IMAGE_MIME_TYPES: set[str] = {"image/png", "image/jpeg", "image/gif", "image/webp"}


async def get_image_attachments(
    db: AsyncSession,
    attachment_ids: list[str],
    user_id: str,
) -> list[dict]:
    """Get image attachments as base64 for vision tool usage.

    Uses asyncio.gather for parallel downloads to improve performance.

    Returns:
        List of dicts with {id, filename, base64_data, mime_type}
    """
    if not attachment_ids:
        return []

    result = await db.execute(
        select(FileModel).where(
            FileModel.id.in_(attachment_ids),
            FileModel.user_id == user_id,
        )
    )
    files = result.scalars().all()

    # Filter to only image files
    image_files = [f for f in files if f.content_type in IMAGE_MIME_TYPES]

    if not image_files:
        return []

    async def process_image(file: FileModel) -> dict | None:
        """Process a single image file and return attachment dict or None on error."""
        try:
            file_data = await file_storage_service.download_file(file.storage_key)
            base64_data = base64.b64encode(file_data.read()).decode("utf-8")

            logger.info(
                "image_attachment_loaded",
                file_id=file.id,
                filename=file.original_filename,
                mime_type=file.content_type,
            )

            return {
                "id": file.id,
                "filename": file.original_filename,
                "base64_data": base64_data,
                "mime_type": file.content_type,
            }
        except Exception as e:
            logger.error(
                "image_attachment_load_failed",
                file_id=file.id,
                error=str(e),
            )
            return None

    # Download all images in parallel
    results = await asyncio.gather(*[process_image(f) for f in image_files])

    # Filter out None results (failed downloads)
    return [r for r in results if r is not None]


def _sse_data(payload: dict[str, Any]) -> str:
    """Format a payload as an SSE data line.

    Args:
        payload: Dictionary to serialize as JSON

    Returns:
        SSE formatted data string
    """
    return f"data: {json.dumps(payload)}\n\n"
