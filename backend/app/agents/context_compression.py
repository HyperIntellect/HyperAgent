"""Context compression for managing long conversations.

This module provides LLM-based context compression to preserve semantic meaning
when conversations exceed token limits. Instead of simply dropping old messages,
it summarizes them using a fast LLM (FLASH tier).
"""

from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.ai.llm import extract_text_from_content, llm_service
from app.ai.model_tiers import ModelTier
from app.core.logging import get_logger

logger = get_logger(__name__)

# Compression prompt template with recoverable reference extraction
COMPRESSION_PROMPT = """You are a conversation summarizer. Your task is to create a concise summary that preserves all essential information AND recoverable references.

Focus on:
1. The main topic/goal of the conversation
2. Key decisions made or conclusions reached
3. Important facts, data, or information exchanged
4. Current state of any ongoing tasks
5. User preferences or requirements mentioned

CRITICAL: You MUST preserve all recoverable references in a dedicated section. These allow the agent to re-access information after compression:

Format your response as:

## Summary
[Concise summary of the conversation]

## Preserved References
- Files: [list all file paths mentioned, e.g., /home/user/app.py, /tmp/output.csv]
- URLs: [list all URLs mentioned]
- Tools Used: [list tool names that were called]
- Variables: [list key variable names, function names, class names mentioned]
- Sandbox IDs: [list any sandbox/session IDs]
- Commands: [list shell commands that were executed]

If a category has no items, omit it.
{language_section}
{existing_summary_section}

Messages to summarize:
{messages_text}

Provide the structured summary with preserved references:"""


@dataclass
class CompressionResult:
    """Result of context compression with preserved references.

    Attributes:
        summary: The compressed conversation summary
        preserved_refs: Recoverable references extracted during compression
    """

    summary: str
    preserved_refs: dict[str, list[str]]


@dataclass
class CompressionConfig:
    """Configuration for context compression.

    Attributes:
        token_threshold: Token count that triggers compression (default 60k = 60% of 100k budget)
        preserve_recent: Number of recent messages to always keep intact
        min_messages_to_compress: Minimum messages needed before compression is worthwhile
        max_summary_tokens: Maximum tokens for the summary itself
        enabled: Whether compression is enabled
    """

    token_threshold: int = 60000
    preserve_recent: int = 10
    min_messages_to_compress: int = 5
    max_summary_tokens: int = 2000
    enabled: bool = True


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses tiktoken for accurate estimation when available; falls back to a
    simple heuristic of ~4 characters per token.  Note: the len(text) // 4
    heuristic underestimates CJK text where each character often maps to a
    single token.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 4 + 1


def estimate_message_tokens(message: BaseMessage) -> int:
    """Estimate token count for a LangChain message.

    Args:
        message: Message to estimate

    Returns:
        Estimated token count
    """
    content = ""
    if isinstance(message.content, str):
        content = message.content
    elif isinstance(message.content, list):
        for item in message.content:
            if isinstance(item, str):
                content += item
            elif isinstance(item, dict) and item.get("type") == "text":
                content += item.get("text", "")
    return estimate_tokens(content)


def _format_message_for_summary(message: BaseMessage) -> str:
    """Format a message for inclusion in the summary prompt.

    Args:
        message: Message to format

    Returns:
        Formatted string representation
    """
    role = "System"
    if isinstance(message, HumanMessage):
        role = "User"
    elif isinstance(message, AIMessage):
        role = "Assistant"

    content = ""
    if isinstance(message.content, str):
        content = message.content
    elif isinstance(message.content, list):
        parts = []
        for item in message.content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "tool_use":
                    parts.append(f"[Tool call: {item.get('name', 'unknown')}]")
        content = " ".join(parts)

    # Truncate very long messages for the summary prompt
    if len(content) > 1000:
        content = content[:1000] + "... [truncated]"

    return f"{role}: {content}"


def _extract_references_from_messages(messages: list[BaseMessage]) -> dict[str, list[str]]:
    """Pre-extract recoverable references from messages before compression.

    This ensures references survive even if the LLM summary misses some.
    The agent can always re-read files, re-visit URLs, etc.

    Args:
        messages: Messages to extract references from

    Returns:
        Dict mapping reference types to lists of references
    """
    import re

    refs: dict[str, set[str]] = {
        "files": set(),
        "urls": set(),
        "tools": set(),
        "commands": set(),
    }

    for msg in messages:
        content = ""
        if isinstance(msg, str):
            content = msg
        elif isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, list):
            content = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in msg.content
            )

        # Extract file paths (Unix-style)
        file_paths = re.findall(r'(?:/[\w.-]+)+(?:\.\w+)?', content)
        for fp in file_paths:
            if len(fp) > 4 and not fp.startswith("//"):  # Filter out short fragments
                refs["files"].add(fp)

        # Extract URLs
        urls = re.findall(r'https?://[^\s<>"\')\]]+', content)
        refs["urls"].update(urls)

        # Extract tool names from ToolMessages
        if isinstance(msg, ToolMessage) and msg.name:
            refs["tools"].add(msg.name)

        # Extract tool calls from AIMessages
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                refs["tools"].add(tc.get("name", ""))

        # Extract shell commands (common patterns)
        cmd_patterns = re.findall(r'(?:shell_exec|command)["\s:]+([^"}\n]+)', content)
        refs["commands"].update(c.strip() for c in cmd_patterns if len(c.strip()) > 3)

    return {k: sorted(v) for k, v in refs.items() if v}


class ContextCompressor:
    """Compresses conversation context using LLM summarization.

    This class handles:
    - Checking if compression is needed based on token thresholds
    - Extracting recoverable references (files, URLs, tools) before summarization
    - Summarizing old messages while preserving recent ones
    - Injecting summaries back into the conversation with preserved references
    """

    def __init__(self, config: CompressionConfig | None = None):
        """Initialize the compressor.

        Args:
            config: Compression configuration (uses defaults if None)
        """
        self.config = config or CompressionConfig()

    @staticmethod
    def _snap_to_tool_pair_boundary(messages: list[BaseMessage], split_index: int) -> int:
        """Adjust split index so it doesn't orphan ToolMessages from their AIMessage.

        Scans backward from the proposed split so that:
        - No ToolMessage is separated from its parent AIMessage
        - No AIMessage with tool_calls is separated from its ToolMessage responses
        """
        if split_index <= 0 or split_index >= len(messages):
            return split_index

        while split_index > 0 and isinstance(messages[split_index], ToolMessage):
            split_index -= 1

        if (
            split_index > 0
            and isinstance(messages[split_index - 1], AIMessage)
            and getattr(messages[split_index - 1], "tool_calls", None)
        ):
            split_index -= 1

        return split_index

    def should_compress(
        self,
        messages: list[BaseMessage],
        existing_summary: str | None = None,
    ) -> bool:
        """Check if compression should be triggered.

        Compression is triggered when:
        1. Compression is enabled
        2. Total tokens exceed threshold
        3. There are enough messages to make compression worthwhile

        Args:
            messages: Current message list
            existing_summary: Any existing summary from previous compression

        Returns:
            True if compression should be performed
        """
        if not self.config.enabled:
            return False

        # Count non-system messages (system messages don't count toward compression)
        non_system_count = sum(
            1 for m in messages if not isinstance(m, SystemMessage)
        )

        # Need minimum messages to make compression worthwhile
        if non_system_count < self.config.preserve_recent + self.config.min_messages_to_compress:
            return False

        # Calculate total token estimate
        total_tokens = sum(estimate_message_tokens(m) for m in messages)
        if existing_summary:
            total_tokens += estimate_tokens(existing_summary)

        return total_tokens > self.config.token_threshold

    async def compress(
        self,
        messages: list[BaseMessage],
        existing_summary: str | None,
        provider: str,
        locale: str = "en",
    ) -> tuple[str | None, list[BaseMessage]]:
        """Compress older messages into a summary with preserved references.

        Separates messages into:
        - System messages (always preserved)
        - Old messages (to be summarized)
        - Recent messages (preserved intact)

        Before summarization, extracts recoverable references (file paths, URLs,
        tool names, commands) so the agent can re-access information after compression.

        Args:
            messages: Full message list
            existing_summary: Any existing summary to build upon
            provider: LLM provider for summarization
            locale: User's preferred language code for summary generation

        Returns:
            Tuple of (new_summary, preserved_messages)
            If compression not needed, returns (None, original_messages)
        """
        if not self.should_compress(messages, existing_summary):
            return None, messages

        # Separate messages by type
        system_messages = []
        other_messages = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_messages.append(msg)
            else:
                other_messages.append(msg)

        # Split into old (to summarize) and recent (to preserve).
        # Snap the boundary so we don't split an AIMessage(tool_calls)
        # from its corresponding ToolMessage responses.
        preserve_count = self.config.preserve_recent
        if len(other_messages) <= preserve_count:
            # Not enough messages to compress
            return None, messages

        raw_split = len(other_messages) - preserve_count
        split_index = self._snap_to_tool_pair_boundary(other_messages, raw_split)

        old_messages = other_messages[:split_index]
        recent_messages = other_messages[split_index:]

        # Check if we have enough old messages to summarize
        if len(old_messages) < self.config.min_messages_to_compress:
            return None, messages

        # --- Recoverable reference extraction (pre-LLM, guaranteed) ---
        extracted_refs = _extract_references_from_messages(old_messages)

        logger.info(
            "context_compression_starting",
            total_messages=len(messages),
            messages_to_compress=len(old_messages),
            messages_to_preserve=len(recent_messages),
            existing_summary_length=len(existing_summary) if existing_summary else 0,
            extracted_refs_count=sum(len(v) for v in extracted_refs.values()),
        )

        # Build summary prompt
        existing_summary_section = ""
        if existing_summary:
            existing_summary_section = f"Previous conversation summary:\n{existing_summary}\n\nAdditional messages to incorporate:"

        # Include language instruction so the summary matches the conversation language
        language_section = ""
        if locale and locale != "en":
            from app.agents.prompts import LANGUAGE_MAP

            language = LANGUAGE_MAP.get(locale, locale)
            language_section = f"\nIMPORTANT: Write the summary in {language} to match the conversation language."

        messages_text = "\n\n".join(
            _format_message_for_summary(m) for m in old_messages
        )

        prompt = COMPRESSION_PROMPT.format(
            existing_summary_section=existing_summary_section,
            messages_text=messages_text,
            language_section=language_section,
        )

        # Use FLASH tier for fast, cheap summarization
        try:
            llm = llm_service.get_llm_for_tier(
                tier=ModelTier.FLASH,
                provider=provider,
            )

            response = await llm.ainvoke([HumanMessage(content=prompt)])
            new_summary = extract_text_from_content(response.content).strip()

            # Append guaranteed extracted references (in case LLM missed some)
            if extracted_refs:
                ref_block = "\n\n## Extracted References (automated)\n"
                for ref_type, ref_list in extracted_refs.items():
                    ref_block += f"- {ref_type.title()}: {', '.join(ref_list[:20])}\n"
                new_summary += ref_block

            # Validate summary isn't too long
            summary_tokens = estimate_tokens(new_summary)
            if summary_tokens > self.config.max_summary_tokens:
                # Truncate summary if too long, but keep references
                max_chars = self.config.max_summary_tokens * 4
                # Try to preserve the references section
                if "## Extracted References" in new_summary:
                    parts = new_summary.split("## Extracted References", 1)
                    summary_part = parts[0][:max_chars - 500] + "... [truncated]"
                    new_summary = summary_part + "\n## Extracted References" + parts[1]
                else:
                    new_summary = new_summary[:max_chars] + "... [summary truncated]"

            logger.info(
                "context_compression_completed",
                summary_tokens=estimate_tokens(new_summary),
                compressed_messages=len(old_messages),
                preserved_messages=len(recent_messages),
                preserved_refs=sum(len(v) for v in extracted_refs.values()),
            )

            # Return the new summary and preserved messages
            preserved = system_messages + recent_messages
            return new_summary, preserved

        except Exception as e:
            logger.error(
                "context_compression_failed",
                error=str(e),
            )
            # On failure, return original messages without compression
            return None, messages


def inject_summary_as_context(
    messages: list[BaseMessage],
    summary: str,
) -> list[BaseMessage]:
    """Inject a context summary into the message list.

    Adds the summary as a system message right after the main system prompt.
    This ensures the LLM has access to the compressed conversation history.

    Args:
        messages: Current message list (should have system message first)
        summary: Summary to inject

    Returns:
        New message list with summary injected
    """
    if not summary:
        return messages

    summary_message = SystemMessage(
        content=f"[Previous conversation summary]\n{summary}\n[End of summary - recent messages follow]"
    )

    result = []
    summary_injected = False

    for msg in messages:
        result.append(msg)
        # Inject after the first system message
        if isinstance(msg, SystemMessage) and not summary_injected:
            result.append(summary_message)
            summary_injected = True

    # If no system message found, prepend the summary
    if not summary_injected:
        result.insert(0, summary_message)

    return result


def get_stable_prefix(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Extract the stable prefix from a message list (system messages at the start).

    The stable prefix consists of all leading SystemMessage instances.
    These typically include the system prompt, user memories, and context
    summaries -- content that doesn't change between ReAct iterations.
    Keeping this prefix unchanged allows KV-cache reuse.

    Args:
        messages: Full message list

    Returns:
        List of leading SystemMessage instances
    """
    prefix: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            prefix.append(msg)
        else:
            break
    return prefix


def get_dynamic_suffix(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Extract the dynamic suffix from a message list.

    The dynamic suffix starts at the first non-SystemMessage. This
    portion changes each iteration (new user/AI/tool messages) and sits
    after the stable prefix for KV-cache efficiency.

    Args:
        messages: Full message list

    Returns:
        List of messages after the leading system messages
    """
    for i, msg in enumerate(messages):
        if not isinstance(msg, SystemMessage):
            return messages[i:]
    return []


# Default compressor instance
default_compressor = ContextCompressor()
