"""Scanner for sanitizing untrusted intermediate content.

This scanner is used for content that should be treated as data (tool outputs,
attachment extracts, compressed summaries), not as executable instructions.
"""

import re
import unicodedata
from typing import Literal

from app.config import settings
from app.core.logging import get_logger
from app.guardrails.base import ScanResult, ViolationType

logger = get_logger(__name__)

Sensitivity = Literal["low", "medium", "high"]

_NEWLINE_COLLAPSE_RE = re.compile(r"\n{3,}")

# Pre-compiled pattern sets by sensitivity. "high" is strictest.
_PATTERNS: dict[Sensitivity, tuple[re.Pattern, ...]] = {
    "low": (
        re.compile(r"(?im)\bignore\b.{0,80}\b(instructions?|policy|rule|safety)\b"),
        re.compile(r"(?im)\boverride\b.{0,80}\b(instructions?|policy|guardrail)\b"),
        re.compile(r"(?im)\bjailbreak\b|\bbypass\b.{0,80}\bguardrail\b"),
    ),
    "medium": (
        re.compile(r"(?im)^\s*(system|developer|instruction|important)\s*:\s*.*$"),
        re.compile(r"(?im)\bignore\b.{0,100}\b(instructions?|policy|rule|safety)\b"),
        re.compile(r"(?im)\boverride\b.{0,100}\b(instructions?|policy|guardrail)\b"),
        re.compile(r"(?im)\breveal\b.{0,100}\b(system prompt|instructions?)\b"),
        re.compile(r"(?im)\bjailbreak\b|\bbypass\b.{0,100}\bguardrail\b"),
    ),
    "high": (
        re.compile(r"(?im)^\s*(system|developer|instruction|important)\s*:\s*.*$"),
        re.compile(r"(?im)^\s*you must\b.*$"),
        re.compile(r"(?im)^\s*follow these steps\b.*$"),
        re.compile(r"(?im)\bignore\b.{0,140}\b(instructions?|policy|rule|safety)\b"),
        re.compile(r"(?im)\boverride\b.{0,140}\b(instructions?|policy|guardrail)\b"),
        re.compile(r"(?im)\breveal\b.{0,120}\b(system prompt|instructions?)\b"),
        re.compile(r"(?im)\b(exfiltrate|leak)\b.{0,80}\b(secret|token|key|credential)\b"),
        re.compile(r"(?im)\bjailbreak\b|\bbypass\b.{0,140}\bguardrail\b"),
    ),
}


def _sanitize_content(content: str, sensitivity: Sensitivity) -> str:
    sanitized = content
    for pattern in _PATTERNS[sensitivity]:
        sanitized = pattern.sub("", sanitized)
    sanitized = sanitized.replace("<", "").replace(">", "")
    sanitized = _NEWLINE_COLLAPSE_RE.sub("\n\n", sanitized).strip()
    return sanitized


def sanitize_untrusted_content(content: str, sensitivity: Sensitivity = "medium") -> str:
    """Synchronous sanitization helper for non-async call sites.

    Runs sanitization in a single pass — applies re.sub unconditionally
    and compares output length to detect whether anything matched.
    """
    if not content:
        return ""
    normalized = unicodedata.normalize("NFKC", content)
    sanitized = _sanitize_content(normalized, sensitivity)
    if len(sanitized) == len(content):
        return content
    return sanitized


class UntrustedContentScanner:
    """Detect and sanitize prompt-injection-like instructions in untrusted content."""

    def __init__(self) -> None:
        self._enabled = settings.guardrails_enabled

    async def scan(
        self,
        content: str,
        *,
        sensitivity: Sensitivity = "medium",
        source: str = "unknown",
    ) -> ScanResult:
        if not self._enabled:
            return ScanResult.allow()
        if not content or not content.strip():
            return ScanResult.allow()

        normalized = unicodedata.normalize("NFKC", content)

        # Single-pass: sanitize and check if anything changed
        sanitized = _sanitize_content(normalized, sensitivity)
        if len(sanitized) == len(content):
            return ScanResult.allow()

        if not sanitized:
            sanitized = "[Content removed due to unsafe instruction-like patterns.]"

        reason = "Instruction-like pattern found in untrusted content"
        logger.warning(
            "untrusted_content_sanitized",
            source=source,
            sensitivity=sensitivity,
            reason=reason,
        )
        return ScanResult.flag(
            violations=[ViolationType.PROMPT_INJECTION],
            reason=reason,
            sanitized_content=sanitized,
        )


untrusted_content_scanner = UntrustedContentScanner()
