"""Guardrails module for input/output safety scanning.

Provides comprehensive guardrails at 4 integration points:
1. Input - Query input before routing (prompt injection, jailbreak detection)
2. Output - LLM response after generation (toxicity, PII, harmful content)
3. Tool Args - Tool arguments before execution (URL validation, code safety)
4. Final Output - Response before streaming to user (final sanitization)
"""

from app.guardrails.base import ScanResult, ViolationType
from app.guardrails.scanners import (
    input_scanner,
    output_scanner,
    tool_scanner,
    untrusted_content_scanner,
)

__all__ = [
    "ScanResult",
    "ViolationType",
    "input_scanner",
    "output_scanner",
    "tool_scanner",
    "untrusted_content_scanner",
]
