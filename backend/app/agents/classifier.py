"""Query complexity classification for the orchestrator.

Determines whether a query needs planning (complex) or can be executed
directly (simple) using heuristic rules. No LLM call needed.
"""

import re

from app.agents.react_helpers import DIRECT_SKILL_MODES
from app.core.logging import get_logger

logger = get_logger(__name__)

_MULTI_STEP_PATTERNS = [
    r"\bfirst\b.*\bthen\b",
    r"\bstep\s*\d",
    r"\b(?:and\s+then|after\s+that|next|finally)\b",
    r"\d+\.\s+\w+.*\n\s*\d+\.\s+\w+",
]

_COMPLEXITY_KEYWORDS = [
    r"\bplan\b",
    r"\bstep.by.step\b",
    r"\bcomprehensive\b",
    r"\bcomplete\s+(?:system|app|application|project|solution|dashboard)\b",
    r"\bbuild\s+(?:a|an|the)\s+[\w\s]+?\s+(?:with|and|including)\b",
    r"\bimplement\s+(?:a|an|the)\s+[\w\s]+?\s+(?:with|and|including)\b",
    r"\bcreate\s+(?:a|an|the)\s+[\w\s]+?\s+(?:with|and|including)\b",
]

_MULTI_STEP_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _MULTI_STEP_PATTERNS]
_COMPLEXITY_RE = [re.compile(p, re.IGNORECASE) for p in _COMPLEXITY_KEYWORDS]


def classify_query(
    query: str,
    mode: str | None = None,
    skills: list[str] | None = None,
) -> str:
    """Classify query complexity as 'simple' or 'complex'.

    Args:
        query: User query text
        mode: Explicit mode override (app, image, slide, data, research)
        skills: Explicitly selected skill IDs from UI

    Returns:
        "simple" or "complex"
    """
    if mode and mode.lower() in DIRECT_SKILL_MODES:
        return "simple"

    if skills:
        return "simple"

    stripped = query.strip()
    if len(stripped.split()) < 5:
        return "simple"

    for pattern in _MULTI_STEP_RE:
        if pattern.search(stripped):
            logger.info("classify_complex_multi_step", pattern=pattern.pattern[:40])
            return "complex"

    for pattern in _COMPLEXITY_RE:
        if pattern.search(stripped):
            logger.info("classify_complex_keyword", pattern=pattern.pattern[:40])
            return "complex"

    return "simple"
