"""Tests for the executor reflection module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.reflection import (
    _parse_reflection_output,
    reflect,
    should_reflect,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> dict:
    """Create a minimal ExecutorState-like dict for testing."""
    defaults = {
        "tool_iterations": 2,
        "reflection_count": 0,
        "mode": None,
        "provider": "anthropic",
        "step_goal": "Summarize the search results",
        "lc_messages": [],
    }
    return {**defaults, **overrides}


# ---------------------------------------------------------------------------
# should_reflect tests
# ---------------------------------------------------------------------------


class TestShouldReflect:
    """Tests for the should_reflect heuristic gate."""

    def test_should_reflect_true(self):
        """Reflect when tools were used and reflection hasn't fired yet."""
        state = _make_state(tool_iterations=2, reflection_count=0)
        with patch("app.agents.reflection.settings") as mock_settings:
            mock_settings.reflection_enabled = True
            mock_settings.reflection_min_tool_iterations = 1
            mock_settings.reflection_max_count = 1
            assert should_reflect(state) is True

    def test_should_reflect_false_no_tools(self):
        """Don't reflect if no tools were used."""
        state = _make_state(tool_iterations=0, reflection_count=0)
        with patch("app.agents.reflection.settings") as mock_settings:
            mock_settings.reflection_enabled = True
            mock_settings.reflection_min_tool_iterations = 1
            mock_settings.reflection_max_count = 1
            assert should_reflect(state) is False

    def test_should_reflect_false_already_reflected(self):
        """Don't reflect if max reflection count is reached."""
        state = _make_state(tool_iterations=2, reflection_count=1)
        with patch("app.agents.reflection.settings") as mock_settings:
            mock_settings.reflection_enabled = True
            mock_settings.reflection_min_tool_iterations = 1
            mock_settings.reflection_max_count = 1
            assert should_reflect(state) is False

    def test_should_reflect_false_disabled(self):
        """Don't reflect when reflection is disabled."""
        state = _make_state(tool_iterations=2, reflection_count=0)
        with patch("app.agents.reflection.settings") as mock_settings:
            mock_settings.reflection_enabled = False
            assert should_reflect(state) is False

    def test_should_reflect_false_dedicated_mode(self):
        """Don't reflect for dedicated modes (app, image, slide)."""
        for mode in ("app", "image", "slide", "slides"):
            state = _make_state(tool_iterations=2, reflection_count=0, mode=mode)
            with patch("app.agents.reflection.settings") as mock_settings:
                mock_settings.reflection_enabled = True
                mock_settings.reflection_min_tool_iterations = 1
                mock_settings.reflection_max_count = 1
                assert should_reflect(state) is False, f"Should not reflect for mode={mode}"


# ---------------------------------------------------------------------------
# _parse_reflection_output tests
# ---------------------------------------------------------------------------


class TestParseReflectionOutput:
    """Tests for parsing the LLM reflection output."""

    def test_parse_pass(self):
        raw = "PASS\n0.92\nResponse fully addresses the goal."
        result = _parse_reflection_output(raw)
        assert result.verdict == "pass"
        assert result.confidence == pytest.approx(0.92)
        assert result.note == "Response fully addresses the goal."

    def test_parse_retry(self):
        raw = "RETRY\n0.3\nMissing key information about the topic."
        result = _parse_reflection_output(raw)
        assert result.verdict == "retry"
        assert result.confidence == pytest.approx(0.3)

    def test_parse_complete_with_note(self):
        raw = "COMPLETE_WITH_NOTE\n0.7\nAcceptable but could include more detail."
        result = _parse_reflection_output(raw)
        assert result.verdict == "complete_with_note"

    def test_parse_malformed_defaults_to_pass(self):
        raw = "something unexpected"
        result = _parse_reflection_output(raw)
        assert result.verdict == "pass"
        assert result.confidence == 0.8  # default

    def test_parse_confidence_clamped(self):
        raw = "PASS\n1.5\nNote"
        result = _parse_reflection_output(raw)
        assert result.confidence == 1.0

        raw_neg = "PASS\n-0.5\nNote"
        result_neg = _parse_reflection_output(raw_neg)
        assert result_neg.confidence == 0.0


# ---------------------------------------------------------------------------
# reflect tests
# ---------------------------------------------------------------------------


class TestReflect:
    """Tests for the LLM-based reflect function."""

    @pytest.mark.asyncio
    async def test_reflect_pass_verdict(self):
        """Mock LITE LLM returns pass verdict."""
        state = _make_state(step_goal="Find Python docs")

        mock_llm = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = "PASS\n0.95\nResponse is comprehensive."
        mock_llm.ainvoke.return_value = mock_result

        with patch("app.agents.reflection.llm_service") as mock_svc, \
             patch("app.agents.reflection.settings") as mock_settings:
            mock_settings.reflection_model_tier = "lite"
            mock_svc.get_llm_for_tier.return_value = mock_llm
            result = await reflect(state)

        assert result.verdict == "pass"
        assert result.confidence == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_reflect_retry_verdict(self):
        """Mock LITE LLM returns retry verdict."""
        state = _make_state(step_goal="Analyze market trends")

        mock_llm = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = "RETRY\n0.25\nResponse does not address the step goal."
        mock_llm.ainvoke.return_value = mock_result

        with patch("app.agents.reflection.llm_service") as mock_svc, \
             patch("app.agents.reflection.settings") as mock_settings:
            mock_settings.reflection_model_tier = "lite"
            mock_svc.get_llm_for_tier.return_value = mock_llm
            result = await reflect(state)

        assert result.verdict == "retry"
        assert result.confidence == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_reflect_error_fallback(self):
        """LLM error defaults to pass so reflection doesn't block."""
        state = _make_state()

        with patch("app.agents.reflection.llm_service") as mock_svc, \
             patch("app.agents.reflection.settings") as mock_settings:
            mock_settings.reflection_model_tier = "lite"
            mock_svc.get_llm_for_tier.side_effect = RuntimeError("API unavailable")
            result = await reflect(state)

        assert result.verdict == "pass"
        assert result.confidence == 0.5
        assert "Reflection error" in result.note
