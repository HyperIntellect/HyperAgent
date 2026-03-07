"""Tests for context compression: priority scoring, pruning, and summary validation."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents.context_compression import (
    CompressionConfig,
    ContextCompressor,
    _prune_low_priority_messages,
    _validate_summary_quality,
    estimate_message_tokens,
    score_message_priority,
    snap_to_tool_pair_boundary,
)


# ---------------------------------------------------------------------------
# Priority scoring tests
# ---------------------------------------------------------------------------


class TestScoreMessagePriority:
    def test_system_message_highest(self):
        assert score_message_priority(SystemMessage(content="system")) == 1.0

    def test_human_message_high(self):
        assert score_message_priority(HumanMessage(content="user input")) == 0.9

    def test_ai_with_tool_calls(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {}, "id": "1", "type": "tool_call"}],
        )
        assert score_message_priority(msg) == 0.7

    def test_ai_text_only(self):
        msg = AIMessage(content="Here's my analysis...")
        assert score_message_priority(msg) == 0.5

    def test_ai_recovery_message(self):
        msg = AIMessage(content="Trying again...", additional_kwargs={"recovery": True})
        assert score_message_priority(msg) == 0.2

    def test_tool_message_success(self):
        msg = ToolMessage(content="Result: 42", tool_call_id="1")
        assert score_message_priority(msg) == 0.6

    def test_tool_message_error(self):
        msg = ToolMessage(content="Error: file not found", tool_call_id="1")
        assert score_message_priority(msg) == 0.3

    def test_tool_message_traceback(self):
        msg = ToolMessage(content="Traceback (most recent call):\n...", tool_call_id="1")
        assert score_message_priority(msg) == 0.3

    def test_tool_message_exception(self):
        msg = ToolMessage(content="An exception occurred in the sandbox", tool_call_id="1")
        assert score_message_priority(msg) == 0.3


# ---------------------------------------------------------------------------
# Priority-aware pruning tests
# ---------------------------------------------------------------------------


class TestPruneLowPriorityMessages:
    def _make_messages(self):
        """Build a list with varying priorities."""
        return [
            SystemMessage(content="You are an assistant"),      # 1.0 — never pruned
            HumanMessage(content="Hello" * 100),                # 0.9
            AIMessage(content="Let me help" * 100),             # 0.5
            AIMessage(
                content="Trying again",
                additional_kwargs={"recovery": True},
            ),                                                  # 0.2
            ToolMessage(
                content="Error: something broke",
                tool_call_id="t1",
            ),                                                  # 0.3
            AIMessage(content="Final answer" * 100),            # 0.5
        ]

    def test_no_pruning_when_under_budget(self):
        messages = self._make_messages()
        total = sum(estimate_message_tokens(m) for m in messages)
        result = _prune_low_priority_messages(messages, total + 1000)
        assert len(result) == len(messages)

    def test_lowest_priority_pruned_first(self):
        messages = self._make_messages()
        # Set a very small target to force pruning
        result = _prune_low_priority_messages(messages, 100)
        # System message (1.0) must survive
        assert any(isinstance(m, SystemMessage) for m in result)
        # Recovery message (0.2) should be pruned first
        recovery_present = any(
            isinstance(m, AIMessage) and m.additional_kwargs.get("recovery")
            for m in result
        )
        assert not recovery_present

    def test_tool_pairs_pruned_together(self):
        messages = [
            SystemMessage(content="sys"),
            HumanMessage(content="query" * 50),
            AIMessage(
                content="",
                tool_calls=[{"name": "search", "args": {}, "id": "t1", "type": "tool_call"}],
            ),
            ToolMessage(content="result data " * 50, tool_call_id="t1"),
            AIMessage(content="final answer" * 50),
        ]
        # Force tight budget
        result = _prune_low_priority_messages(messages, 50)
        # If the tool pair is present, both parts must be present
        ai_tool_present = any(
            isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
            for m in result
        )
        tool_msg_present = any(isinstance(m, ToolMessage) for m in result)
        # They're either both present or both absent
        assert ai_tool_present == tool_msg_present

    def test_system_messages_never_pruned(self):
        messages = [
            SystemMessage(content="important " * 500),
            HumanMessage(content="hi"),
        ]
        result = _prune_low_priority_messages(messages, 10)
        assert any(isinstance(m, SystemMessage) for m in result)


# ---------------------------------------------------------------------------
# Summary validation tests
# ---------------------------------------------------------------------------


class TestValidateSummaryQuality:
    def test_complete_summary_passes(self):
        summary = "Used web_search and execute_code to analyze /home/user/data.csv. Built a dashboard."
        refs = {
            "tools": ["web_search", "execute_code"],
            "files": ["/home/user/data.csv"],
        }
        old_messages = [HumanMessage(content="Build a dashboard from data.csv")]
        is_valid, missing = _validate_summary_quality(summary, refs, old_messages)
        assert is_valid
        assert missing == []

    def test_missing_tools_detected(self):
        summary = "Completed the task successfully."
        refs = {
            "tools": ["web_search", "execute_code", "file_read"],
        }
        old_messages = [HumanMessage(content="Search and analyze")]
        is_valid, missing = _validate_summary_quality(summary, refs, old_messages)
        assert not is_valid
        tool_missing = [m for m in missing if m.startswith("tool:")]
        assert len(tool_missing) > 0

    def test_missing_files_detected(self):
        summary = "Analyzed the data."
        refs = {
            "files": ["/home/user/data.csv", "/tmp/output.json", "/var/log/app.log"],
        }
        old_messages = []
        is_valid, missing = _validate_summary_quality(summary, refs, old_messages)
        assert not is_valid
        file_missing = [m for m in missing if m.startswith("file:")]
        assert len(file_missing) > 0

    def test_no_false_positives_on_complete_summary(self):
        summary = "Used web_search to find information. Read /tmp/data.txt."
        refs = {
            "tools": ["web_search"],
            "files": ["/tmp/data.txt"],
        }
        old_messages = []
        is_valid, missing = _validate_summary_quality(summary, refs, old_messages)
        assert is_valid

    def test_empty_refs_always_valid(self):
        summary = "A simple conversation about weather."
        refs = {}
        old_messages = [HumanMessage(content="How's the weather?")]
        is_valid, missing = _validate_summary_quality(summary, refs, old_messages)
        # With no refs, only keyword check applies
        # "weather" should be in summary
        assert is_valid

    def test_keyword_preservation_check(self):
        summary = "Discussed some topics."
        refs = {}
        old_messages = [
            HumanMessage(content="Explain kubernetes deployment strategies and autoscaling"),
        ]
        is_valid, missing = _validate_summary_quality(summary, refs, old_messages)
        assert not is_valid
        keyword_missing = [m for m in missing if m.startswith("keyword:")]
        assert len(keyword_missing) > 0


# ---------------------------------------------------------------------------
# Integration: CompressionConfig flags
# ---------------------------------------------------------------------------


class TestCompressionConfigFlags:
    def test_pruning_enabled_default(self):
        config = CompressionConfig()
        assert config.pruning_enabled is True

    def test_summary_validation_enabled_default(self):
        config = CompressionConfig()
        assert config.summary_validation_enabled is True

    def test_pruning_can_be_disabled(self):
        config = CompressionConfig(pruning_enabled=False)
        assert config.pruning_enabled is False
