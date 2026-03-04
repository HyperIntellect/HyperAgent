"""Tests for deterministic routing mode."""

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.routing import route_query


@pytest.mark.asyncio
async def test_deterministic_routing_skips_llm():
    state = {"query": "hello world", "mode": "task"}
    with patch("app.agents.routing.settings.routing_mode", "deterministic"):
        with patch("app.agents.routing.llm_service.get_llm_for_tier") as mocked_get_llm:
            result = await route_query(state)
            assert result["selected_agent"] == "task"
            assert result["routing_confidence"] == 1.0
            assert "Deterministic routing" in result["routing_reason"]
            mocked_get_llm.assert_not_called()


@pytest.mark.asyncio
async def test_llm_routing_mode_invokes_router_llm():
    state = {"query": "hello world"}
    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(return_value=type("Resp", (), {"content": '{"agent":"task","confidence":0.9,"reason":"ok"}'})())
    with patch("app.agents.routing.settings.routing_mode", "llm"):
        with patch("app.agents.routing.llm_service.get_llm_for_tier", return_value=fake_llm):
            result = await route_query(state)
            assert result["selected_agent"] == "task"
            fake_llm.ainvoke.assert_awaited_once()
