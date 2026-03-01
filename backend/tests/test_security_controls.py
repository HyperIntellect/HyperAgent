"""Security regression tests for authz, path validation, and tool policy controls."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.agents.hitl.tool_risk import (
    ToolRiskLevel,
    get_skill_risk_level,
    requires_approval_for_skill,
)
from app.agents.tools.tool_pipeline import (
    CanonicalToolHooks,
    ResearchToolHooks,
    ToolExecutionContext,
)
from app.agents.tools.registry import get_all_tools, validate_tool_contracts
from app.agents.policy import PolicyInput, get_policy_engine
from app.api.sandbox import _validate_sandbox_path
from app.api.skills import (
    _enforce_skill_ownership_or_builtin,
    _validate_skill_tool_metadata,
    get_skill,
    update_skill,
    UpdateSkillRequest,
)
from app.services.skill_registry import _create_safe_namespace


class TestSandboxPathValidation:
    """Tests for sandbox path boundary checks."""

    def test_accepts_base_directory(self):
        assert _validate_sandbox_path("/home/user") == "/home/user"

    def test_accepts_subpath(self):
        assert _validate_sandbox_path("/home/user/project/app.py") == "/home/user/project/app.py"

    def test_rejects_prefix_confusion_path(self):
        with pytest.raises(HTTPException) as exc:
            _validate_sandbox_path("/home/user2/secrets.txt")
        assert exc.value.status_code == 400

    def test_rejects_parent_escape(self):
        with pytest.raises(HTTPException) as exc:
            _validate_sandbox_path("/home/user/../other/path")
        assert exc.value.status_code == 400


class TestSkillOwnershipAuthz:
    """Tests for dynamic skill ownership enforcement."""

    def test_allows_builtin(self):
        skill_def = SimpleNamespace(is_builtin=True, author="user_a")
        _enforce_skill_ownership_or_builtin(skill_def, "user_b", "skill_x")

    def test_allows_owner_for_dynamic_skill(self):
        skill_def = SimpleNamespace(is_builtin=False, author="user_a")
        _enforce_skill_ownership_or_builtin(skill_def, "user_a", "skill_x")

    def test_rejects_non_owner_for_dynamic_skill(self):
        skill_def = SimpleNamespace(is_builtin=False, author="user_a")
        with pytest.raises(HTTPException) as exc:
            _enforce_skill_ownership_or_builtin(skill_def, "user_b", "skill_x")
        assert exc.value.status_code == 403

    @pytest.mark.asyncio
    async def test_get_skill_endpoint_denies_non_owner(self):
        skill_id = "private_skill"
        current_user = SimpleNamespace(id="user_b")
        skill_def = SimpleNamespace(
            id=skill_id,
            is_builtin=False,
            author="user_a",
            source_code="class X: pass",
            created_at=None,
            updated_at=None,
        )
        loaded_skill = SimpleNamespace(
            metadata=SimpleNamespace(
                id=skill_id,
                name="Private Skill",
                version="1.0.0",
                description="desc",
                category="code",
                parameters=[],
                output_schema={},
                tags=[],
                enabled=True,
                required_tools=[],
                risk_level=None,
            )
        )

        class _Result:
            def scalar_one_or_none(self):
                return skill_def

        class _DB:
            async def execute(self, *_args, **_kwargs):
                return _Result()

        with patch("app.api.skills.skill_registry.get_skill", return_value=loaded_skill):
            with pytest.raises(HTTPException) as exc:
                await get_skill(skill_id=skill_id, current_user=current_user, db=_DB())
        assert exc.value.status_code == 403

    @pytest.mark.asyncio
    async def test_update_skill_endpoint_denies_non_owner(self):
        skill_id = "private_skill"
        current_user = SimpleNamespace(id="user_b")
        skill_def = SimpleNamespace(
            id=skill_id,
            is_builtin=False,
            author="user_a",
            metadata_json="{}",
        )

        class _Result:
            def scalar_one_or_none(self):
                return skill_def

        class _DB:
            async def execute(self, *_args, **_kwargs):
                return _Result()

        with pytest.raises(HTTPException) as exc:
            await update_skill(
                skill_id=skill_id,
                request=UpdateSkillRequest(description="new"),
                current_user=current_user,
                db=_DB(),
            )
        assert exc.value.status_code == 403


class TestSkillRiskPolicy:
    """Tests for skill-level HITL risk logic."""

    def test_unknown_skill_fails_closed_as_high_risk(self):
        with patch("app.services.skill_registry.skill_registry.get_skill", return_value=None):
            assert get_skill_risk_level("does_not_exist") == ToolRiskLevel.HIGH

    def test_skill_risk_inferred_from_required_tools(self):
        fake_skill = SimpleNamespace(metadata=SimpleNamespace(required_tools=["web_search"]))
        with patch("app.services.skill_registry.skill_registry.get_skill", return_value=fake_skill):
            assert get_skill_risk_level("web_research") == ToolRiskLevel.LOW

        fake_skill = SimpleNamespace(metadata=SimpleNamespace(required_tools=["execute_code"]))
        with patch("app.services.skill_registry.skill_registry.get_skill", return_value=fake_skill):
            assert get_skill_risk_level("data_analysis") == ToolRiskLevel.HIGH

    def test_requires_approval_for_high_risk_skill(self):
        fake_skill = SimpleNamespace(metadata=SimpleNamespace(required_tools=["execute_code"]))
        with patch("app.services.skill_registry.skill_registry.get_skill", return_value=fake_skill):
            assert requires_approval_for_skill("data_analysis", hitl_enabled=True) is True
            assert (
                requires_approval_for_skill(
                    "data_analysis",
                    hitl_enabled=True,
                    auto_approve_tools=["invoke_skill:data_analysis"],
                )
                is False
            )


class TestSkillMetadataValidation:
    """Tests for required_tools/risk_level registration validation."""

    def test_rejects_unknown_required_tools(self):
        with patch("app.agents.tools.get_all_tools", return_value=[]):
            with pytest.raises(HTTPException) as exc:
                _validate_skill_tool_metadata(["no_such_tool"], None)
        assert exc.value.status_code == 400

    def test_rejects_understated_risk_level(self):
        fake_tool = SimpleNamespace(name="execute_code")
        with patch("app.agents.tools.get_all_tools", return_value=[fake_tool]):
            with pytest.raises(HTTPException) as exc:
                _validate_skill_tool_metadata(["execute_code"], "low")
        assert exc.value.status_code == 400


class TestDynamicSkillLoaderBootstrap:
    """Tests for dynamic skill execution namespace bootstrap primitives."""

    def test_safe_namespace_includes_class_bootstrap_builtins(self):
        namespace = _create_safe_namespace()
        builtins = namespace.get("__builtins__", {})
        assert "__build_class__" in builtins
        assert "object" in builtins
        assert "type" in builtins


class TestToolHookGuardrails:
    """Tests for guardrail coverage in research/canonical execution hooks."""

    @pytest.mark.asyncio
    async def test_research_hook_blocks_unsafe_url(self):
        hooks = ResearchToolHooks()
        ctx = ToolExecutionContext(
            tool_name="browser_navigate",
            tool_args={"url": "http://127.0.0.1:8000"},
            tool_call_id="tc_1",
            tool=object(),
        )
        result = await hooks.before_execution(ctx)
        assert result is not None
        assert result.pending_interrupt is not None

    @pytest.mark.asyncio
    async def test_canonical_hook_blocks_unsafe_code(self):
        hooks = CanonicalToolHooks()
        ctx = ToolExecutionContext(
            tool_name="execute_code",
            tool_args={"code": "rm -rf /"},
            tool_call_id="tc_2",
            tool=object(),
        )
        result = await hooks.before_execution(ctx)
        assert result is not None
        assert result.is_error is True
        assert result.message is not None
        assert "Tool blocked:" in result.message.content


class TestPolicyCoverage:
    """Tests for policy/tool contract coverage across registry tools."""

    def test_tool_contracts_cover_registry(self):
        validate_tool_contracts()

    def test_policy_engine_covers_all_tools(self):
        engine = get_policy_engine()
        tools = get_all_tools()
        assert tools, "Expected at least one registered tool"
        for tool in tools:
            decision = engine.decide(
                PolicyInput(
                    tool_name=tool.name,
                    tool_args={},
                    hitl_enabled=True,
                    risk_threshold="high",
                )
            )
            assert decision.decision.value in {"allow", "require_approval", "deny"}
