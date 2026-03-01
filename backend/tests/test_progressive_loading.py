"""Tests for progressive skill loading in SkillRegistry."""

from unittest.mock import MagicMock, patch

import pytest

from app.agents.skills.skill_base import Skill, SkillMetadata, SkillParameter, ToolSkill
from app.services.skill_registry import SkillLoadLevel, SkillRegistry


# ---------------------------------------------------------------------------
# Helpers: lightweight stub skill classes for testing
# ---------------------------------------------------------------------------


class _StubSkillA(ToolSkill):
    """Minimal stub skill for testing."""

    metadata = SkillMetadata(
        id="stub_a",
        name="Stub A",
        version="1.0.0",
        description="A test stub skill",
        category="code",
        parameters=[
            SkillParameter(
                name="query",
                type="string",
                description="Test query",
                required=True,
            )
        ],
        output_schema={"type": "object"},
        risk_level="low",
        tags=["test"],
        enabled=True,
    )

    async def execute(self, params, context):
        return {"result": "ok"}


class _StubSkillB(ToolSkill):
    """Second stub skill with a different category."""

    metadata = SkillMetadata(
        id="stub_b",
        name="Stub B",
        version="1.0.0",
        description="Another test stub skill",
        category="research",
        parameters=[],
        output_schema={"type": "object"},
        risk_level="low",
        tags=["test"],
        enabled=True,
    )

    async def execute(self, params, context):
        return {"result": "ok"}


class _DisabledStubSkill(ToolSkill):
    """Stub skill that is disabled."""

    metadata = SkillMetadata(
        id="stub_disabled",
        name="Disabled Stub",
        version="1.0.0",
        description="Disabled stub",
        category="code",
        parameters=[],
        output_schema={"type": "object"},
        risk_level="low",
        tags=[],
        enabled=False,
    )

    async def execute(self, params, context):
        return {}


# ---------------------------------------------------------------------------
# Tests: SkillLoadLevel enum
# ---------------------------------------------------------------------------


class TestSkillLoadLevel:
    """Tests for the SkillLoadLevel enum values and ordering."""

    def test_metadata_value(self):
        assert SkillLoadLevel.METADATA == 1

    def test_instructions_value(self):
        assert SkillLoadLevel.INSTRUCTIONS == 2

    def test_resources_value(self):
        assert SkillLoadLevel.RESOURCES == 3

    def test_ordering(self):
        assert SkillLoadLevel.METADATA < SkillLoadLevel.INSTRUCTIONS
        assert SkillLoadLevel.INSTRUCTIONS < SkillLoadLevel.RESOURCES

    def test_is_int_subclass(self):
        """SkillLoadLevel values should be usable as ints."""
        assert isinstance(SkillLoadLevel.METADATA, int)


# ---------------------------------------------------------------------------
# Tests: SkillRegistry with progressive loading
# ---------------------------------------------------------------------------


def _make_registry_with_stubs() -> SkillRegistry:
    """Create a SkillRegistry pre-populated at Level 1 with stub skills.

    This simulates what ``_register_builtin_skills`` does at startup:
    metadata is cached, skill classes are registered, but no full Skill
    instances live in ``_loaded_skills``.
    """
    registry = SkillRegistry()

    for skill_class in [_StubSkillA, _StubSkillB, _DisabledStubSkill]:
        skill = skill_class()
        metadata = skill.metadata
        registry._builtin_skills[metadata.id] = skill_class
        registry._skill_metadata_cache[metadata.id] = metadata
        registry._skill_load_levels[metadata.id] = SkillLoadLevel.METADATA

    return registry


class TestListSkillsWithMetadataCache:
    """list_skills should use the metadata cache (Level 1)."""

    def test_list_all_enabled(self):
        registry = _make_registry_with_stubs()
        skills = registry.list_skills()
        ids = {s.id for s in skills}
        assert "stub_a" in ids
        assert "stub_b" in ids
        # Disabled skill should be excluded
        assert "stub_disabled" not in ids

    def test_list_all_including_disabled(self):
        registry = _make_registry_with_stubs()
        skills = registry.list_skills(enabled_only=False)
        ids = {s.id for s in skills}
        assert "stub_disabled" in ids

    def test_list_by_category(self):
        registry = _make_registry_with_stubs()
        skills = registry.list_skills(category="research")
        ids = {s.id for s in skills}
        assert ids == {"stub_b"}

    def test_list_by_nonexistent_category(self):
        registry = _make_registry_with_stubs()
        skills = registry.list_skills(category="nonexistent")
        assert skills == []

    def test_list_does_not_trigger_full_loading(self):
        """Listing skills should NOT populate _loaded_skills."""
        registry = _make_registry_with_stubs()
        registry.list_skills()
        assert len(registry._loaded_skills) == 0


class TestGetSkillAutoPromotes:
    """get_skill should auto-promote from Level 1 to Level 2."""

    def test_auto_promotes_builtin(self):
        registry = _make_registry_with_stubs()
        # Initially no full skills loaded
        assert len(registry._loaded_skills) == 0

        skill = registry.get_skill("stub_a")
        assert skill is not None
        assert isinstance(skill, _StubSkillA)
        # Should now be at Level 2
        assert registry._skill_load_levels["stub_a"] == SkillLoadLevel.INSTRUCTIONS
        assert "stub_a" in registry._loaded_skills

    def test_returns_cached_if_already_loaded(self):
        registry = _make_registry_with_stubs()
        # First call loads it
        skill1 = registry.get_skill("stub_a")
        # Second call returns the same instance from cache
        skill2 = registry.get_skill("stub_a")
        assert skill1 is skill2

    def test_returns_none_for_unknown_skill(self):
        registry = _make_registry_with_stubs()
        assert registry.get_skill("nonexistent") is None

    def test_only_promotes_requested_skill(self):
        """Getting one skill should not load others."""
        registry = _make_registry_with_stubs()
        registry.get_skill("stub_a")
        assert "stub_a" in registry._loaded_skills
        assert "stub_b" not in registry._loaded_skills
        assert registry._skill_load_levels["stub_b"] == SkillLoadLevel.METADATA


class TestEnsureLoaded:
    """Tests for the ensure_loaded async method."""

    @pytest.mark.asyncio
    async def test_promote_to_level2(self):
        registry = _make_registry_with_stubs()
        skill = await registry.ensure_loaded("stub_a", SkillLoadLevel.INSTRUCTIONS)
        assert skill is not None
        assert isinstance(skill, _StubSkillA)
        assert registry._skill_load_levels["stub_a"] == SkillLoadLevel.INSTRUCTIONS

    @pytest.mark.asyncio
    async def test_promote_to_level3(self):
        registry = _make_registry_with_stubs()
        skill = await registry.ensure_loaded("stub_a", SkillLoadLevel.RESOURCES)
        assert skill is not None
        assert registry._skill_load_levels["stub_a"] == SkillLoadLevel.RESOURCES

    @pytest.mark.asyncio
    async def test_noop_when_already_at_level(self):
        registry = _make_registry_with_stubs()
        # First load to Level 2
        skill1 = await registry.ensure_loaded("stub_a", SkillLoadLevel.INSTRUCTIONS)
        # Second call should be a no-op and return the same instance
        skill2 = await registry.ensure_loaded("stub_a", SkillLoadLevel.INSTRUCTIONS)
        assert skill1 is skill2

    @pytest.mark.asyncio
    async def test_noop_for_metadata_level(self):
        """Requesting Level 1 should be a no-op since it is the startup level."""
        registry = _make_registry_with_stubs()
        # Level 1 is already loaded; ensure_loaded should return None
        # because the skill is not in _loaded_skills at Level 1
        result = await registry.ensure_loaded("stub_a", SkillLoadLevel.METADATA)
        # current_level (1) >= requested level (1), returns from _loaded_skills
        # which is empty at Level 1, so result is None
        assert result is None
        # But the skill should still be in metadata cache
        assert "stub_a" in registry._skill_metadata_cache

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown(self):
        registry = _make_registry_with_stubs()
        result = await registry.ensure_loaded("nonexistent", SkillLoadLevel.INSTRUCTIONS)
        assert result is None


class TestLoadSkillFull:
    """Tests for _load_skill_full private method."""

    def test_instantiates_builtin_class(self):
        registry = _make_registry_with_stubs()
        skill = registry._load_skill_full("stub_a")
        assert skill is not None
        assert isinstance(skill, _StubSkillA)
        assert skill.metadata.id == "stub_a"

    def test_returns_none_for_non_builtin(self):
        registry = _make_registry_with_stubs()
        # Add metadata but no builtin class
        registry._skill_metadata_cache["dynamic_only"] = _StubSkillA.metadata
        result = registry._load_skill_full("dynamic_only")
        assert result is None

    def test_returns_none_on_instantiation_error(self):
        registry = SkillRegistry()

        class _BrokenSkill(ToolSkill):
            def __init__(self):
                raise RuntimeError("Broken!")

            metadata = SkillMetadata(
                id="broken",
                name="Broken",
                version="1.0.0",
                description="Broken skill",
                category="code",
                parameters=[],
                output_schema={},
                risk_level="low",
            )

        registry._builtin_skills["broken"] = _BrokenSkill
        result = registry._load_skill_full("broken")
        assert result is None


class TestGetLoadStats:
    """Tests for get_load_stats."""

    def test_all_at_metadata_level(self):
        registry = _make_registry_with_stubs()
        stats = registry.get_load_stats()
        assert stats["total_skills"] == 3  # stub_a, stub_b, stub_disabled
        assert stats["loaded_full"] == 0
        assert stats["by_level"]["metadata"] == 3

    def test_mixed_levels(self):
        registry = _make_registry_with_stubs()
        # Promote one skill to Level 2
        registry.get_skill("stub_a")
        stats = registry.get_load_stats()
        assert stats["total_skills"] == 3
        assert stats["loaded_full"] == 1
        assert stats["by_level"]["instructions"] == 1
        assert stats["by_level"]["metadata"] == 2

    def test_empty_registry(self):
        registry = SkillRegistry()
        stats = registry.get_load_stats()
        assert stats["total_skills"] == 0
        assert stats["loaded_full"] == 0
        assert stats["by_level"] == {}


class TestUnloadSkill:
    """Tests for unload_skill with progressive loading."""

    def test_unload_demotes_to_level1(self):
        registry = _make_registry_with_stubs()
        # Load to Level 2
        registry.get_skill("stub_a")
        assert "stub_a" in registry._loaded_skills

        # Unload
        result = registry.unload_skill("stub_a")
        assert result is True
        assert "stub_a" not in registry._loaded_skills
        # Should be demoted back to Level 1
        assert registry._skill_load_levels["stub_a"] == SkillLoadLevel.METADATA
        # Metadata should still be cached
        assert "stub_a" in registry._skill_metadata_cache

    def test_unload_nonexistent_returns_false(self):
        registry = _make_registry_with_stubs()
        assert registry.unload_skill("nonexistent") is False

    def test_can_reload_after_unload(self):
        registry = _make_registry_with_stubs()
        # Load, unload, then reload
        skill1 = registry.get_skill("stub_a")
        registry.unload_skill("stub_a")
        skill2 = registry.get_skill("stub_a")
        assert skill2 is not None
        assert isinstance(skill2, _StubSkillA)
        # Should be a new instance
        assert skill1 is not skill2


class TestCategoryFiltering:
    """Tests for list_skills category filtering with metadata cache."""

    def test_filter_code_category(self):
        registry = _make_registry_with_stubs()
        skills = registry.list_skills(category="code")
        ids = {s.id for s in skills}
        # stub_a is code, stub_disabled is code but disabled
        assert ids == {"stub_a"}

    def test_filter_research_category(self):
        registry = _make_registry_with_stubs()
        skills = registry.list_skills(category="research")
        ids = {s.id for s in skills}
        assert ids == {"stub_b"}

    def test_filter_code_including_disabled(self):
        registry = _make_registry_with_stubs()
        skills = registry.list_skills(category="code", enabled_only=False)
        ids = {s.id for s in skills}
        assert ids == {"stub_a", "stub_disabled"}
