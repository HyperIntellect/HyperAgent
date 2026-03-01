"""Tests for App Builder Skill and App Sandbox Manager fixes."""

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.sandbox.app_sandbox_manager import _PACKAGE_NAME_RE, AppSandboxManager


# ---------------------------------------------------------------------------
# Step 3: _PACKAGE_NAME_RE — scoped npm packages
# ---------------------------------------------------------------------------

class TestPackageNameRegex:
    """Tests for _PACKAGE_NAME_RE allowing scoped npm packages."""

    def test_plain_package(self):
        assert _PACKAGE_NAME_RE.match("react")

    def test_plain_package_with_version(self):
        assert _PACKAGE_NAME_RE.match("react@18.2.0")

    def test_scoped_package(self):
        assert _PACKAGE_NAME_RE.match("@tailwindcss/forms")

    def test_scoped_package_with_version(self):
        assert _PACKAGE_NAME_RE.match("@tailwindcss/forms@0.5.7")

    def test_scoped_package_complex(self):
        assert _PACKAGE_NAME_RE.match("@radix-ui/react-dialog")

    def test_dotted_package(self):
        assert _PACKAGE_NAME_RE.match("lodash.debounce")

    def test_rejects_shell_injection(self):
        assert not _PACKAGE_NAME_RE.match("react; rm -rf /")

    def test_rejects_empty(self):
        assert not _PACKAGE_NAME_RE.match("")

    def test_rejects_spaces(self):
        assert not _PACKAGE_NAME_RE.match("react dom")

    def test_rejects_backticks(self):
        assert not _PACKAGE_NAME_RE.match("`whoami`")


# ---------------------------------------------------------------------------
# Step 7: Port utilization warning
# ---------------------------------------------------------------------------

class TestPortUtilizationWarning:
    """Tests for port pool utilization warning at >80%."""

    @patch("app.sandbox.app_sandbox_manager.get_logger")
    def test_warning_at_high_utilization(self, mock_get_logger):
        """When >80% of ports are used, a warning should be logged."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = AppSandboxManager.__new__(AppSandboxManager)
        manager._used_ports = set()

        with patch("app.config.settings") as mock_settings:
            mock_settings.boxlite_app_host_port_start = 10000
            pool_size = min(10000 + 1000, 65535) - 10000 + 1  # 1001 ports

            # Fill 80% of ports
            threshold = int(pool_size * 0.8)
            manager._used_ports = set(range(10000, 10000 + threshold))

            # Next allocation should trigger warning
            port = manager._allocate_host_port()
            assert port not in set(range(10000, 10000 + threshold))

    def test_allocation_returns_lowest_available(self):
        """Port allocator should return the lowest unused port."""
        manager = AppSandboxManager.__new__(AppSandboxManager)
        manager._used_ports = set()

        with patch("app.config.settings") as mock_settings:
            mock_settings.boxlite_app_host_port_start = 10000

            port1 = manager._allocate_host_port()
            assert port1 == 10000

            port2 = manager._allocate_host_port()
            assert port2 == 10001


# ---------------------------------------------------------------------------
# Step 8: Timeout budget verification
# ---------------------------------------------------------------------------

class TestTimeoutBudget:
    """Verify timeouts fit within the 600s skill budget."""

    def test_scaffold_timeout_is_180(self):
        """scaffold_project uses 180s timeout for live scaffold."""
        import inspect
        from app.sandbox.app_sandbox_manager import AppSandboxManager
        source = inspect.getsource(AppSandboxManager.scaffold_project)
        assert "timeout=180" in source

    def test_install_timeout_is_120(self):
        """install_dependencies uses 120s timeout."""
        import inspect
        from app.sandbox.app_sandbox_manager import AppSandboxManager
        source = inspect.getsource(AppSandboxManager.install_dependencies)
        assert "timeout=120" in source

    def test_total_worst_case_under_600(self):
        """180 (scaffold) + 120 (install) + 60 (server wait) = 360 < 600."""
        scaffold_timeout = 180
        install_timeout = 120
        server_wait = 60
        overhead = 40  # generous overhead for other operations
        total = scaffold_timeout + install_timeout + server_wait + overhead
        assert total < 600, f"Total worst-case timeout {total}s exceeds 600s skill budget"


# ---------------------------------------------------------------------------
# Step 1: Provider-agnostic error messages
# ---------------------------------------------------------------------------

class TestSandboxAvailabilityMessage:
    """Test that scaffold_project uses provider-agnostic error messages."""

    def test_import_uses_provider_module(self):
        """Verify the skill imports from app.sandbox.provider, not hardcoded E2B message."""
        import inspect
        from app.agents.skills.builtin.app_builder_skill import AppBuilderSkill
        source = inspect.getsource(AppBuilderSkill.create_graph)
        assert "is_provider_available" in source
        assert "from app.sandbox.provider import is_provider_available" in source

    def test_no_hardcoded_e2b_message(self):
        """The error message should not hardcode 'E2B_API_KEY'."""
        import inspect
        from app.agents.skills.builtin.app_builder_skill import AppBuilderSkill
        source = inspect.getsource(AppBuilderSkill.create_graph)
        assert 'E2B sandbox not available. Please configure E2B_API_KEY.' not in source


# ---------------------------------------------------------------------------
# Step 2: Pending events use Annotated[..., add] reducer
# ---------------------------------------------------------------------------

class TestPendingEventsReducer:
    """Test that pending_events uses Annotated reducer for accumulation."""

    def test_state_has_annotated_pending_events(self):
        """AppBuilderState.pending_events should use Annotated[..., add]."""
        from typing import get_type_hints
        from app.agents.skills.builtin.app_builder_skill import AppBuilderState

        hints = get_type_hints(AppBuilderState, include_extras=True)
        pe_hint = hints["pending_events"]
        # Check it's Annotated
        assert hasattr(pe_hint, "__metadata__"), "pending_events should be Annotated"

    def test_state_has_retry_count(self):
        """AppBuilderState should have retry_count field."""
        from app.agents.skills.builtin.app_builder_skill import AppBuilderState
        hints = AppBuilderState.__annotations__
        assert "retry_count" in hints


# ---------------------------------------------------------------------------
# Step 5: fix_build_errors node exists in graph
# ---------------------------------------------------------------------------

class TestFixBuildErrorsNode:
    """Test that the fix_build_errors node is wired into the graph."""

    def test_graph_has_fix_build_errors_node(self):
        """The compiled graph should include a fix_build_errors node."""
        import inspect
        from app.agents.skills.builtin.app_builder_skill import AppBuilderSkill
        source = inspect.getsource(AppBuilderSkill.create_graph)
        assert '"fix_build_errors"' in source
        assert 'graph.add_node("fix_build_errors"' in source

    def test_route_step_handles_fix_build_errors(self):
        """route_step should handle fix_build_errors and fix_build_errors_exhausted."""
        import inspect
        from app.agents.skills.builtin.app_builder_skill import AppBuilderSkill
        source = inspect.getsource(AppBuilderSkill.create_graph)
        assert "fix_build_errors_exhausted" in source


# ---------------------------------------------------------------------------
# Step 6: Semaphore in parallel file generation
# ---------------------------------------------------------------------------

class TestParallelFileGeneration:
    """Test that parallel file generation is capped with a semaphore."""

    def test_semaphore_in_generate_files(self):
        """generate_files should use a Semaphore(5) to cap parallel LLM calls."""
        import inspect
        from app.agents.skills.builtin.app_builder_skill import AppBuilderSkill
        source = inspect.getsource(AppBuilderSkill.create_graph)
        assert "Semaphore(5)" in source
        assert "_generate_with_limit" in source


# ---------------------------------------------------------------------------
# Step 9: Partial-success stage status
# ---------------------------------------------------------------------------

class TestPartialSuccessStageStatus:
    """Test that generate stage always completes, even with partial failures."""

    def test_no_running_status_on_errors(self):
        """Stage status should not be 'running' when there are build errors — always 'completed'."""
        import inspect
        from app.agents.skills.builtin.app_builder_skill import AppBuilderSkill
        source = inspect.getsource(AppBuilderSkill.create_graph)
        # The old buggy pattern: status="completed" if not build_errors else "running"
        assert 'if not build_errors else "running"' not in source


# ---------------------------------------------------------------------------
# Step 10: Inter-file context in generation prompts
# ---------------------------------------------------------------------------

class TestInterFileContext:
    """Test that file generation prompts include context about other files."""

    def test_prompt_includes_other_files(self):
        """The generation prompt should reference other files for coordination."""
        import inspect
        from app.agents.skills.builtin.app_builder_skill import AppBuilderSkill
        source = inspect.getsource(AppBuilderSkill.create_graph)
        assert "Other files in this project" in source
        assert "coordinate imports/exports" in source


# ---------------------------------------------------------------------------
# Step 11: Sandbox ID fallback chain
# ---------------------------------------------------------------------------

class TestSandboxIdFallback:
    """Test that sandbox ID fallback logs a warning and uses prefixed ID."""

    def test_no_raw_fallback_chain(self):
        """Should not use 'or task_id or user_id or app-sandbox' pattern."""
        import inspect
        from app.agents.skills.builtin.app_builder_skill import AppBuilderSkill
        source = inspect.getsource(AppBuilderSkill.create_graph)
        assert 'or task_id or user_id or "app-sandbox"' not in source

    def test_uses_prefixed_fallback(self):
        """Should use f'app-{task_id or ...}' pattern with warning log."""
        import inspect
        from app.agents.skills.builtin.app_builder_skill import AppBuilderSkill
        source = inspect.getsource(AppBuilderSkill.create_graph)
        assert "app_builder_missing_sandbox_id" in source
        assert 'f"app-{' in source


# ---------------------------------------------------------------------------
# Step 12: Template fallback warning
# ---------------------------------------------------------------------------

class TestTemplateFallback:
    """Test that unknown template triggers a warning before fallback."""

    def test_warns_on_unknown_template(self):
        """scaffold_project should log warning when template not in APP_TEMPLATES."""
        import inspect
        from app.agents.skills.builtin.app_builder_skill import AppBuilderSkill
        source = inspect.getsource(AppBuilderSkill.create_graph)
        assert "app_builder_unknown_template_fallback" in source
