"""Tests for cross-sandbox handoff artifact transfer."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.tools.handoff import HandoffInfo, build_query_with_context
from app.sandbox.artifact_transfer import (
    DEFAULT_ARTIFACT_PATTERNS,
    MAX_ARTIFACT_FILES,
    MAX_ARTIFACT_SIZE_BYTES,
    cleanup_artifacts,
    collect_artifacts,
    format_artifact_summary,
    restore_artifacts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STORAGE_PATCH = "app.services.file_storage.file_storage_service"


def _make_runtime(
    find_stdout: str = "",
    find_exit: int = 0,
    stat_stdout: str = "100",
    stat_exit: int = 0,
    read_content: bytes = b"hello world",
) -> AsyncMock:
    """Create a mock SandboxRuntime."""
    runtime = AsyncMock()
    runtime.sandbox_id = "sbx-test-123"

    async def _run_command(cmd: str, timeout: int = 60, cwd: str | None = None):
        if cmd.startswith("find "):
            return SimpleNamespace(exit_code=find_exit, stdout=find_stdout, stderr="")
        if "stat " in cmd:
            return SimpleNamespace(exit_code=stat_exit, stdout=stat_stdout, stderr="")
        if cmd.startswith("mkdir "):
            return SimpleNamespace(exit_code=0, stdout="", stderr="")
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    runtime.run_command = AsyncMock(side_effect=_run_command)
    runtime.read_file = AsyncMock(return_value=read_content)
    runtime.write_file = AsyncMock()
    return runtime


def _mock_storage_service(backend: str = "local") -> MagicMock:
    """Create a mock FileStorageService."""
    svc = MagicMock()
    svc.backend = backend
    svc.upload_file = AsyncMock(return_value={"storage_key": "test"})
    svc.download_file = AsyncMock()
    svc.delete_file = AsyncMock(return_value=True)
    return svc


# ---------------------------------------------------------------------------
# HandoffInfo type
# ---------------------------------------------------------------------------


class TestHandoffInfoArtifacts:
    """Test that handoff_artifacts field works in HandoffInfo."""

    def test_handoff_info_without_artifacts(self):
        info: HandoffInfo = {
            "source_agent": "task",
            "target_agent": "research",
            "task_description": "do research",
            "context": "",
        }
        assert "handoff_artifacts" not in info

    def test_handoff_info_with_artifacts(self):
        info: HandoffInfo = {
            "source_agent": "task",
            "target_agent": "research",
            "task_description": "do research",
            "context": "",
            "handoff_artifacts": [
                {"path": "/home/user/data.csv", "storage_key": "abc/data.csv", "size": 1024},
            ],
        }
        assert len(info["handoff_artifacts"]) == 1
        assert info["handoff_artifacts"][0]["path"] == "/home/user/data.csv"


# ---------------------------------------------------------------------------
# collect_artifacts
# ---------------------------------------------------------------------------


class TestCollectArtifacts:
    """Tests for artifact collection from sandbox."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_files_found(self):
        runtime = _make_runtime(find_stdout="", find_exit=1)
        result = await collect_artifacts(runtime)
        assert result == []

    @pytest.mark.asyncio
    async def test_collects_matching_files(self):
        runtime = _make_runtime(
            find_stdout="/home/user/script.py\n/home/user/data.csv\n",
            stat_stdout="256",
            read_content=b"file content here",
        )

        mock_storage = _mock_storage_service()

        with patch(_STORAGE_PATCH, mock_storage), \
             patch("app.sandbox.artifact_transfer.uuid") as mock_uuid, \
             patch("app.config.settings") as mock_settings:
            mock_uuid.uuid4.return_value = MagicMock(hex="abc123")
            mock_settings.local_storage_path = "/tmp/test_storage"

            # Mock file I/O for local storage upload
            with patch("builtins.open", MagicMock()), \
                 patch("os.makedirs", MagicMock()):
                # Patch pathlib for mkdir
                with patch("pathlib.Path.mkdir", MagicMock()):
                    result = await collect_artifacts(runtime)

        assert len(result) == 2
        assert result[0]["path"] == "/home/user/script.py"
        assert result[0]["size"] == 256
        assert "storage_key" in result[0]

    @pytest.mark.asyncio
    async def test_respects_max_files_limit(self):
        paths = "\n".join(f"/home/user/file{i}.py" for i in range(25))
        runtime = _make_runtime(find_stdout=paths, stat_stdout="100")

        mock_storage = _mock_storage_service()

        with patch(_STORAGE_PATCH, mock_storage), \
             patch("app.sandbox.artifact_transfer.uuid") as mock_uuid, \
             patch("app.config.settings") as mock_settings, \
             patch("builtins.open", MagicMock()), \
             patch("pathlib.Path.mkdir", MagicMock()):
            mock_uuid.uuid4.return_value = MagicMock(hex="abc123")
            mock_settings.local_storage_path = "/tmp/test_storage"

            result = await collect_artifacts(runtime, max_files=5)

        assert len(result) <= 5

    @pytest.mark.asyncio
    async def test_respects_size_budget(self):
        runtime = _make_runtime(
            find_stdout="/home/user/big.csv\n",
            stat_stdout=str(MAX_ARTIFACT_SIZE_BYTES + 1),
        )

        result = await collect_artifacts(runtime)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_graceful_on_find_failure(self):
        runtime = AsyncMock()
        runtime.sandbox_id = "sbx-broken"
        runtime.run_command = AsyncMock(side_effect=Exception("sandbox down"))

        result = await collect_artifacts(runtime)
        assert result == []


# ---------------------------------------------------------------------------
# restore_artifacts
# ---------------------------------------------------------------------------


class TestRestoreArtifacts:
    """Tests for artifact restoration to sandbox."""

    @pytest.mark.asyncio
    async def test_restores_files_to_sandbox(self):
        runtime = _make_runtime()
        artifacts = [
            {"path": "/home/user/data.csv", "storage_key": "handoff_artifacts/abc/data.csv", "size": 100},
        ]

        mock_storage = _mock_storage_service()

        with patch(_STORAGE_PATCH, mock_storage), \
             patch("app.config.settings") as mock_settings:
            mock_settings.local_storage_path = "/tmp/test_storage"

            with patch("pathlib.Path.exists", return_value=True), \
                 patch("pathlib.Path.read_bytes", return_value=b"csv data"):
                restored = await restore_artifacts(runtime, artifacts)

        assert len(restored) == 1
        assert restored[0] == "/home/user/data.csv"
        runtime.write_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_missing_artifacts(self):
        runtime = _make_runtime()
        artifacts = [
            {"path": "/home/user/missing.csv", "storage_key": "handoff_artifacts/xyz/missing.csv", "size": 50},
        ]

        mock_storage = _mock_storage_service()

        with patch(_STORAGE_PATCH, mock_storage), \
             patch("app.config.settings") as mock_settings:
            mock_settings.local_storage_path = "/tmp/test_storage"

            with patch("pathlib.Path.exists", return_value=False):
                restored = await restore_artifacts(runtime, artifacts)

        assert len(restored) == 0

    @pytest.mark.asyncio
    async def test_handles_empty_artifacts(self):
        runtime = _make_runtime()
        with patch(_STORAGE_PATCH, _mock_storage_service()):
            restored = await restore_artifacts(runtime, [])
        assert restored == []

    @pytest.mark.asyncio
    async def test_handles_malformed_artifact_entries(self):
        runtime = _make_runtime()
        artifacts = [
            {"path": "", "storage_key": "", "size": 0},
            {},
        ]

        with patch(_STORAGE_PATCH, _mock_storage_service()):
            restored = await restore_artifacts(runtime, artifacts)

        assert restored == []


# ---------------------------------------------------------------------------
# cleanup_artifacts
# ---------------------------------------------------------------------------


class TestCleanupArtifacts:
    """Tests for artifact cleanup from storage."""

    @pytest.mark.asyncio
    async def test_cleans_up_local_files(self):
        artifacts = [
            {"path": "/home/user/data.csv", "storage_key": "handoff_artifacts/abc/data.csv", "size": 100},
        ]

        mock_storage = _mock_storage_service()

        with patch(_STORAGE_PATCH, mock_storage), \
             patch("app.config.settings") as mock_settings, \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.unlink") as mock_unlink:
            mock_settings.local_storage_path = "/tmp/test_storage"
            count = await cleanup_artifacts(artifacts)

        assert count == 1
        mock_unlink.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_empty_artifact_list(self):
        count = await cleanup_artifacts([])
        assert count == 0


# ---------------------------------------------------------------------------
# format_artifact_summary
# ---------------------------------------------------------------------------


class TestFormatArtifactSummary:
    """Tests for artifact summary formatting."""

    def test_empty_artifacts(self):
        assert format_artifact_summary([]) == ""

    def test_single_artifact(self):
        artifacts = [{"path": "/home/user/data.csv", "storage_key": "abc", "size": 1024}]
        summary = format_artifact_summary(artifacts)
        assert "1 file(s)" in summary
        assert "data.csv" in summary
        assert "1,024 bytes" in summary

    def test_multiple_artifacts(self):
        artifacts = [
            {"path": "/home/user/script.py", "storage_key": "abc", "size": 500},
            {"path": "/home/user/data.json", "storage_key": "def", "size": 2000},
        ]
        summary = format_artifact_summary(artifacts)
        assert "2 file(s)" in summary
        assert "script.py" in summary
        assert "data.json" in summary


# ---------------------------------------------------------------------------
# build_query_with_context integration
# ---------------------------------------------------------------------------


class TestBuildQueryWithArtifactSummary:
    """Test that artifact summary is included in query context."""

    def test_query_without_artifact_summary(self):
        result = build_query_with_context(query="hello")
        assert result == "hello"

    def test_query_with_artifact_summary(self):
        result = build_query_with_context(
            query="analyze this",
            delegated_task="Analyze the dataset",
            artifact_summary="Transferred 1 file(s):\n  - /home/user/data.csv",
        )
        assert "Analyze the dataset" in result
        assert "Transferred 1 file(s)" in result
        assert "data.csv" in result

    def test_artifact_summary_appended_after_context(self):
        result = build_query_with_context(
            query="base",
            delegated_task="task",
            handoff_context="some context",
            artifact_summary="files transferred",
        )
        assert "task" in result
        assert "some context" in result
        assert "files transferred" in result
