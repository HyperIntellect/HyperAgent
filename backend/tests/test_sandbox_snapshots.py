"""Tests for Persistent Sandbox Snapshots."""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.sandbox.runtime import CommandResult


def _make_mock_runtime(sandbox_id: str = "test-sandbox-123") -> MagicMock:
    """Create a mock SandboxRuntime with snapshot support."""
    runtime = MagicMock()
    runtime.sandbox_id = sandbox_id
    runtime.run_command = AsyncMock(
        return_value=CommandResult(exit_code=0, stdout="health_check\n", stderr="")
    )
    runtime.kill = AsyncMock()
    runtime.read_file = AsyncMock(return_value=b"fake tar content here")
    runtime.write_file = AsyncMock()
    runtime.get_host_url = AsyncMock(return_value="http://localhost:3000")

    # Snapshot methods return valid data
    runtime.save_snapshot = AsyncMock(return_value=b"fake-tar-gz-data-12345")
    runtime.restore_snapshot = AsyncMock(return_value=True)

    return runtime


class TestSandboxRuntimeProtocolSnapshot:
    """Tests for snapshot methods on SandboxRuntime protocol."""

    @pytest.mark.asyncio
    async def test_save_snapshot_returns_bytes(self):
        runtime = _make_mock_runtime()
        data = await runtime.save_snapshot(["/home/user"], "snap-001")
        assert isinstance(data, bytes)
        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_restore_snapshot_returns_bool(self):
        runtime = _make_mock_runtime()
        result = await runtime.restore_snapshot(b"fake-data", "/")
        assert result is True

    @pytest.mark.asyncio
    async def test_restore_snapshot_failure(self):
        runtime = _make_mock_runtime()
        runtime.restore_snapshot = AsyncMock(return_value=False)
        result = await runtime.restore_snapshot(b"bad-data", "/")
        assert result is False


class TestSnapshotService:
    """Tests for the snapshot service save/restore functions."""

    @pytest.mark.asyncio
    async def test_save_snapshot_success(self):
        """Test saving a snapshot to local storage."""
        runtime = _make_mock_runtime()

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("app.services.snapshot_service.settings") as mock_settings,
                patch("app.services.snapshot_service._record_snapshot", new_callable=AsyncMock) as mock_record,
            ):
                mock_settings.storage_backend = "local"
                mock_settings.local_storage_path = tmpdir
                mock_settings.snapshot_max_size_bytes = 100 * 1024 * 1024
                mock_settings.snapshot_retention_hours = 24
                mock_settings.snapshot_default_paths_execution = ["/home/user"]

                mock_record.return_value = {
                    "id": "test-id",
                    "user_id": "user1",
                    "task_id": "task1",
                    "sandbox_type": "execution",
                    "storage_key": "snapshots/user1/task1/test.tar.gz",
                    "paths_included": ["/home/user"],
                    "size_bytes": 21,
                }

                from app.services.snapshot_service import save_snapshot

                result = await save_snapshot(
                    runtime=runtime,
                    user_id="user1",
                    task_id="task1",
                    sandbox_type="execution",
                )

                assert result is not None
                runtime.save_snapshot.assert_called_once()
                mock_record.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_snapshot_too_large(self):
        """Test that oversized snapshots are rejected."""
        runtime = _make_mock_runtime()
        runtime.save_snapshot = AsyncMock(return_value=b"x" * 200)

        with patch("app.services.snapshot_service.settings") as mock_settings:
            mock_settings.snapshot_max_size_bytes = 100  # Very small limit
            mock_settings.snapshot_retention_hours = 24
            mock_settings.snapshot_default_paths_execution = ["/home/user"]

            from app.services.snapshot_service import save_snapshot

            result = await save_snapshot(
                runtime=runtime,
                user_id="user1",
                task_id="task1",
                sandbox_type="execution",
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_save_snapshot_empty(self):
        """Test that empty snapshots are skipped."""
        runtime = _make_mock_runtime()
        runtime.save_snapshot = AsyncMock(return_value=b"")

        with patch("app.services.snapshot_service.settings") as mock_settings:
            mock_settings.snapshot_max_size_bytes = 100 * 1024 * 1024
            mock_settings.snapshot_retention_hours = 24
            mock_settings.snapshot_default_paths_execution = ["/home/user"]

            from app.services.snapshot_service import save_snapshot

            result = await save_snapshot(
                runtime=runtime,
                user_id="user1",
                task_id="task1",
                sandbox_type="execution",
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_save_snapshot_runtime_error(self):
        """Test that runtime errors are handled gracefully."""
        runtime = _make_mock_runtime()
        runtime.save_snapshot = AsyncMock(side_effect=Exception("sandbox dead"))

        with patch("app.services.snapshot_service.settings") as mock_settings:
            mock_settings.snapshot_max_size_bytes = 100 * 1024 * 1024
            mock_settings.snapshot_retention_hours = 24
            mock_settings.snapshot_default_paths_execution = ["/home/user"]

            from app.services.snapshot_service import save_snapshot

            result = await save_snapshot(
                runtime=runtime,
                user_id="user1",
                task_id="task1",
                sandbox_type="execution",
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_restore_snapshot_success(self):
        """Test restoring a snapshot from storage."""
        runtime = _make_mock_runtime()
        snapshot_data = b"fake-tar-gz-data"

        with (
            patch(
                "app.services.snapshot_service._find_snapshot",
                new_callable=AsyncMock,
                return_value={
                    "id": "snap-123",
                    "user_id": "user1",
                    "task_id": "task1",
                    "sandbox_type": "execution",
                    "storage_key": "snapshots/user1/task1/snap.tar.gz",
                    "paths_included": ["/home/user"],
                    "size_bytes": len(snapshot_data),
                    "created_at": datetime.now(timezone.utc),
                    "expires_at": datetime.now(timezone.utc) + timedelta(hours=24),
                },
            ),
            patch(
                "app.services.snapshot_service._retrieve_snapshot",
                new_callable=AsyncMock,
                return_value=snapshot_data,
            ),
        ):
            from app.services.snapshot_service import restore_snapshot

            result = await restore_snapshot(
                runtime=runtime,
                user_id="user1",
                task_id="task1",
                sandbox_type="execution",
            )

            assert result is True
            runtime.restore_snapshot.assert_called_once_with(snapshot_data, "/")

    @pytest.mark.asyncio
    async def test_restore_snapshot_not_found(self):
        """Test restoring when no snapshot exists."""
        runtime = _make_mock_runtime()

        with patch(
            "app.services.snapshot_service._find_snapshot",
            new_callable=AsyncMock,
            return_value=None,
        ):
            from app.services.snapshot_service import restore_snapshot

            result = await restore_snapshot(
                runtime=runtime,
                user_id="user1",
                task_id="task1",
                sandbox_type="execution",
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_restore_snapshot_expired(self):
        """Test restoring an expired snapshot."""
        runtime = _make_mock_runtime()

        with patch(
            "app.services.snapshot_service._find_snapshot",
            new_callable=AsyncMock,
            return_value={
                "id": "snap-123",
                "user_id": "user1",
                "task_id": "task1",
                "sandbox_type": "execution",
                "storage_key": "snapshots/user1/task1/snap.tar.gz",
                "paths_included": ["/home/user"],
                "size_bytes": 100,
                "created_at": datetime.now(timezone.utc) - timedelta(hours=48),
                "expires_at": datetime.now(timezone.utc) - timedelta(hours=24),
            },
        ):
            from app.services.snapshot_service import restore_snapshot

            result = await restore_snapshot(
                runtime=runtime,
                user_id="user1",
                task_id="task1",
                sandbox_type="execution",
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_restore_snapshot_data_missing(self):
        """Test restoring when storage data is missing."""
        runtime = _make_mock_runtime()

        with (
            patch(
                "app.services.snapshot_service._find_snapshot",
                new_callable=AsyncMock,
                return_value={
                    "id": "snap-123",
                    "user_id": "user1",
                    "task_id": "task1",
                    "sandbox_type": "execution",
                    "storage_key": "snapshots/user1/task1/snap.tar.gz",
                    "paths_included": ["/home/user"],
                    "size_bytes": 100,
                    "created_at": datetime.now(timezone.utc),
                    "expires_at": datetime.now(timezone.utc) + timedelta(hours=24),
                },
            ),
            patch(
                "app.services.snapshot_service._retrieve_snapshot",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            from app.services.snapshot_service import restore_snapshot

            result = await restore_snapshot(
                runtime=runtime,
                user_id="user1",
                task_id="task1",
                sandbox_type="execution",
            )

            assert result is False


class TestStorageBackend:
    """Tests for storage backend operations."""

    @pytest.mark.asyncio
    async def test_store_snapshot_local(self):
        """Test storing snapshot to local filesystem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.snapshot_service.settings") as mock_settings:
                mock_settings.storage_backend = "local"
                mock_settings.local_storage_path = tmpdir

                from app.services.snapshot_service import _store_snapshot

                data = b"test snapshot data"
                await _store_snapshot("snapshots/user/task/test.tar.gz", data)

                # Verify file was written
                file_path = os.path.join(tmpdir, "snapshots/user/task/test.tar.gz")
                assert os.path.exists(file_path)
                with open(file_path, "rb") as f:
                    assert f.read() == data

    @pytest.mark.asyncio
    async def test_retrieve_snapshot_local(self):
        """Test retrieving snapshot from local filesystem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write test file
            storage_key = "snapshots/user/task/test.tar.gz"
            file_path = os.path.join(tmpdir, storage_key)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            test_data = b"test snapshot content"
            with open(file_path, "wb") as f:
                f.write(test_data)

            with patch("app.services.snapshot_service.settings") as mock_settings:
                mock_settings.storage_backend = "local"
                mock_settings.local_storage_path = tmpdir

                from app.services.snapshot_service import _retrieve_snapshot

                result = await _retrieve_snapshot(storage_key)
                assert result == test_data

    @pytest.mark.asyncio
    async def test_retrieve_snapshot_local_not_found(self):
        """Test retrieving non-existent snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.snapshot_service.settings") as mock_settings:
                mock_settings.storage_backend = "local"
                mock_settings.local_storage_path = tmpdir

                from app.services.snapshot_service import _retrieve_snapshot

                result = await _retrieve_snapshot("nonexistent/path.tar.gz")
                assert result is None


class TestSandboxSnapshotModel:
    """Tests for the SandboxSnapshot database model."""

    def test_model_to_dict(self):
        from app.db.models import SandboxSnapshot

        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=24)

        snapshot = SandboxSnapshot(
            id="snap-123",
            user_id="user1",
            task_id="task1",
            sandbox_type="execution",
            storage_key="snapshots/user1/task1/snap.tar.gz",
            paths_included=["/home/user", "/tmp/outputs"],
            size_bytes=1024,
            created_at=now,
            expires_at=expires,
        )

        d = snapshot.to_dict()
        assert d["id"] == "snap-123"
        assert d["user_id"] == "user1"
        assert d["task_id"] == "task1"
        assert d["sandbox_type"] == "execution"
        assert d["storage_key"] == "snapshots/user1/task1/snap.tar.gz"
        assert d["paths_included"] == ["/home/user", "/tmp/outputs"]
        assert d["size_bytes"] == 1024
        assert d["created_at"] is not None
        assert d["expires_at"] is not None


class TestExecutionManagerSnapshot:
    """Tests for snapshot integration in ExecutionSandboxManager."""

    def setup_method(self):
        from app.sandbox.execution_sandbox_manager import ExecutionSandboxManager
        ExecutionSandboxManager._instance = None
        ExecutionSandboxManager._lock = None

    @pytest.mark.asyncio
    async def test_save_snapshot_no_session(self):
        """Test save_snapshot returns None when no session exists."""
        from app.sandbox.execution_sandbox_manager import ExecutionSandboxManager

        manager = ExecutionSandboxManager()
        result = await manager.save_snapshot(user_id="user1", task_id="task1")
        assert result is None

    @pytest.mark.asyncio
    async def test_restore_snapshot_no_session(self):
        """Test restore_snapshot returns False when no session exists."""
        from app.sandbox.execution_sandbox_manager import ExecutionSandboxManager

        manager = ExecutionSandboxManager()
        result = await manager.restore_snapshot(user_id="user1", task_id="task1")
        assert result is False


class TestAppManagerSnapshot:
    """Tests for snapshot integration in AppSandboxManager."""

    def setup_method(self):
        from app.sandbox.app_sandbox_manager import AppSandboxManager
        AppSandboxManager._instance = None
        AppSandboxManager._lock = None

    @pytest.mark.asyncio
    async def test_save_snapshot_no_session(self):
        """Test save_snapshot returns None when no session exists."""
        from app.sandbox.app_sandbox_manager import AppSandboxManager

        manager = AppSandboxManager()
        result = await manager.save_snapshot(user_id="user1", task_id="task1")
        assert result is None

    @pytest.mark.asyncio
    async def test_restore_snapshot_no_session(self):
        """Test restore_snapshot returns False when no session exists."""
        from app.sandbox.app_sandbox_manager import AppSandboxManager

        manager = AppSandboxManager()
        result = await manager.restore_snapshot(user_id="user1", task_id="task1")
        assert result is False
