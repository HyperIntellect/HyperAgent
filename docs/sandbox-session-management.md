# Sandbox Session Management

## Overview

HyperAgent implements **session-based sandbox reuse** to optimize sandbox usage and maintain state across multiple tool calls within the same conversation. The system supports two sandbox providers:

- **E2B** — Cloud-based sandboxes (default)
- **BoxLite** — Local Docker-based sandboxes

Both providers share the same session management logic via the `SandboxRuntime` protocol abstraction.

## Provider Selection

The active provider is configured in `backend/app/config.py`:

```python
sandbox_provider: Literal["e2b", "boxlite"] = "e2b"
```

Set via the `SANDBOX_PROVIDER` environment variable. The provider factory in `backend/app/sandbox/provider.py` routes all sandbox creation through the configured provider.

### Provider Comparison

| Aspect | E2B | BoxLite |
|--------|-----|---------|
| **Type** | Cloud-based sandboxes | Local Docker containers |
| **Requirement** | `E2B_API_KEY` | Local Docker daemon |
| **Port Access** | Public URLs (`*.e2b.dev`) | Localhost port mapping (proxied via backend) |
| **Runtime ID** | E2B native sandbox ID | `boxlite-{uuid}` |
| **Streaming** | WebRTC live streaming | Screenshot-based |
| **Startup Time** | Slower (cloud provisioning) | Faster (local) |
| **Cost** | Paid API usage | Local infrastructure only |

### Availability Check

```python
from app.sandbox.provider import is_provider_available

available, error_msg = is_provider_available(sandbox_type="execution")
```

BoxLite requires the package to be installed: `pip install 'hyperagent-api[local-sandbox]'`

## How It Works

### Session Key Format

All sandbox types use a prefixed session key format:

| Sandbox Type | Key Format |
|-------------|------------|
| Unified sandbox | `unified:{user_id}:{task_id}` |
| App sandbox | `app:{user_id}:{task_id}` |
| Desktop sandbox | `desktop:{user_id}:{task_id}` |

If `user_id` or `task_id` is not provided, they default to `"anonymous"` and `"default"` respectively.

**Source:** `backend/app/sandbox/app_sandbox_manager.py`
```python
@staticmethod
def make_session_key(user_id: str | None, task_id: str | None) -> str:
    user = user_id or "anonymous"
    task = task_id or "default"
    return f"app:{user}:{task}"
```

### Task ID Resolution

The backend automatically determines the `task_id` using this priority order:

1. **Explicit `task_id`** from request (if provided)
2. **`conversation_id`** from request (fallback)
3. **Generated UUID** (if neither provided)

**Source:** `backend/app/api/query.py`
```python
chat_task_id = request.task_id or request.conversation_id or str(uuid.uuid4())
```

### Session Lifecycle

1. **First request in conversation:**
   - New sandbox created (E2B cloud instance or BoxLite Docker container)
   - Session stored with key: `app:{user_id}:{conversation_id}`
   - Session timeout starts
   - Background cleanup task started (if not already running)

2. **Subsequent requests in same conversation:**
   - Same `conversation_id` -> same session key
   - Existing sandbox retrieved and reused
   - Health check performed (with skip optimization)
   - `last_accessed` timestamp refreshed

3. **Session expiration:**
   - Background cleanup runs every 60 seconds
   - Expired sessions are cleaned up
   - Sandbox closed and removed from memory

## Implementation Details

### Sandbox Managers

**Unified Sandbox Manager** (`backend/app/sandbox/unified_sandbox_manager.py`)
- Provides a single shared `SandboxRuntime` for both code execution and app development within one agent run
- Session key: `unified:{user_id}:{task_id}`
- Session timeout: 30 minutes (default)
- `get_or_create_runtime(user_id, task_id)` — shared runtime for both code and app
- `get_code_executor(user_id, task_id)` — wraps shared runtime as `BaseCodeExecutor`
- `get_app_session(user_id, task_id, template)` — wraps shared runtime as `AppSandboxSession`
- Desktop sandboxes remain separate (different VM image requirement)
- Health check: `echo 'health_check'` with 5s timeout, skipped if last check < 30s ago
- Background cleanup loop: every 60 seconds, removes expired sessions
- `cleanup_sandboxes_for_task()` prioritizes unified sessions first

**App Sandbox Manager** (`backend/app/sandbox/app_sandbox_manager.py`)
- Manages app development sandboxes (can also delegate to unified manager)
- Session timeout: 30 minutes (default)
- Health check: `echo 'health_check'` command with 5-second timeout
- Health check skip: If last successful check was < 30 seconds ago, skip
- Automatic port allocation for BoxLite apps

**Execution Sandbox Manager** (`backend/app/sandbox/execution_sandbox_manager.py`)
- Manages code execution sandboxes (Python, JS, Bash)
- Session timeout: 10 minutes (E2B, configurable via `e2b_session_timeout_minutes`)
- Supports `get_or_create_sandbox_with_runtime()` for sharing a runtime from unified manager

**Desktop Sandbox Manager** (`backend/app/sandbox/desktop_sandbox_manager.py`)
- Manages browser automation sandboxes
- Session timeout: 15 minutes (E2B, configurable via `e2b_desktop_session_timeout_minutes`)
- Stream management for live browser viewing

### BoxLite Components

| Component | Class | File |
|-----------|-------|------|
| Runtime Adapter | `BoxLiteRuntime` | `backend/app/sandbox/boxlite/runtime.py` |
| Code Executor | `BoxLiteCodeExecutor` | `backend/app/sandbox/boxlite/code_executor.py` |
| Desktop Executor | `BoxLiteDesktopExecutor` | `backend/app/sandbox/boxlite/desktop_executor.py` |
| Provider Factory | `create_app_runtime()` | `backend/app/sandbox/provider.py` |

### BoxLite Images

| Sandbox Type | Default Image | Use Case |
|-------------|---------------|----------|
| Code Execution | `python:3.12-slim` | Python code execution |
| App Development | `node:20-slim` | Web app scaffolding (React, Vue, Express) |
| Desktop/Browser | `boxlite/desktop:latest` | Browser automation |

### Port Forwarding (BoxLite)

BoxLite uses automatic host port allocation for app sandboxes:

```python
# Ports start from boxlite_app_host_port_start (default: 10000)
host_port = settings.boxlite_app_host_port_start + self._next_host_port_offset
self._next_host_port_offset += 1
```

For BoxLite, preview URLs are proxied through the backend so iframes don't need direct Docker port access:

```python
if settings.sandbox_provider == "boxlite":
    preview_url = f"/api/v1/sandbox/app/{session.sandbox_id}/"
else:
    preview_url = raw_url  # E2B public URL
```

### Health Checks

Health checks use a monotonic clock optimization to avoid excessive checks:

```python
async def _is_sandbox_healthy(self, session: AppSandboxSession) -> bool:
    # Skip if last check was < 30 seconds ago
    now = time.monotonic()
    if session.last_health_check > 0 and (now - session.last_health_check) < 30:
        return True

    result = await session.sandbox.run_command("echo 'health_check'", timeout=5)
    healthy = result.exit_code == 0 and "health_check" in result.stdout
    if healthy:
        session.last_health_check = now
    return healthy
```

## Persistent Sandbox Snapshots

**File:** `backend/app/services/snapshot_service.py`

Sandbox state (installed packages, generated files) is preserved across SSE disconnects and timeouts via workspace snapshots.

### How It Works

1. **On disconnect/cleanup**: Execution and app managers call `save_snapshot()` to tar key directories and upload to storage
2. **On reconnect**: `restore_snapshot()` downloads the latest snapshot and extracts it into the new sandbox runtime
3. **Storage**: R2 (production) or local filesystem (development), selected via `STORAGE_BACKEND` env var
4. **Database**: `SandboxSnapshot` model tracks metadata (user_id, task_id, sandbox_type, storage_key, size, expiry)

### Snapshot Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `SNAPSHOT_MAX_SIZE_BYTES` | 100MB | Maximum snapshot archive size |
| `SNAPSHOT_RETENTION_HOURS` | 24 | How long snapshots are kept |
| `SNAPSHOT_DEFAULT_PATHS_EXECUTION` | `/home/user`, `/tmp/outputs` | Paths snapshotted for execution sandboxes |
| `SNAPSHOT_DEFAULT_PATHS_APP` | `/home/user/app` | Paths snapshotted for app sandboxes |

### Snapshot Lifecycle

```
Save: runtime → tar paths → check size → upload to storage → record in DB
Restore: find latest snapshot → check expiry → download from storage → extract into runtime
```

## Cross-Sandbox Handoff Artifacts

**File:** `backend/app/sandbox/artifact_transfer.py`

When agents hand off work (Task → Research), files from the source sandbox can be transferred to the target sandbox.

- `collect_artifacts(runtime, patterns, max_files=10, max_size_mb=50)` — find and upload matching files
- `restore_artifacts(runtime, artifacts)` — download files into target sandbox
- `cleanup_artifacts(artifacts)` — remove transferred files from storage after completion
- `format_artifact_summary(artifacts)` — human-readable summary appended to handoff context
- Default patterns: `*.py`, `*.csv`, `*.json`, `*.txt`, `*.md`, `*.html`, `*.js`, `*.ts`
- `HandoffInfo` includes optional `handoff_artifacts` field
- Supervisor calls `_restore_handoff_artifacts()` during handoff flow

## Configuration

### E2B Settings

Environment variables in `backend/app/config.py`:

```python
# Execution Sandbox
e2b_api_key: str = ""
e2b_template_id: str = ""  # Optional custom template
e2b_code_timeout: int = 300  # 5 minutes
e2b_session_timeout_minutes: int = 10

# Desktop Sandbox
e2b_desktop_timeout: int = 900  # 15 minutes
e2b_desktop_session_timeout_minutes: int = 15
e2b_desktop_stream_ready_wait_ms: int = 3000
```

### BoxLite Settings

```python
# Provider
sandbox_provider: str = "e2b"  # Set to "boxlite" for local sandboxes

# Images
boxlite_code_image: str = "python:3.12-slim"
boxlite_desktop_image: str = "boxlite/desktop:latest"
boxlite_app_image: str = "node:20-slim"

# Timeouts
boxlite_code_timeout: int = 300   # 5 minutes
boxlite_desktop_timeout: int = 900  # 15 minutes

# Resources
boxlite_cpus: int = 2
boxlite_memory_mib: int = 1024
boxlite_disk_size_gb: int = 4
boxlite_working_dir: str = "/home/user"
boxlite_auto_remove: bool = True

# Port Allocation
boxlite_app_host_port_start: int = 10000
```

### App Session Settings (shared)

```python
# Hardcoded defaults in app_sandbox_manager.py
SESSION_TIMEOUT = 30 minutes
HEALTH_CHECK_SKIP_SECONDS = 30
CLEANUP_INTERVAL_SECONDS = 60
```

## Metrics

Get sandbox metrics via:
```python
from app.sandbox import get_sandbox_metrics

metrics = get_sandbox_metrics()
# Returns:
{
    "unified": {
        "active_sessions": int,
        "total_created": int,
        "total_cleaned": int,
        "total_reused": int,
        "health_check_failures": int,
    },
    "execution": { ... },
    "desktop": { ... },
    "app": { ... },
    "totals": { ... }
}
```

## Recent Fix (2026-01-24)

### Problem

Local conversations (created when user is not authenticated or backend is unavailable) were NOT reusing sandboxes.

**Root cause:** Frontend was filtering out local conversation IDs before sending to backend:
```typescript
// OLD - Incorrect behavior
conversation_id: conversationId?.startsWith("local-") ? null : conversationId
```

This caused:
- `conversation_id` sent as `null` for local conversations
- Backend generated new UUID for each request
- **New sandbox created every time!**

### Solution

**Always send `conversation_id`**, regardless of whether it's local or remote:

```typescript
// NEW - Correct behavior
conversation_id: conversationId
```

**Files changed:**
- `web/components/query/chat-interface.tsx` (2 locations)
  - Line ~693: `handleChat` function
  - Line ~1077: `handleAgentTask` function

## Benefits

1. **Cost Optimization**
   - Reduced sandbox creation overhead
   - Lower API costs (E2B) or container overhead (BoxLite)
   - Fewer cold starts

2. **Performance**
   - Instant sandbox availability for subsequent requests
   - No setup delay for repeated tool calls
   - Persistent file system within session
   - Health check skip optimization avoids redundant checks

3. **User Experience**
   - Files persist across code execution calls
   - Faster response times
   - Consistent environment state

4. **Flexibility**
   - Switch between cloud (E2B) and local (BoxLite) with a single config change
   - Same session management logic for both providers
   - BoxLite enables offline development without API keys

## Troubleshooting

### Sandbox not reusing

**Check 1:** Verify `conversation_id` is being sent
```bash
# In browser dev tools network tab, check request payload
{
  "conversation_id": "abc-123-def",  # Should NOT be null
  "message": "...",
  ...
}
```

**Check 2:** Verify session key format
```python
# In backend logs, look for:
[AppSandboxManager] Session key: app:user123:conv-abc-123
```

**Check 3:** Check session timeout
```python
# App sessions expire after 30 minutes of inactivity
# E2B execution sessions: 10 minutes
# E2B desktop sessions: 15 minutes
```

### Health check failures

If health checks are failing frequently, check:
1. Network connectivity to E2B service (if using E2B)
2. Docker daemon is running (if using BoxLite)
3. Sandbox resource limits
4. API key validity (E2B)

### BoxLite-specific issues

**Docker not running:**
```bash
docker info  # Verify Docker daemon is accessible
```

**Port conflicts:**
```bash
# Check if BoxLite port range is available
lsof -i :10000-10100
```

**Container cleanup:**
```bash
# If auto_remove is disabled, manually clean up
docker ps -a --filter "name=boxlite-" | grep Exited
```

## Future Enhancements

Potential improvements to consider:

1. **Smart session timeout**
   - Extend timeout based on usage patterns
   - Proactive cleanup of idle sessions

2. **Sandbox pooling**
   - Pre-warm sandbox pool for faster first request
   - Reduced cold start latency (especially for E2B)

3. **Hybrid provider support**
   - Use BoxLite for development, E2B for production
   - Automatic fallback between providers

Previously planned items that are now implemented:
- **Unified sandbox per task** — `UnifiedSandboxManager` shares a single runtime across code execution and app development
- **Persistent snapshots** — Workspace state saved to storage and restored on reconnect
- **Cross-sandbox handoff** — Files transferred between sandboxes during agent handoffs
