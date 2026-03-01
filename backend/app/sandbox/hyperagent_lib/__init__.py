"""HyperAgent helper library for CodeAct sandbox execution.

This lightweight module is installed inside the sandbox and provides
convenient helper functions that wrap sandbox runtime operations. The
functions communicate with the host agent via a simple JSON-over-stdin/stdout
protocol or by delegating to shell commands.

Usage inside sandbox scripts:
    import hyperagent
    results = hyperagent.web_search("python async best practices")
    content = hyperagent.read_file("/home/user/data.csv")
    hyperagent.write_file("/home/user/output.txt", "processed data")
    output = hyperagent.run_command("ls -la /home/user")
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return results.

    This is a lightweight wrapper that writes a search request file
    and returns cached results if available. In CodeAct mode, web search
    is primarily done before the script runs via the agent's own tools.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of result dicts with keys: title, url, snippet
    """
    # Check for pre-cached search results (agent deposits these before execution)
    cache_dir = Path("/tmp/hyperagent/search_cache")
    if cache_dir.exists():
        for cache_file in sorted(cache_dir.glob("*.json")):
            try:
                data = json.loads(cache_file.read_text())
                if data.get("query", "").lower() == query.lower():
                    return data.get("results", [])[:max_results]
            except (json.JSONDecodeError, OSError):
                continue

    # Write request for the agent to pick up on next iteration
    request_dir = Path("/tmp/hyperagent/requests")
    request_dir.mkdir(parents=True, exist_ok=True)
    request = {"type": "web_search", "query": query, "max_results": max_results}
    request_file = request_dir / f"search_{hash(query) & 0xFFFFFFFF:08x}.json"
    request_file.write_text(json.dumps(request))

    return [{"title": "Search pending", "url": "", "snippet": f"Search for '{query}' has been queued."}]


def read_file(path: str) -> str:
    """Read a file from the sandbox filesystem.

    Args:
        path: Absolute or relative file path

    Returns:
        File content as string

    Raises:
        FileNotFoundError: If file does not exist
        UnicodeDecodeError: If file is binary
    """
    return Path(path).read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    """Write content to a file in the sandbox.

    Creates parent directories if they don't exist.

    Args:
        path: Absolute or relative file path
        content: Content to write

    Returns:
        Confirmation message with path and byte count
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"Written {len(content)} bytes to {path}"


def run_command(
    cmd: str,
    timeout: int = 60,
    cwd: str | None = None,
) -> dict:
    """Run a shell command in the sandbox.

    Args:
        cmd: Shell command to execute
        timeout: Timeout in seconds
        cwd: Working directory (defaults to current)

    Returns:
        Dict with keys: exit_code, stdout, stderr
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
        }
    except Exception as e:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
        }


def browse(url: str) -> str:
    """Fetch a URL and return its text content.

    Uses curl for lightweight HTTP fetching within the sandbox.
    For full browser automation, use the agent's browser tools instead.

    Args:
        url: URL to fetch

    Returns:
        Response body as text
    """
    result = run_command(
        f"curl -sL -m 30 --max-filesize 5242880 '{url}'",
        timeout=35,
    )
    if result["exit_code"] != 0:
        return f"Error fetching {url}: {result['stderr']}"
    return result["stdout"]


def list_files(directory: str = ".") -> list[str]:
    """List files in a directory.

    Args:
        directory: Directory path to list

    Returns:
        List of file/directory names
    """
    try:
        return sorted(os.listdir(directory))
    except OSError as e:
        return [f"Error: {e}"]
