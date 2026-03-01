"""Browser automation tools using E2B Desktop Sandbox.

Provides browser automation capabilities in a secure, isolated environment.
Supports navigation, screenshots, and computer-use style interactions.
"""

import asyncio
import json
from typing import Literal
from urllib.parse import urlparse

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.guardrails.scanners.tool_scanner import tool_scanner
from app.sandbox import is_desktop_sandbox_available
from app.sandbox.base_desktop_executor import get_screenshot_as_base64

logger = get_logger(__name__)

# Constants for timing (in milliseconds)
WAIT_AFTER_FOCUS_ADDRESS_BAR = 800
WAIT_AFTER_SELECT_ALL = 300
WAIT_AFTER_PASTE_URL = 800  # Longer wait to ensure URL is fully pasted
WAIT_AFTER_CLICK = 500
WAIT_AFTER_SCROLL = 500

# URL truncation length for logging
URL_LOG_TRUNCATE_LENGTH = 100

# Maximum content length to return (in characters)
# This prevents token limit errors when pages have very long content
# ~4 chars per token, so 50K chars ≈ 12.5K tokens
MAX_CONTENT_LENGTH = 50000


def _validate_url(url: str) -> tuple[bool, str | None]:
    """Validate URL format.

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            return False, "URL must include a scheme (e.g., https://)"
        if parsed.scheme not in ("http", "https"):
            return False, f"Unsupported URL scheme: {parsed.scheme}. Use http or https."
        if not parsed.netloc:
            return False, "URL must include a domain (e.g., example.com)"
        return True, None
    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"


async def _get_browser_session(user_id: str | None, task_id: str | None):
    """Get an existing browser session.

    Returns:
        Tuple of (manager, session) or (None, None) if unavailable
    """
    from app.sandbox import get_desktop_sandbox_manager

    manager = get_desktop_sandbox_manager()
    session = await manager.get_session(user_id=user_id, task_id=task_id)
    return manager, session


def _error_response(**kwargs) -> str:
    """Create a JSON error response."""
    return json.dumps({"success": False, **kwargs})


def _success_response(**kwargs) -> str:
    """Create a JSON success response."""
    return json.dumps({"success": True, **kwargs})


class BrowserNavigateInput(BaseModel):
    """Input schema for browser navigate tool."""

    url: str = Field(
        description="The URL to navigate to (e.g., 'https://example.com')"
    )
    extract_content: bool = Field(
        default=True,
        description="Whether to extract text content from the page (recommended for research)",
    )
    take_screenshot: bool = Field(
        default=False,
        description="Whether to capture a screenshot after navigation",
    )
    wait_ms: int = Field(
        default=5000,
        ge=1000,
        le=30000,
        description="Time to wait after navigation in milliseconds (1000-30000)",
    )
    user_id: str | None = Field(
        default=None,
        description="User ID for session management (injected by system)",
    )
    task_id: str | None = Field(
        default=None,
        description="Task ID for session management (injected by system)",
    )


@tool(args_schema=BrowserNavigateInput)
async def browser_navigate(
    url: str,
    extract_content: bool = True,
    take_screenshot: bool = False,
    wait_ms: int = 5000,
    user_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Navigate to a URL in an isolated browser and extract page content.

    This tool runs a real browser in a secure E2B Desktop sandbox, providing:
    - Full browser functionality with JavaScript support
    - Text content extraction from web pages
    - Optional visual screenshots
    - Isolated environment for safe browsing

    Use this when you need to:
    - Visit websites and extract their text content for analysis
    - Access pages that require JavaScript rendering
    - Capture screenshots of web pages (optional)
    - Browse in a secure, isolated environment

    The sandbox is reused across multiple calls within the same task,
    so subsequent navigations are faster.

    Args:
        url: The URL to navigate to
        extract_content: Whether to extract text content from the page (default: True)
        take_screenshot: Whether to capture a screenshot after navigation (default: False)
        wait_ms: Time to wait after navigation for page to load
        user_id: User ID for session management (injected by system)
        task_id: Task ID for session management (injected by system)

    Returns:
        JSON string with navigation results, extracted content, and optional screenshot
    """
    # Validate URL format (basic validation)
    is_valid, error_msg = _validate_url(url)
    if not is_valid:
        logger.warning("browser_navigate_invalid_url", url=url[:URL_LOG_TRUNCATE_LENGTH], error=error_msg)
        return _error_response(error=error_msg, url=url)

    # Apply guardrail scanner for security checks (blocks internal IPs, dangerous schemes, etc.)
    scan_result = await tool_scanner.scan("browser_navigate", {"url": url})
    if not scan_result.passed:
        logger.warning(
            "browser_navigate_blocked_by_guardrail",
            url=url[:URL_LOG_TRUNCATE_LENGTH],
            reason=scan_result.reason,
            violations=[v.value for v in scan_result.violations],
        )
        return _error_response(
            error=f"URL blocked by security policy: {scan_result.reason}",
            url=url,
            blocked_by="guardrail",
        )

    if not is_desktop_sandbox_available():
        logger.error("browser_unavailable")
        return _error_response(
            error="Desktop sandbox not available. Check SANDBOX_PROVIDER configuration.",
            url=url,
        )

    logger.info(
        "browser_navigate_invoked",
        url=url[:URL_LOG_TRUNCATE_LENGTH],
        extract_content=extract_content,
        take_screenshot=take_screenshot,
        wait_ms=wait_ms,
    )

    try:
        from app.sandbox import get_desktop_sandbox_manager

        manager = get_desktop_sandbox_manager()
        session_key = manager.make_session_key(user_id, task_id)

        # Get existing session - supervisor should have already created it
        session = await manager.get_session(user_id=user_id, task_id=task_id)
        newly_created = False

        if session:
            logger.info(
                "browser_navigate_using_existing_session",
                session_key=session_key,
                sandbox_id=session.sandbox_id,
                browser_launched=session.browser_launched,
                stream_ready=session.is_stream_ready,
            )
        else:
            # Fallback: create session if not exists (shouldn't happen normally)
            logger.warning(
                "browser_navigate_creating_new_session",
                session_key=session_key,
                user_id=user_id,
                task_id=task_id,
            )
            session = await manager.get_or_create_sandbox(
                user_id=user_id,
                task_id=task_id,
                launch_browser=True,
            )
            newly_created = True

        executor = session.executor

        # Launch browser if needed
        if not session.browser_launched:
            logger.info("browser_navigate_launching_browser", sandbox_id=session.sandbox_id)
            await executor.launch_browser()

        # Ensure stream is ready before any visible actions
        stream_url, auth_key = await manager.ensure_stream_ready(session)

        logger.info(
            "browser_navigate_stream_ready",
            sandbox_id=session.sandbox_id,
            stream_url=stream_url[:URL_LOG_TRUNCATE_LENGTH] if stream_url else None,
            newly_created=newly_created,
        )

        # Navigate to URL using keyboard actions
        # Step 1: Focus address bar with Ctrl+L (E2B uses Linux)
        logger.info("browser_action_step1_focus_address_bar", sandbox_id=session.sandbox_id)
        await executor.press_key(["ctrl", "l"])
        await executor.wait(WAIT_AFTER_FOCUS_ADDRESS_BAR)

        # Step 2: Select all existing text with Ctrl+A
        logger.info("browser_action_step2_select_all", sandbox_id=session.sandbox_id)
        await executor.press_key(["ctrl", "a"])
        await executor.wait(WAIT_AFTER_SELECT_ALL)

        # Step 3: Type the URL (uses clipboard for non-ASCII, direct typing for ASCII)
        logger.info("browser_action_step3_type_url", sandbox_id=session.sandbox_id, url=url[:URL_LOG_TRUNCATE_LENGTH])
        await executor.type_text_via_clipboard(url)
        await executor.wait(WAIT_AFTER_PASTE_URL)

        # Step 4: Press Enter to navigate
        logger.info("browser_action_step4_press_enter", sandbox_id=session.sandbox_id)
        await executor.press_key("enter")

        # Step 5: Wait for page to load
        logger.info("browser_action_step5_waiting_for_page", sandbox_id=session.sandbox_id, wait_ms=wait_ms)
        await executor.wait(wait_ms)

        logger.info("browser_action_navigation_complete", sandbox_id=session.sandbox_id, url=url[:URL_LOG_TRUNCATE_LENGTH])

        result = {
            "success": True,
            "url": url,
            "sandbox_id": session.sandbox_id,
            "stream_url": stream_url,
            "stream_auth_key": auth_key,
            "stream_ready": session.is_stream_ready,
        }

        # Extract page content if requested
        # Note: Uses curl+html2text for extraction, separate from browser rendering
        if extract_content:
            try:
                page_content = await executor.extract_page_content(url)
                original_length = len(page_content)

                # Truncate content to prevent token limit errors
                if original_length > MAX_CONTENT_LENGTH:
                    page_content = page_content[:MAX_CONTENT_LENGTH] + f"\n\n... [Content truncated. Showing {MAX_CONTENT_LENGTH:,} of {original_length:,} characters]"
                    result["content_truncated"] = True

                result["content"] = page_content
                result["content_length"] = original_length

                logger.info(
                    "browser_content_extracted",
                    url=url[:URL_LOG_TRUNCATE_LENGTH],
                    content_length=original_length,
                    truncated=original_length > MAX_CONTENT_LENGTH,
                )
            except Exception as e:
                logger.warning(
                    "browser_content_extraction_failed",
                    url=url[:URL_LOG_TRUNCATE_LENGTH],
                    error=str(e),
                )
                result["content"] = f"Failed to extract content: {str(e)}"
                result["content_error"] = str(e)

        # Capture screenshot if requested
        if take_screenshot:
            screenshot_bytes = await executor.screenshot()
            result["screenshot"] = get_screenshot_as_base64(screenshot_bytes)

        logger.info(
            "browser_navigate_completed",
            url=url[:URL_LOG_TRUNCATE_LENGTH],
            sandbox_id=session.sandbox_id,
            has_content=extract_content,
            has_screenshot=take_screenshot,
        )

        return json.dumps(result)

    except Exception as e:
        logger.error(
            "browser_navigate_failed",
            url=url[:URL_LOG_TRUNCATE_LENGTH],
            error=str(e),
        )
        return _error_response(error=str(e), url=url)


class BrowserScreenshotInput(BaseModel):
    """Input schema for browser screenshot tool."""

    user_id: str | None = Field(
        default=None,
        description="User ID for session management (injected by system)",
    )
    task_id: str | None = Field(
        default=None,
        description="Task ID for session management (injected by system)",
    )


@tool(args_schema=BrowserScreenshotInput)
async def browser_screenshot(
    user_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Capture a screenshot of the current browser state.

    Takes a screenshot of the entire sandbox, including the browser
    and any visible content. Use this to see what's currently displayed
    or to verify the results of previous actions.

    The sandbox must have been created by a previous browser_navigate
    call within the same task.

    Args:
        user_id: User ID for session management (injected by system)
        task_id: Task ID for session management (injected by system)

    Returns:
        JSON string with screenshot data
    """
    if not is_desktop_sandbox_available():
        return _error_response(error="Desktop sandbox not available. Check SANDBOX_PROVIDER configuration.")

    logger.info("browser_screenshot_invoked")

    try:
        manager, session = await _get_browser_session(user_id, task_id)

        if not session:
            return _error_response(error="No active browser sandbox session. Use browser_navigate first.")

        screenshot_bytes = await session.executor.screenshot()

        logger.info("browser_screenshot_completed", sandbox_id=session.sandbox_id)

        return _success_response(
            screenshot=get_screenshot_as_base64(screenshot_bytes),
            sandbox_id=session.sandbox_id,
        )

    except Exception as e:
        logger.error("browser_screenshot_failed", error=str(e))
        return _error_response(error=str(e))


class BrowserClickInput(BaseModel):
    """Input schema for browser click tool."""

    x: int = Field(description="X coordinate to click (relative to screen)")
    y: int = Field(description="Y coordinate to click (relative to screen)")
    button: Literal["left", "right", "middle"] = Field(
        default="left",
        description="Mouse button to click",
    )
    user_id: str | None = Field(
        default=None,
        description="User ID for session management (injected by system)",
    )
    task_id: str | None = Field(
        default=None,
        description="Task ID for session management (injected by system)",
    )


@tool(args_schema=BrowserClickInput)
async def browser_click(
    x: int,
    y: int,
    button: Literal["left", "right", "middle"] = "left",
    user_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Click at a specific position in the browser.

    Performs a mouse click at the specified coordinates. Use this for
    interacting with buttons, links, form fields, or any clickable elements.

    Coordinates are relative to the sandbox screen.

    Args:
        x: X coordinate to click
        y: Y coordinate to click
        button: Mouse button to use (left, right, middle)
        user_id: User ID for session management (injected by system)
        task_id: Task ID for session management (injected by system)

    Returns:
        JSON string with click result
    """
    if not is_desktop_sandbox_available():
        return _error_response(error="Desktop sandbox not available. Check SANDBOX_PROVIDER configuration.")

    logger.info("browser_click_invoked", x=x, y=y, button=button)

    try:
        manager, session = await _get_browser_session(user_id, task_id)

        if not session:
            return _error_response(error="No active browser sandbox session. Use browser_navigate first.")

        # Ensure stream is ready before visible action
        await manager.ensure_stream_ready(session)

        await session.executor.click(x, y, button)
        await session.executor.wait(WAIT_AFTER_CLICK)

        logger.info("browser_click_completed", x=x, y=y, button=button)

        return _success_response(x=x, y=y, button=button, sandbox_id=session.sandbox_id)

    except Exception as e:
        logger.error("browser_click_failed", error=str(e))
        return _error_response(error=str(e))


class BrowserTypeInput(BaseModel):
    """Input schema for browser type tool."""

    text: str = Field(description="Text to type into the focused element")
    user_id: str | None = Field(
        default=None,
        description="User ID for session management (injected by system)",
    )
    task_id: str | None = Field(
        default=None,
        description="Task ID for session management (injected by system)",
    )


@tool(args_schema=BrowserTypeInput)
async def browser_type(
    text: str,
    user_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Type text in the browser.

    Types the specified text using the keyboard. The text will be typed
    wherever the cursor/focus currently is. Use browser_click first to
    focus on input fields before typing.

    Uses clipboard paste internally to handle non-ASCII characters correctly.

    Args:
        text: Text to type
        user_id: User ID for session management (injected by system)
        task_id: Task ID for session management (injected by system)

    Returns:
        JSON string with typing result
    """
    if not is_desktop_sandbox_available():
        return _error_response(error="Desktop sandbox not available. Check SANDBOX_PROVIDER configuration.")

    logger.info("browser_type_invoked", text_length=len(text))

    try:
        manager, session = await _get_browser_session(user_id, task_id)

        if not session:
            return _error_response(error="No active browser sandbox session. Use browser_navigate first.")

        # Ensure stream is ready before visible action
        await manager.ensure_stream_ready(session)

        # Use clipboard paste to handle non-ASCII characters correctly
        await session.executor.type_text_via_clipboard(text)

        logger.info("browser_type_completed", text_length=len(text))

        return _success_response(text_length=len(text), sandbox_id=session.sandbox_id)

    except Exception as e:
        logger.error("browser_type_failed", error=str(e))
        return _error_response(error=str(e))


class BrowserPressKeyInput(BaseModel):
    """Input schema for browser press key tool."""

    key: str = Field(
        description="Key to press (e.g., 'enter', 'tab', 'escape', 'space', 'backspace') "
        "or combinations with '+' separator (e.g., 'ctrl+a', 'ctrl+shift+t')"
    )
    user_id: str | None = Field(
        default=None,
        description="User ID for session management (injected by system)",
    )
    task_id: str | None = Field(
        default=None,
        description="Task ID for session management (injected by system)",
    )


@tool(args_schema=BrowserPressKeyInput)
async def browser_press_key(
    key: str,
    user_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Press a key or key combination in the browser.

    Presses a single key or key combination. Useful for:
    - Submitting forms (enter)
    - Navigation (tab, arrow keys)
    - Keyboard shortcuts (ctrl+a, ctrl+c, ctrl+v)
    - Closing dialogs (escape)

    Use lowercase key names (e.g., 'enter', 'tab', 'escape', 'space', 'backspace').
    For combinations, use '+' separator: 'ctrl+a', 'ctrl+shift+t'

    Args:
        key: Key or key combination to press
        user_id: User ID for session management (injected by system)
        task_id: Task ID for session management (injected by system)

    Returns:
        JSON string with key press result
    """
    if not is_desktop_sandbox_available():
        return _error_response(error="Desktop sandbox not available. Check SANDBOX_PROVIDER configuration.")

    logger.info("browser_press_key_invoked", key=key)

    try:
        manager, session = await _get_browser_session(user_id, task_id)

        if not session:
            return _error_response(error="No active browser sandbox session. Use browser_navigate first.")

        # Ensure stream is ready before visible action
        await manager.ensure_stream_ready(session)

        # Parse key combinations (supports both '+' and ' ' separators)
        if "+" in key:
            keys = [k.strip() for k in key.split("+")]
            await session.executor.press_key(keys)
        elif " " in key:
            keys = [k.strip() for k in key.split()]
            await session.executor.press_key(keys)
        else:
            await session.executor.press_key(key)

        logger.info("browser_press_key_completed", key=key)

        return _success_response(key=key, sandbox_id=session.sandbox_id)

    except Exception as e:
        logger.error("browser_press_key_failed", error=str(e))
        return _error_response(error=str(e))


class BrowserScrollInput(BaseModel):
    """Input schema for browser scroll tool."""

    direction: Literal["up", "down"] = Field(
        default="down",
        description="Scroll direction (up or down)",
    )
    amount: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of scroll steps (1-20, each step is roughly one mouse wheel tick)",
    )
    user_id: str | None = Field(
        default=None,
        description="User ID for session management (injected by system)",
    )
    task_id: str | None = Field(
        default=None,
        description="Task ID for session management (injected by system)",
    )


@tool(args_schema=BrowserScrollInput)
async def browser_scroll(
    direction: Literal["up", "down"] = "down",
    amount: int = 3,
    user_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Scroll the page in the browser.

    Scrolls the page up or down by the specified amount.
    Useful for viewing content below the fold or navigating long pages.

    Args:
        direction: Scroll direction (up or down)
        amount: Number of scroll steps (1-20)
        user_id: User ID for session management (injected by system)
        task_id: Task ID for session management (injected by system)

    Returns:
        JSON string with scroll result
    """
    if not is_desktop_sandbox_available():
        return _error_response(error="Desktop sandbox not available. Check SANDBOX_PROVIDER configuration.")

    logger.info("browser_scroll_invoked", direction=direction, amount=amount)

    try:
        manager, session = await _get_browser_session(user_id, task_id)

        if not session:
            return _error_response(error="No active browser sandbox session. Use browser_navigate first.")

        # Ensure stream is ready before visible action
        await manager.ensure_stream_ready(session)

        await session.executor.scroll(direction, amount)
        await session.executor.wait(WAIT_AFTER_SCROLL)

        logger.info("browser_scroll_completed", direction=direction, amount=amount)

        return _success_response(direction=direction, amount=amount, sandbox_id=session.sandbox_id)

    except Exception as e:
        logger.error("browser_scroll_failed", error=str(e))
        return _error_response(error=str(e))


class BrowserGetStreamInput(BaseModel):
    """Input schema for browser get stream URL tool."""

    user_id: str | None = Field(
        default=None,
        description="User ID for session management (injected by system)",
    )
    task_id: str | None = Field(
        default=None,
        description="Task ID for session management (injected by system)",
    )


@tool(args_schema=BrowserGetStreamInput)
async def browser_get_stream_url(
    user_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Get a live stream URL of the browser.

    Returns a URL that can be used to view the browser in real-time.
    Useful for monitoring browser activity or debugging.

    The stream URL requires authentication by default.

    Args:
        user_id: User ID for session management (injected by system)
        task_id: Task ID for session management (injected by system)

    Returns:
        JSON string with stream URL and auth key
    """
    if not is_desktop_sandbox_available():
        return _error_response(error="Desktop sandbox not available. Check SANDBOX_PROVIDER configuration.")

    logger.info("browser_get_stream_url_invoked")

    try:
        _, session = await _get_browser_session(user_id, task_id)

        if not session:
            return _error_response(error="No active browser sandbox session. Use browser_navigate first.")

        stream_url, auth_key = await session.executor.get_stream_url(require_auth=True)

        logger.info("browser_get_stream_url_completed")

        return _success_response(
            stream_url=stream_url,
            auth_key=auth_key,
            sandbox_id=session.sandbox_id,
        )

    except Exception as e:
        logger.error("browser_get_stream_url_failed", error=str(e))
        return _error_response(error=str(e))


# JavaScript execution helper script template.
# Uses CDP (Chrome DevTools Protocol) to evaluate JavaScript in the browser.
# The script discovers the browser's CDP endpoint, connects, and evaluates the expression.
_CDP_EVAL_SCRIPT = r"""
import json, sys, urllib.request

# Discover browser CDP endpoint
try:
    with urllib.request.urlopen("http://localhost:9222/json/list", timeout=5) as r:
        targets = json.loads(r.read())
except Exception:
    targets = []

ws_url = None
for t in targets:
    if t.get("type") == "page":
        ws_url = t.get("webSocketDebuggerUrl")
        break

if not ws_url:
    print(json.dumps({"error": "No browser page found via CDP. Is the browser running with --remote-debugging-port=9222?"}))
    sys.exit(0)

# Use websocket via subprocess since ws library may not be installed
import socket, struct, hashlib, base64, os

# Parse ws URL
host_port = ws_url.replace("ws://", "").split("/")[0]
path = "/" + "/".join(ws_url.replace("ws://", "").split("/")[1:])
host, port = host_port.split(":")
port = int(port)

# WebSocket handshake
key = base64.b64encode(os.urandom(16)).decode()
sock = socket.create_connection((host, port), timeout=10)
req = (
    f"GET {path} HTTP/1.1\r\n"
    f"Host: {host_port}\r\n"
    f"Upgrade: websocket\r\n"
    f"Connection: Upgrade\r\n"
    f"Sec-WebSocket-Key: {key}\r\n"
    f"Sec-WebSocket-Version: 13\r\n\r\n"
)
sock.sendall(req.encode())
resp = b""
while b"\r\n\r\n" not in resp:
    resp += sock.recv(4096)

def ws_send(sock, data):
    payload = data.encode()
    frame = bytearray()
    frame.append(0x81)
    mask_key = os.urandom(4)
    length = len(payload)
    if length < 126:
        frame.append(0x80 | length)
    elif length < 65536:
        frame.append(0x80 | 126)
        frame.extend(struct.pack(">H", length))
    else:
        frame.append(0x80 | 127)
        frame.extend(struct.pack(">Q", length))
    frame.extend(mask_key)
    masked = bytearray(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    frame.extend(masked)
    sock.sendall(frame)

def ws_recv(sock):
    header = sock.recv(2)
    if len(header) < 2:
        return ""
    length = header[1] & 0x7F
    if length == 126:
        length = struct.unpack(">H", sock.recv(2))[0]
    elif length == 127:
        length = struct.unpack(">Q", sock.recv(8))[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            break
        data += chunk
    return data.decode(errors="replace")

js_code = sys.argv[1]
msg = json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression": js_code, "returnByValue": True}})
ws_send(sock, msg)

result_data = ws_recv(sock)
sock.close()

try:
    result = json.loads(result_data)
    r = result.get("result", {}).get("result", {})
    if r.get("subtype") == "error":
        print(json.dumps({"error": r.get("description", "Unknown error")}))
    else:
        print(json.dumps({"value": r.get("value"), "type": r.get("type", "undefined")}))
except Exception as e:
    print(json.dumps({"error": f"Failed to parse CDP response: {e}"}))
"""


class BrowserConsoleExecInput(BaseModel):
    """Input schema for browser console exec tool."""

    javascript: str = Field(
        description="JavaScript code to execute in the browser console"
    )
    user_id: str | None = Field(
        default=None,
        description="User ID for session management (injected by system)",
    )
    task_id: str | None = Field(
        default=None,
        description="Task ID for session management (injected by system)",
    )


@tool(args_schema=BrowserConsoleExecInput)
async def browser_console_exec(
    javascript: str,
    user_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Execute JavaScript in the browser console via CDP.

    Evaluates JavaScript code in the context of the current browser page
    using the Chrome DevTools Protocol. Use this for:
    - DOM manipulation and querying
    - SPA state inspection
    - Extracting data from the page
    - Triggering events programmatically
    - Debugging and testing

    The browser must have been launched by a previous browser_navigate call.

    Args:
        javascript: JavaScript code to execute
        user_id: User ID for session management (injected by system)
        task_id: Task ID for session management (injected by system)

    Returns:
        JSON string with the evaluation result or error
    """
    if not is_desktop_sandbox_available():
        return _error_response(error="Desktop sandbox not available. Check SANDBOX_PROVIDER configuration.")

    logger.info("browser_console_exec_invoked", js_length=len(javascript))

    try:
        _, session = await _get_browser_session(user_id, task_id)

        if not session:
            return _error_response(error="No active browser sandbox session. Use browser_navigate first.")

        # Write the CDP evaluation script to the sandbox
        import shlex

        safe_js = shlex.quote(javascript)
        cmd = f"python3 -c {shlex.quote(_CDP_EVAL_SCRIPT)} {safe_js}"
        stdout, stderr, exit_code = await session.executor.run_command(cmd, timeout_ms=15000)

        if exit_code != 0:
            logger.warning(
                "browser_console_exec_command_failed",
                exit_code=exit_code,
                stderr=stderr[:200] if stderr else "",
            )
            return _error_response(
                error=f"JavaScript execution failed: {stderr[:500] if stderr else 'Unknown error'}",
                sandbox_id=session.sandbox_id,
            )

        # Parse the result from the CDP script
        try:
            result = json.loads(stdout.strip())
        except json.JSONDecodeError:
            result = {"value": stdout.strip(), "type": "string"}

        if "error" in result:
            logger.warning("browser_console_exec_js_error", error=result["error"])
            return _error_response(error=result["error"], sandbox_id=session.sandbox_id)

        logger.info("browser_console_exec_completed", result_type=result.get("type"))

        return _success_response(
            result=result.get("value"),
            result_type=result.get("type", "undefined"),
            sandbox_id=session.sandbox_id,
        )

    except Exception as e:
        logger.error("browser_console_exec_failed", error=str(e))
        return _error_response(error=str(e))


class BrowserSelectOptionInput(BaseModel):
    """Input schema for browser select option tool."""

    selector: str = Field(
        description=(
            "CSS selector for the <select> element"
            " (e.g., '#country', 'select[name=\"lang\"]')"
        )
    )
    value: str = Field(
        description="The value to select (matches the 'value' attribute of an <option>)"
    )
    user_id: str | None = Field(
        default=None,
        description="User ID for session management (injected by system)",
    )
    task_id: str | None = Field(
        default=None,
        description="Task ID for session management (injected by system)",
    )


@tool(args_schema=BrowserSelectOptionInput)
async def browser_select_option(
    selector: str,
    value: str,
    user_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Select an option from a dropdown/select element.

    Finds a <select> element by CSS selector and sets its value,
    then dispatches a 'change' event to trigger any associated handlers.

    Use this instead of clicking and typing to interact with dropdown menus.

    Args:
        selector: CSS selector for the <select> element
        value: The option value to select
        user_id: User ID for session management (injected by system)
        task_id: Task ID for session management (injected by system)

    Returns:
        JSON string with the selection result
    """
    if not is_desktop_sandbox_available():
        return _error_response(error="Desktop sandbox not available. Check SANDBOX_PROVIDER configuration.")

    logger.info("browser_select_option_invoked", selector=selector, value=value)

    try:
        _, session = await _get_browser_session(user_id, task_id)

        if not session:
            return _error_response(error="No active browser sandbox session. Use browser_navigate first.")

        # Build JS to select the option and dispatch change event
        # Escape quotes in selector and value for safe embedding in JS
        js_selector = selector.replace("\\", "\\\\").replace("'", "\\'")
        js_value = value.replace("\\", "\\\\").replace("'", "\\'")
        js_code = (
            f"(function() {{"
            f"  var el = document.querySelector('{js_selector}');"
            f"  if (!el) return JSON.stringify("
            f"{{error: 'Element not found: {js_selector}'}});"
            f"  if (el.tagName !== 'SELECT') return JSON.stringify("
            f"{{error: 'Element is not a <select>: ' + el.tagName}});"
            f"  el.value = '{js_value}';"
            f"  el.dispatchEvent(new Event('change', {{bubbles: true}}));"
            f"  var txt = el.options[el.selectedIndex]"
            f" ? el.options[el.selectedIndex].text : '';"
            f"  return JSON.stringify({{selected: el.value, text: txt}});"
            f"}})()"
        )

        import shlex

        safe_js = shlex.quote(js_code)
        cmd = f"python3 -c {shlex.quote(_CDP_EVAL_SCRIPT)} {safe_js}"
        stdout, stderr, exit_code = await session.executor.run_command(cmd, timeout_ms=15000)

        if exit_code != 0:
            logger.warning(
                "browser_select_option_command_failed",
                exit_code=exit_code,
                stderr=stderr[:200] if stderr else "",
            )
            return _error_response(
                error=f"Select option failed: {stderr[:500] if stderr else 'Unknown error'}",
                sandbox_id=session.sandbox_id,
            )

        # Parse the CDP result
        try:
            cdp_result = json.loads(stdout.strip())
        except json.JSONDecodeError:
            cdp_result = {"error": f"Unexpected output: {stdout.strip()[:200]}"}

        if "error" in cdp_result:
            logger.warning("browser_select_option_error", error=cdp_result["error"])
            return _error_response(error=cdp_result["error"], sandbox_id=session.sandbox_id)

        # The JS returns a JSON string as the value, parse it
        inner_result = cdp_result.get("value", "")
        if isinstance(inner_result, str):
            try:
                inner_result = json.loads(inner_result)
            except (json.JSONDecodeError, TypeError):
                pass

        if isinstance(inner_result, dict) and "error" in inner_result:
            logger.warning("browser_select_option_js_error", error=inner_result["error"])
            return _error_response(error=inner_result["error"], sandbox_id=session.sandbox_id)

        if isinstance(inner_result, dict):
            selected_value = inner_result.get("selected", value)
            selected_text = inner_result.get("text", "")
        else:
            selected_value = value
            selected_text = ""

        logger.info(
            "browser_select_option_completed",
            selector=selector,
            selected_value=selected_value,
        )

        return _success_response(
            selector=selector,
            value=selected_value,
            text=selected_text,
            sandbox_id=session.sandbox_id,
        )

    except Exception as e:
        logger.error("browser_select_option_failed", error=str(e))
        return _error_response(error=str(e))


class BrowserWaitForElementInput(BaseModel):
    """Input schema for browser wait for element tool."""

    selector: str = Field(
        description=(
            "CSS selector for the element to wait for"
            " (e.g., '#result', '.loaded', '[data-ready]')"
        )
    )
    timeout: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Maximum time to wait in seconds (1-30)",
    )
    user_id: str | None = Field(
        default=None,
        description="User ID for session management (injected by system)",
    )
    task_id: str | None = Field(
        default=None,
        description="Task ID for session management (injected by system)",
    )


@tool(args_schema=BrowserWaitForElementInput)
async def browser_wait_for_element(
    selector: str,
    timeout: int = 10,
    user_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Wait for an element to appear in the DOM.

    Polls the page at regular intervals until the specified CSS selector
    matches an element, or the timeout is reached. Use this for:
    - Waiting for dynamic content to load
    - Waiting for AJAX requests to complete
    - Ensuring an element exists before interacting with it

    Args:
        selector: CSS selector for the element to wait for
        timeout: Maximum time to wait in seconds (default: 10, max: 30)
        user_id: User ID for session management (injected by system)
        task_id: Task ID for session management (injected by system)

    Returns:
        JSON string indicating whether the element was found or timed out
    """
    if not is_desktop_sandbox_available():
        return _error_response(error="Desktop sandbox not available. Check SANDBOX_PROVIDER configuration.")

    logger.info("browser_wait_for_element_invoked", selector=selector, timeout=timeout)

    try:
        _, session = await _get_browser_session(user_id, task_id)

        if not session:
            return _error_response(error="No active browser sandbox session. Use browser_navigate first.")

        # Escape quotes in selector for safe embedding in JS
        js_selector = selector.replace("\\", "\\\\").replace("'", "\\'")
        check_js = f"!!document.querySelector('{js_selector}')"

        import shlex

        poll_interval = 0.5  # seconds
        elapsed = 0.0

        while elapsed < timeout:
            safe_js = shlex.quote(check_js)
            cmd = f"python3 -c {shlex.quote(_CDP_EVAL_SCRIPT)} {safe_js}"
            stdout, stderr, exit_code = await session.executor.run_command(cmd, timeout_ms=10000)

            if exit_code == 0:
                try:
                    cdp_result = json.loads(stdout.strip())
                    if cdp_result.get("value") is True:
                        logger.info(
                            "browser_wait_for_element_found",
                            selector=selector,
                            elapsed=elapsed,
                        )
                        return _success_response(
                            selector=selector,
                            found=True,
                            elapsed_seconds=round(elapsed, 1),
                            sandbox_id=session.sandbox_id,
                        )
                except json.JSONDecodeError:
                    pass

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        logger.warning(
            "browser_wait_for_element_timeout",
            selector=selector,
            timeout=timeout,
        )

        return _error_response(
            error=f"Timeout: element '{selector}' not found after {timeout} seconds",
            selector=selector,
            timeout=timeout,
            sandbox_id=session.sandbox_id,
        )

    except Exception as e:
        logger.error("browser_wait_for_element_failed", error=str(e))
        return _error_response(error=str(e))


class BrowserDomQueryInput(BaseModel):
    """Input schema for browser DOM query tool."""

    selector: str = Field(
        description="CSS selector to query (e.g., '.result-item a', 'table tr')"
    )
    max_results: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum matched elements to return",
    )
    include_html: bool = Field(
        default=False,
        description="Whether to include outerHTML snippet for each matched element",
    )
    user_id: str | None = Field(
        default=None,
        description="User ID for session management (injected by system)",
    )
    task_id: str | None = Field(
        default=None,
        description="Task ID for session management (injected by system)",
    )


@tool(args_schema=BrowserDomQueryInput)
async def browser_dom_query(
    selector: str,
    max_results: int = 20,
    include_html: bool = False,
    user_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Query DOM elements using a CSS selector and return structured results.

    Use this when coordinate-based clicking is unreliable. This tool inspects
    the current page DOM and returns matched elements with useful fields:
    text, tagName, href/src, id, className, and optional outerHTML snippet.
    """
    if not is_desktop_sandbox_available():
        return _error_response(error="Desktop sandbox not available. Check SANDBOX_PROVIDER configuration.")

    logger.info(
        "browser_dom_query_invoked",
        selector=selector,
        max_results=max_results,
        include_html=include_html,
    )

    try:
        _, session = await _get_browser_session(user_id, task_id)
        if not session:
            return _error_response(error="No active browser sandbox session. Use browser_navigate first.")

        js_selector = selector.replace("\\", "\\\\").replace("'", "\\'")
        include_html_js = "true" if include_html else "false"
        js_code = (
            f"(function() {{"
            f"  var nodes = Array.from(document.querySelectorAll('{js_selector}')).slice(0, {max_results});"
            f"  var items = nodes.map(function(el) {{"
            f"    return {{"
            f"      tag: el.tagName ? el.tagName.toLowerCase() : '',"
            f"      text: (el.innerText || el.textContent || '').trim(),"
            f"      id: el.id || '',"
            f"      className: el.className || '',"
            f"      href: el.href || '',"
            f"      src: el.src || '',"
            f"      value: (typeof el.value !== 'undefined') ? String(el.value) : '',"
            f"      outer_html: {include_html_js} ? (el.outerHTML || '').slice(0, 500) : ''"
            f"    }};"
            f"  }});"
            f"  return JSON.stringify({{count: items.length, selector: '{js_selector}', items: items}});"
            f"}})()"
        )

        import shlex

        safe_js = shlex.quote(js_code)
        cmd = f"python3 -c {shlex.quote(_CDP_EVAL_SCRIPT)} {safe_js}"
        stdout, stderr, exit_code = await session.executor.run_command(cmd, timeout_ms=15000)

        if exit_code != 0:
            return _error_response(
                error=f"DOM query failed: {stderr[:500] if stderr else 'Unknown error'}",
                selector=selector,
                sandbox_id=session.sandbox_id,
            )

        try:
            cdp_result = json.loads(stdout.strip())
        except json.JSONDecodeError:
            return _error_response(
                error=f"Invalid DOM query output: {stdout[:200]}",
                selector=selector,
                sandbox_id=session.sandbox_id,
            )

        if "error" in cdp_result:
            return _error_response(
                error=str(cdp_result["error"]),
                selector=selector,
                sandbox_id=session.sandbox_id,
            )

        value = cdp_result.get("value", "")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass

        if not isinstance(value, dict):
            value = {"count": 0, "selector": selector, "items": []}

        logger.info(
            "browser_dom_query_completed",
            selector=selector,
            count=value.get("count", 0),
        )
        return _success_response(
            selector=selector,
            count=value.get("count", 0),
            items=value.get("items", []),
            sandbox_id=session.sandbox_id,
        )
    except Exception as e:
        logger.error("browser_dom_query_failed", error=str(e))
        return _error_response(error=str(e), selector=selector)


# JavaScript snippet to capture the accessibility tree via CDP.
# Uses Accessibility.getFullAXTree and formats the result as indented text.
_A11Y_TREE_SCRIPT = r"""
import json, sys, urllib.request, socket, struct, base64, os

max_depth = int(sys.argv[1]) if len(sys.argv) > 1 else 5
include_roles_raw = sys.argv[2] if len(sys.argv) > 2 else ""
include_roles = set(include_roles_raw.split(",")) if include_roles_raw else None

# Discover browser CDP endpoint
try:
    with urllib.request.urlopen("http://localhost:9222/json/list", timeout=5) as r:
        targets = json.loads(r.read())
except Exception:
    targets = []

ws_url = None
for t in targets:
    if t.get("type") == "page":
        ws_url = t.get("webSocketDebuggerUrl")
        break

if not ws_url:
    print(json.dumps({"error": "No browser page found via CDP"}))
    sys.exit(0)

# Parse ws URL
host_port = ws_url.replace("ws://", "").split("/")[0]
path = "/" + "/".join(ws_url.replace("ws://", "").split("/")[1:])
host, port = host_port.split(":")
port = int(port)

# WebSocket handshake
key = base64.b64encode(os.urandom(16)).decode()
sock = socket.create_connection((host, port), timeout=10)
req = (
    f"GET {path} HTTP/1.1\r\n"
    f"Host: {host_port}\r\n"
    f"Upgrade: websocket\r\n"
    f"Connection: Upgrade\r\n"
    f"Sec-WebSocket-Key: {key}\r\n"
    f"Sec-WebSocket-Version: 13\r\n\r\n"
)
sock.sendall(req.encode())
resp = b""
while b"\r\n\r\n" not in resp:
    resp += sock.recv(4096)

def ws_send(sock, data):
    payload = data.encode()
    frame = bytearray()
    frame.append(0x81)
    mask_key = os.urandom(4)
    length = len(payload)
    if length < 126:
        frame.append(0x80 | length)
    elif length < 65536:
        frame.append(0x80 | 126)
        frame.extend(struct.pack(">H", length))
    else:
        frame.append(0x80 | 127)
        frame.extend(struct.pack(">Q", length))
    frame.extend(mask_key)
    masked = bytearray(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    frame.extend(masked)
    sock.sendall(frame)

def ws_recv(sock):
    header = sock.recv(2)
    if len(header) < 2:
        return ""
    length = header[1] & 0x7F
    if length == 126:
        length = struct.unpack(">H", sock.recv(2))[0]
    elif length == 127:
        length = struct.unpack(">Q", sock.recv(8))[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            break
        data += chunk
    return data.decode(errors="replace")

# Get accessibility tree
msg = json.dumps({"id": 1, "method": "Accessibility.getFullAXTree", "params": {"depth": max_depth}})
ws_send(sock, msg)
result_data = ws_recv(sock)
sock.close()

try:
    result = json.loads(result_data)
    nodes = result.get("result", {}).get("nodes", [])
except Exception as e:
    print(json.dumps({"error": f"Failed to parse response: {e}"}))
    sys.exit(0)

# Build tree structure
node_map = {}
for n in nodes:
    nid = n.get("nodeId")
    role_obj = n.get("role", {})
    role = role_obj.get("value", "") if isinstance(role_obj, dict) else str(role_obj)
    name_obj = n.get("name", {})
    name = name_obj.get("value", "") if isinstance(name_obj, dict) else str(name_obj)
    value_obj = n.get("value", {})
    value = value_obj.get("value", "") if isinstance(value_obj, dict) else ""

    props = {}
    for p in n.get("properties", []):
        pname = p.get("name", "")
        pval = p.get("value", {})
        pv = pval.get("value", "") if isinstance(pval, dict) else str(pval)
        if pv and pv not in ("false", "0", ""):
            props[pname] = pv

    node_map[nid] = {
        "role": role,
        "name": name,
        "value": value,
        "children": n.get("childIds", []),
        "props": props,
        "ignored": n.get("ignored", False),
    }

# Format as indented text
lines = []
def format_node(nid, depth=0):
    if depth > max_depth:
        return
    node = node_map.get(nid)
    if not node or node["ignored"]:
        return

    role = node["role"]
    if include_roles and role not in include_roles:
        # Still recurse children
        for cid in node["children"]:
            format_node(cid, depth)
        return

    indent = "  " * depth
    parts = [f"[{role}]"]
    if node["name"]:
        parts.append(f'"{node["name"]}"')
    if node["value"]:
        parts.append(f'value="{node["value"]}"')

    # Mark interactive elements
    focusable = node["props"].get("focusable")
    if focusable and focusable != "false":
        parts.append("(interactive)")

    lines.append(indent + " ".join(parts))

    for cid in node["children"]:
        format_node(cid, depth + 1)

# Find root node (first node)
if nodes:
    root_id = nodes[0].get("nodeId")
    format_node(root_id)

tree_text = "\n".join(lines)
# Limit output to ~5KB
if len(tree_text) > 5000:
    tree_text = tree_text[:5000] + "\n... [truncated]"

print(json.dumps({"tree": tree_text, "node_count": len(nodes)}))
"""


class BrowserGetAccessibilityTreeInput(BaseModel):
    """Input schema for browser accessibility tree tool."""

    max_depth: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum depth of the accessibility tree (1-10)",
    )
    include_roles: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of ARIA roles to include (filters the tree). "
            "Common roles: button, link, textbox, heading, img, list, listitem, "
            "navigation, main, form, checkbox, radio, combobox, menuitem"
        ),
    )
    user_id: str | None = Field(
        default=None,
        description="User ID for session management (injected by system)",
    )
    task_id: str | None = Field(
        default=None,
        description="Task ID for session management (injected by system)",
    )


@tool(args_schema=BrowserGetAccessibilityTreeInput)
async def browser_get_accessibility_tree(
    max_depth: int = 5,
    include_roles: list[str] | None = None,
    user_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Get the accessibility tree of the current page for efficient interaction.

    Returns a structured text representation of the page (2-5KB) that's much
    faster and cheaper than screenshots for understanding page structure and
    finding interactive elements.

    The tree shows element roles, names, values, and marks interactive elements.
    Use this to:
    - Understand page layout without screenshots
    - Find buttons, links, and form fields to interact with
    - Verify page content and structure
    - Navigate complex pages efficiently

    Args:
        max_depth: Maximum depth of the accessibility tree (1-10)
        include_roles: Optional list of ARIA roles to include (filters the tree)
        user_id: User ID for session management (injected by system)
        task_id: Task ID for session management (injected by system)

    Returns:
        Formatted accessibility tree with element roles, names, and interactive elements
    """
    if not is_desktop_sandbox_available():
        return _error_response(error="Desktop sandbox not available. Check SANDBOX_PROVIDER configuration.")

    logger.info(
        "browser_get_accessibility_tree_invoked",
        max_depth=max_depth,
        include_roles=include_roles,
    )

    try:
        _, session = await _get_browser_session(user_id, task_id)

        if not session:
            return _error_response(error="No active browser sandbox session. Use browser_navigate first.")

        import shlex

        roles_arg = ",".join(include_roles) if include_roles else ""
        cmd = f"python3 -c {shlex.quote(_A11Y_TREE_SCRIPT)} {max_depth} {shlex.quote(roles_arg)}"
        stdout, stderr, exit_code = await session.executor.run_command(cmd, timeout_ms=15000)

        if exit_code != 0:
            logger.warning(
                "browser_get_accessibility_tree_command_failed",
                exit_code=exit_code,
                stderr=stderr[:200] if stderr else "",
            )
            return _error_response(
                error=f"Failed to get accessibility tree: {stderr[:500] if stderr else 'Unknown error'}",
                sandbox_id=session.sandbox_id,
            )

        try:
            result = json.loads(stdout.strip())
        except json.JSONDecodeError:
            result = {"error": f"Unexpected output: {stdout.strip()[:200]}"}

        if "error" in result:
            logger.warning("browser_get_accessibility_tree_error", error=result["error"])
            return _error_response(error=result["error"], sandbox_id=session.sandbox_id)

        tree_text = result.get("tree", "")
        node_count = result.get("node_count", 0)

        logger.info(
            "browser_get_accessibility_tree_completed",
            node_count=node_count,
            tree_length=len(tree_text),
        )

        return _success_response(
            tree=tree_text,
            node_count=node_count,
            max_depth=max_depth,
            sandbox_id=session.sandbox_id,
        )

    except Exception as e:
        logger.error("browser_get_accessibility_tree_failed", error=str(e))
        return _error_response(error=str(e))
