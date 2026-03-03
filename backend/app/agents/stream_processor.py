"""Stream processor for normalizing and handling agent events."""

import uuid
from typing import Any, AsyncGenerator

from app.agents import events as agent_events
from app.core.logging import get_logger
from app.sandbox import get_desktop_sandbox_manager, is_desktop_sandbox_available

logger = get_logger(__name__)

# Streaming configuration - declarative mapping of which nodes should stream tokens
STREAMING_CONFIG: dict[str, bool] = {
    "summarize": True,
    "agent": True,
    "reason": True,  # Chat agent's reason node - stream LLM responses to user
    "write": True,  # Research agent's write node - stream report to user
    "generate": False,  # Code generation - internal step, don't stream to chat
    "synthesize": True,
    "analyze": False,  # Analysis step - internal, only show final summary
    "router": False,
    "tools": False,
    "search_agent": False,
    "search_tools": False,
    # Research subgraph nodes
    "research_post": False,
    "init_config": False,
    "collect_sources": False,
}

# Browser tools set
BROWSER_TOOLS: set[str] = {
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_screenshot",
    "browser_scroll",
    "browser_press_key",
}

# Mapping of node suffixes to stage information
# Format: (suffix, stage_name, running_desc, completed_desc)
STAGE_DEFINITIONS: list[tuple[str, str, str, str | None]] = [
    ("plan", "plan", "Planning analysis...", "Analysis planned"),
    ("generate", "generate", "Generating code...", "Code generated"),
    ("execute", "execute", "Executing code...", "Code executed"),
    ("summarize", "summarize", "Summarizing results...", "Results summarized"),
    ("init_config", "config", "Initializing research...", "Research configured"),
    ("search_agent", "search", "Searching for sources...", None),  # Loops
    ("search_tools", "search_tools", "Executing search...", "Search executed"),
    ("collect_sources", "collect", "Collecting sources...", "Sources collected"),
    ("synthesize", "synthesize", "Synthesizing findings...", "Findings synthesized"),
    ("analyze", "analyze", "Analyzing sources...", "Sources analyzed"),
    ("write", "report", "Writing research report...", "Research report complete"),
]

# Browser action descriptions
BROWSER_ACTION_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "browser_navigate": ("navigate", "Navigating to URL"),
    "browser_click": ("click", "Clicking on element"),
    "browser_type": ("type", "Typing text"),
    "browser_screenshot": ("screenshot", "Taking screenshot"),
    "browser_scroll": ("scroll", "Scrolling page"),
    "browser_press_key": ("key", "Pressing key"),
}

# Content length limits for tool results
TOOL_CONTENT_LIMITS: dict[str, int] = {
    "invoke_skill": 5000,  # Extra space for skill output with preview URLs
    "generate_slides": 3000,  # Slide output with download URL and sources
    "app_start_server": 3000,  # Extra space for terminal events
    "app_get_preview_url": 2000,
    "create_app_project": 3000,  # Extra space for terminal events
    "app_run_command": 3000,  # Extra space for terminal events
    "app_install_packages": 3000,  # Extra space for terminal events
    "shell_exec": 3000,  # Extra space for terminal events
    "execute_code": 3000,  # Extra space for terminal events
}
DEFAULT_CONTENT_LIMIT = 500


class StreamProcessor:
    """Process and normalize events from the agent graph."""

    def __init__(
        self,
        user_id: str | None,
        task_id: str | None,
        thread_id: str,
        run_id: str | None = None,
    ):
        self.user_id = user_id
        self.task_id = task_id
        self.thread_id = thread_id
        self.run_id = run_id

        # State tracking
        self.emitted_tool_call_ids: set[str] = set()
        self.emitted_stage_keys: set[str] = set()
        self.emitted_image_indices: set[int] = set()
        self.emitted_interrupt_ids: set[str] = set()  # Track emitted HITL interrupts
        self.emitted_skill_output_keys: set[str] = set()  # Track emitted skill_output events
        self.emitted_terminal_keys: set[str] = set()  # Dedup terminal/workspace events
        self.pending_tool_calls: dict[str, dict] = {}
        self.pending_tool_calls_by_tool: dict[str, list[str]] = {}
        self.node_path: list[str] = []
        self.current_content_node: str | None = None
        self.streamed_tokens: bool = False
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cached_tokens: int = 0

    def node_matches_streaming(self, node_name: str, node_key: str) -> bool:
        """Check if a node name matches a streaming config key."""
        return node_name == node_key or node_name.endswith(f":{node_key}")

    def path_has_streamable_node(self, node_path_str: str) -> bool:
        """Check if any node in the current path should stream."""
        if not node_path_str:
            return False
        for segment in node_path_str.split("/"):
            for node, enabled in STREAMING_CONFIG.items():
                if enabled and self.node_matches_streaming(segment, node):
                    return True
        return False

    async def process_event(self, event: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Process a single LangGraph event and yield normalized events."""
        event_type = event.get("event")
        node_name = event.get("name", "")

        if event_type == "on_chain_start":
            async for e in self._handle_chain_start(node_name):
                yield e

        elif event_type == "on_chain_end":
            async for e in self._handle_chain_end(node_name, event):
                yield e

        elif event_type == "on_chat_model_stream":
            async for e in self._handle_chat_stream(event):
                yield e

        elif event_type == "on_tool_start":
            async for e in self._handle_tool_start(event):
                yield e

        elif event_type == "on_tool_end":
            async for e in self._handle_tool_end(event):
                yield e

        elif event_type == "on_chain_error":
            async for e in self._handle_chain_error(node_name, event):
                yield e

        elif event_type == "on_chat_model_end":
            async for e in self._handle_chat_model_end(event):
                yield e

        elif event_type == "on_custom_event":
            async for e in self._handle_custom_event(event):
                yield e

    async def _handle_chain_start(self, node_name: str) -> AsyncGenerator[dict[str, Any], None]:
        self.node_path.append(node_name)

        # Track content-generating nodes
        if any(
            enabled and self.node_matches_streaming(node_name, node)
            for node, enabled in STREAMING_CONFIG.items()
        ):
            self.current_content_node = node_name

        # Detect and emit stage running events via configuration
        for suffix, stage_name, desc, _ in STAGE_DEFINITIONS:
            if (
                node_name == suffix
                or node_name.endswith(f":{suffix}")
                or node_name.endswith(suffix)
            ):
                # Special case for analyze: distinguish research analyze from others if needed
                if stage_name == "analyze" and not any("research" in p for p in self.node_path):
                    continue
                # Special case for write: require research path
                if stage_name == "report" and not any("research" in p for p in self.node_path):
                    continue

                stage_key = f"{stage_name}:running"
                if stage_key not in self.emitted_stage_keys:
                    self.emitted_stage_keys.add(stage_key)
                    yield agent_events.stage(
                        name=stage_name,
                        description=desc,
                        status="running",
                    )

    async def _handle_chain_end(
        self, node_name: str, event: dict
    ) -> AsyncGenerator[dict[str, Any], None]:
        # Update node path
        if self.node_path and self.node_path[-1] == node_name:
            self.node_path.pop()
        if node_name == self.current_content_node:
            self.current_content_node = None

        # Thinking stage completion
        if node_name == "router" and "thinking" not in self.emitted_stage_keys:
            self.emitted_stage_keys.add("thinking")
            yield agent_events.stage(
                name="thinking",
                description="Request processed",
                status="completed",
            )

        # Stage completions via configuration
        for suffix, stage_name, _, completed_desc in STAGE_DEFINITIONS:
            if not completed_desc:
                continue
            if (
                node_name == suffix
                or node_name.endswith(f":{suffix}")
                or node_name.endswith(suffix)
            ):
                # Same special cases
                if stage_name == "analyze" and not any("research" in p for p in self.node_path):
                    continue
                if stage_name == "report" and not any("research" in p for p in self.node_path):
                    continue

                stage_key = f"{stage_name}:completed"
                if stage_key not in self.emitted_stage_keys:
                    self.emitted_stage_keys.add(stage_key)
                    yield agent_events.stage(
                        name=stage_name,
                        description=completed_desc,
                        status="completed",
                    )

        # Extract and forward events from subagent output
        event_data = event.get("data") or {}
        output = event_data.get("output") or {}
        if isinstance(output, dict):
            events_list = output.get("events", [])
            if isinstance(events_list, list):
                for e in events_list:
                    if isinstance(e, dict):
                        # Filter/Deduplicate logic
                        if self._should_emit_subevent(e, node_name):
                            yield e

    def _should_emit_subevent(self, e: dict, node_name: str) -> bool:
        event_type = e.get("type")

        # Token deduplication
        is_programmatic_node = any(
            keyword in node_name.lower() for keyword in ["present", "image", "summarize"]
        )
        if event_type == "token" and self.streamed_tokens and not is_programmatic_node:
            return False

        # Stage deduplication
        if event_type == "stage":
            stage_key = f"{e.get('name')}:{e.get('status')}"
            if stage_key in self.emitted_stage_keys:
                return False
            self.emitted_stage_keys.add(stage_key)

        # Tool call deduplication
        if event_type == "tool_call":
            tool_id = e.get("id")
            if tool_id and tool_id in self.emitted_tool_call_ids:
                return False
            if tool_id:
                self.emitted_tool_call_ids.add(tool_id)

        # Tool result deduplication
        if event_type == "tool_result":
            tool_id = e.get("id")
            result_key = f"result:{tool_id}"
            if tool_id and result_key in self.emitted_tool_call_ids:
                return False
            if tool_id:
                self.emitted_tool_call_ids.add(result_key)

        # Image deduplication
        if event_type == "image":
            image_index = e.get("index", 0)
            if image_index in self.emitted_image_indices:
                return False
            self.emitted_image_indices.add(image_index)
            logger.debug("yielding_image_event", index=image_index, node_name=node_name)

        # Skill output deduplication
        if event_type == "skill_output":
            skill_id = e.get("skill_id", "")
            output = e.get("output") or {}
            # Key by skill_id + download_url (or preview_url) for uniqueness
            dedup_value = ""
            if isinstance(output, dict):
                dedup_value = output.get("download_url") or output.get("preview_url") or ""
            skill_key = f"{skill_id}:{dedup_value}"
            if skill_key in self.emitted_skill_output_keys:
                return False
            self.emitted_skill_output_keys.add(skill_key)

        # Terminal and workspace event deduplication
        # These can arrive via multiple paths: custom event (real-time),
        # tool_end JSON extraction, and chain_end state extraction.
        if event_type in (
            "terminal_command",
            "terminal_output",
            "terminal_error",
            "terminal_complete",
            "workspace_update",
            "browser_stream",
        ):
            # Build a content-based dedup key (include timestamp so events
            # from different tool calls with similar content don't collide)
            ts = e.get("timestamp", "")
            if event_type == "terminal_command":
                key = f"tcmd:{ts}:{e.get('command', '')}"
            elif event_type == "terminal_output":
                key = f"tout:{ts}:{e.get('content', '')[:200]}"
            elif event_type == "terminal_error":
                key = f"terr:{ts}:{e.get('content', '')[:200]}"
            elif event_type == "terminal_complete":
                key = f"tcmp:{ts}:{e.get('exit_code', '')}"
            elif event_type == "workspace_update":
                key = f"ws:{e.get('operation', '')}:{e.get('path', '')}"
            elif event_type == "browser_stream":
                key = f"bs:{e.get('stream_url', '')}"
            else:
                key = ""

            if key and key in self.emitted_terminal_keys:
                return False
            if key:
                self.emitted_terminal_keys.add(key)

        # Interrupt deduplication (HITL events)
        if event_type == "interrupt":
            interrupt_id = e.get("interrupt_id")
            if interrupt_id and interrupt_id in self.emitted_interrupt_ids:
                logger.debug(
                    "interrupt_deduplicated",
                    interrupt_id=interrupt_id,
                    message=e.get("message", "")[:50],
                )
                return False
            if interrupt_id:
                self.emitted_interrupt_ids.add(interrupt_id)
                logger.debug(
                    "interrupt_emitting",
                    interrupt_id=interrupt_id,
                    message=e.get("message", "")[:50],
                )

        return True

    async def _handle_chat_stream(self, event: dict) -> AsyncGenerator[dict[str, Any], None]:
        node_path_str = "/".join(self.node_path)
        chunk = (event.get("data") or {}).get("chunk")

        # Tool call chunks
        if chunk and getattr(chunk, "tool_calls", None):
            for tool_call in chunk.tool_calls:
                if not tool_call or not isinstance(tool_call, dict):
                    continue
                func = (
                    tool_call.get("function")
                    if isinstance(tool_call.get("function"), dict)
                    else None
                )
                tool_name = (
                    tool_call.get("name")
                    or tool_call.get("tool")
                    or (func.get("name") if func else None)
                )

                if not tool_name:
                    continue

                tool_id = tool_call.get("id") or tool_call.get("tool_call_id") or str(uuid.uuid4())

                if tool_id in self.emitted_tool_call_ids:
                    continue

                self.emitted_tool_call_ids.add(tool_id)
                tool_args = tool_call.get("args") or {}

                self.pending_tool_calls[tool_id] = {
                    "tool": tool_name,
                    "args": tool_args,
                }
                self.pending_tool_calls_by_tool.setdefault(tool_name, []).append(tool_id)

                yield agent_events.tool_call(
                    tool=tool_name,
                    args=tool_args,
                    tool_id=tool_id,
                )

        # Reasoning content extraction (from thinking-mode providers like DeepSeek, Kimi, Qwen)
        if chunk and hasattr(chunk, "additional_kwargs"):
            reasoning_content = chunk.additional_kwargs.get("reasoning_content")
            if reasoning_content and isinstance(reasoning_content, str) and reasoning_content.strip():
                yield agent_events.reasoning(
                    thinking=reasoning_content,
                    context="llm_thinking",
                )

        # Streaming content tokens
        if self.path_has_streamable_node(node_path_str):
            chunk = (event.get("data") or {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                from app.ai.llm import extract_text_from_content

                content = extract_text_from_content(chunk.content)
                if content:
                    self.streamed_tokens = True
                    yield {"type": "token", "content": content}

    async def _handle_chat_model_end(self, event: dict) -> AsyncGenerator[dict[str, Any], None]:
        """Extract token usage from on_chat_model_end events.

        LangGraph astream_events v2 includes usage_metadata on the output
        AIMessage when the provider reports token counts.
        """
        output = (event.get("data") or {}).get("output")
        if output is None:
            return

        usage_meta = getattr(output, "usage_metadata", None)
        if not usage_meta or not isinstance(usage_meta, dict):
            return

        input_tokens = usage_meta.get("input_tokens", 0)
        output_tokens = usage_meta.get("output_tokens", 0)
        if input_tokens == 0 and output_tokens == 0:
            return

        cached_tokens = 0
        input_detail = usage_meta.get("input_token_details") or {}
        if isinstance(input_detail, dict):
            cached_tokens = input_detail.get("cache_read", 0) or 0

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cached_tokens += cached_tokens

        resp_meta = getattr(output, "response_metadata", {}) or {}
        model_name = resp_meta.get("model_name", "")

        from app.services.usage_tracker import calculate_cost

        cost = calculate_cost(model_name, input_tokens, output_tokens, cached_tokens)

        yield agent_events.usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            model=model_name,
            tier="",
        )

    async def _handle_tool_start(self, event: dict) -> AsyncGenerator[dict[str, Any], None]:
        run_id = event.get("run_id", "")
        tool_name = event.get("name", "")
        tool_input = (event.get("data") or {}).get("input") or {}

        # Handle Browser Stream (Side Effect)
        if tool_name in BROWSER_TOOLS:
            async for e in self._emit_browser_stream(tool_input):
                yield e

        # Track tool call - deduplicate with on_chat_model_stream emissions.
        # on_chat_model_stream uses LLM-assigned tool_call_id, while on_tool_start
        # uses run_id as fallback. Skip emitting if we already have a pending
        # (unresolved) tool_call for the same tool name from on_chat_model_stream.
        tool_call_id = None
        if isinstance(tool_input, dict):
            tool_call_id = tool_input.get("tool_call_id")

        if run_id and not tool_call_id:
            tool_call_id = str(run_id)

        if tool_name and tool_call_id and tool_call_id not in self.emitted_tool_call_ids:
            # Check if a tool_call was already emitted for this tool via on_chat_model_stream.
            # Count pending (emitted but unresolved) calls for this tool name.
            pending_ids_for_tool = self.pending_tool_calls_by_tool.get(tool_name, [])
            # Count how many on_tool_start we've already processed for this tool
            # (tracked by having run_id in pending_tool_calls)
            already_started = sum(
                1 for pid in pending_ids_for_tool
                if self.pending_tool_calls.get(pid, {}).get("run_id")
            )
            unmatched_pending = len(pending_ids_for_tool) - already_started

            if unmatched_pending > 0:
                # This on_tool_start corresponds to an already-emitted tool_call from chat stream.
                # Register the run_id mapping for tool_result resolution but don't emit again.
                self.emitted_tool_call_ids.add(tool_call_id)
                # Link the run_id to the first unmatched pending call
                for pid in pending_ids_for_tool:
                    if not self.pending_tool_calls.get(pid, {}).get("run_id"):
                        self.pending_tool_calls[pid]["run_id"] = run_id
                        break
                self.pending_tool_calls[tool_call_id] = {
                    "tool": tool_name,
                    "run_id": run_id,
                }
            else:
                # No pending call from chat stream — this is a fresh tool_call (e.g., only on_tool_start fires)
                self.emitted_tool_call_ids.add(tool_call_id)
                self.pending_tool_calls[tool_call_id] = {
                    "tool": tool_name,
                    "run_id": run_id,
                }
                self.pending_tool_calls_by_tool.setdefault(tool_name, []).append(tool_call_id)

                yield agent_events.tool_call(
                    tool=tool_name,
                    args=tool_input if isinstance(tool_input, dict) else {},
                    tool_id=tool_call_id,
                )

        # Browser action events
        if tool_name in BROWSER_ACTION_DESCRIPTIONS:
            action, desc = BROWSER_ACTION_DESCRIPTIONS[tool_name]

            # Extract target from specific keys based on action
            target = ""
            if isinstance(tool_input, dict):
                if action == "navigate":
                    target = tool_input.get("url", "")
                elif action == "click":
                    target = f"({tool_input.get('x', '?')}, {tool_input.get('y', '?')})"
                elif action == "type":
                    target = tool_input.get("text", "")[:50]
                elif action == "scroll":
                    target = tool_input.get("direction", "down")
                elif action == "key":
                    target = tool_input.get("key", "")

            yield agent_events.browser_action(
                action=action,
                description=desc,
                target=target,
                status="running",
            )

    async def _emit_browser_stream(self, tool_input: Any) -> AsyncGenerator[dict[str, Any], None]:
        if not is_desktop_sandbox_available():
            return

        uid = tool_input.get("user_id") if isinstance(tool_input, dict) else None
        tid = tool_input.get("task_id") if isinstance(tool_input, dict) else None
        uid = uid if uid is not None else self.user_id
        tid = tid if tid is not None else self.task_id

        try:
            manager = get_desktop_sandbox_manager()
            session = await manager.get_or_create_sandbox(
                user_id=uid,
                task_id=tid,
                launch_browser=True,
            )
            stream_url, auth_key = await manager.ensure_stream_ready(session)
            if stream_url:
                yield agent_events.browser_stream(
                    stream_url=stream_url,
                    sandbox_id=session.sandbox_id,
                    auth_key=auth_key,
                )
        except Exception as e:
            logger.warning("browser_stream_emit_failed", error=str(e))

    @staticmethod
    def _extract_tool_output_str(output: Any) -> str:
        """Extract string content from tool output (handles both str and ToolMessage)."""
        if isinstance(output, str):
            return output
        # LangGraph returns ToolMessage objects from on_tool_end
        content = getattr(output, "content", None)
        if isinstance(content, str):
            return content
        if output:
            return str(output)
        return ""

    async def _handle_tool_end(self, event: dict) -> AsyncGenerator[dict[str, Any], None]:
        run_id = event.get("run_id", "")
        tool_name = event.get("name", "")
        raw_output = (event.get("data") or {}).get("output", "")
        output = self._extract_tool_output_str(raw_output)

        # Browser action completion
        if tool_name in BROWSER_TOOLS:
            action = BROWSER_ACTION_DESCRIPTIONS.get(tool_name, (tool_name,))[0]
            readable_name = tool_name.replace("browser_", "").replace("_", " ").title()
            yield agent_events.browser_action(
                action=action,
                description=f"{readable_name} completed",
                status="completed",
            )

        # Extract skill_output from invoke_skill results (for frontend preview URLs etc.)
        # Note: stage/terminal/browser_stream events are dispatched in real-time via
        # dispatch_custom_event and handled in _handle_custom_event — not extracted here.
        if tool_name == "invoke_skill" and output:
            try:
                import json

                parsed = json.loads(output)
                if isinstance(parsed, dict):
                    # Emit skill_output event for frontend to extract preview URLs etc.
                    if parsed.get("success") and parsed.get("output"):
                        skill_id = parsed.get("skill_id", "")
                        output_dict = parsed.get("output", {})
                        preview_url = output_dict.get("preview_url") if isinstance(output_dict, dict) else None
                        download_url = output_dict.get("download_url") if isinstance(output_dict, dict) else None
                        dedup_value = download_url or preview_url or ""
                        skill_key = f"{skill_id}:{dedup_value}"
                        if skill_key not in self.emitted_skill_output_keys:
                            self.emitted_skill_output_keys.add(skill_key)
                            skill_output_event = {
                                "type": "skill_output",
                                "skill_id": skill_id,
                                "output": output_dict,
                            }
                            logger.info(
                                "emitting_skill_output",
                                skill_id=skill_id,
                                has_preview_url="preview_url" in str(output_dict),
                                preview_url=preview_url,
                            )
                            yield skill_output_event
            except (json.JSONDecodeError, ValueError):
                pass

        # Extract slide generation output and emit as skill_output event
        if tool_name == "generate_slides" and output:
            try:
                import json

                parsed = json.loads(output)
                if (
                    isinstance(parsed, dict)
                    and parsed.get("success")
                    and parsed.get("download_url")
                ):
                    slide_event = {
                        "type": "skill_output",
                        "skill_id": "slide_generation",
                        "output": {
                            "download_url": parsed["download_url"],
                            "storage_key": parsed.get("storage_key", ""),
                            "title": parsed.get("title", ""),
                            "slide_count": parsed.get("slide_count", 0),
                            "sources": parsed.get("sources", []),
                            "slide_outline": parsed.get("slide_outline", []),
                        },
                    }
                    # Track for deduplication against chain_end forwarded events
                    self.emitted_skill_output_keys.add(
                        f"slide_generation:{parsed['download_url']}"
                    )
                    yield slide_event
                    logger.info(
                        "emitting_slide_output",
                        title=parsed.get("title", ""),
                        download_url=parsed["download_url"][:80],
                    )
            except (json.JSONDecodeError, ValueError):
                pass

        # Extract terminal/workspace events from shell_exec and execute_code
        if tool_name in ("shell_exec", "execute_code") and output:
            try:
                import json

                parsed = json.loads(output)
                if isinstance(parsed, dict):
                    terminal_events = parsed.get("terminal_events", [])
                    if terminal_events:
                        evt_types = [
                            e.get("type")
                            for e in terminal_events
                            if isinstance(e, dict)
                        ]
                        logger.info(
                            "shell_code_tool_events_extracted",
                            tool_name=tool_name,
                            event_count=len(terminal_events),
                            event_types=evt_types,
                        )
                    for evt in terminal_events:
                        if isinstance(evt, dict) and self._should_emit_subevent(evt, tool_name):
                            yield evt

                    workspace_events = parsed.get("workspace_events", [])
                    for evt in workspace_events:
                        if isinstance(evt, dict) and self._should_emit_subevent(evt, tool_name):
                            yield evt
            except (json.JSONDecodeError, ValueError):
                pass

        # Extract terminal events from app builder tools
        app_builder_tools = {
            "create_app_project",
            "app_start_server",
            "app_run_command",
            "app_install_packages",
        }
        if tool_name in app_builder_tools and output:
            try:
                import json

                parsed = json.loads(output)
                if isinstance(parsed, dict):
                    terminal_events = parsed.get("terminal_events", [])
                    if terminal_events:
                        logger.info(
                            "app_builder_tool_events_extracted",
                            tool_name=tool_name,
                            event_count=len(terminal_events),
                            event_types=[e.get("type") for e in terminal_events if isinstance(e, dict)],
                        )
                    for evt in terminal_events:
                        if isinstance(evt, dict) and self._should_emit_subevent(evt, tool_name):
                            yield evt

                    # Also emit browser_stream for app_start_server
                    if tool_name == "app_start_server" and parsed.get("success"):
                        preview_url = parsed.get("preview_url")
                        if preview_url:
                            sandbox_id = parsed.get("sandbox_id", "app-sandbox")
                            yield agent_events.browser_stream(
                                stream_url=preview_url,
                                sandbox_id=sandbox_id,
                                auth_key=None,
                            )
                            logger.info(
                                "app_builder_browser_stream_emitted",
                                preview_url=preview_url,
                                sandbox_id=sandbox_id,
                            )
            except (json.JSONDecodeError, ValueError):
                pass

        # Resolve tool call ID
        tool_call_id = self._resolve_tool_call_id(run_id, tool_name)

        # Emit tool result with appropriate content length limit
        max_content_length = TOOL_CONTENT_LIMITS.get(tool_name, DEFAULT_CONTENT_LIMIT)
        content = output[:max_content_length] if output else ""
        yield agent_events.tool_result(
            tool=tool_name,
            content=content,
            max_length=max_content_length,
            tool_id=tool_call_id,
        )

        # Cleanup pending
        self.pending_tool_calls.pop(tool_call_id, None)
        if tool_name in self.pending_tool_calls_by_tool:
            try:
                self.pending_tool_calls_by_tool[tool_name].remove(tool_call_id)
                if not self.pending_tool_calls_by_tool[tool_name]:
                    self.pending_tool_calls_by_tool.pop(tool_name, None)
            except ValueError:
                pass

    def _resolve_tool_call_id(self, run_id: str, tool_name: str) -> str:
        tool_call_id = None

        # Try finding by run_id
        if run_id:
            for tid, info in self.pending_tool_calls.items():
                if info.get("run_id") == run_id:
                    tool_call_id = tid
                    break

        # Try popping from queue by tool name
        if not tool_call_id and tool_name:
            tool_queue = self.pending_tool_calls_by_tool.get(tool_name) or []
            if tool_queue:
                tool_call_id = tool_queue.pop(0)

        return tool_call_id or str(run_id) if run_id else str(uuid.uuid4())

    async def _handle_custom_event(
        self, event: dict
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Handle custom events dispatched via dispatch_custom_event from within tools.

        This enables real-time streaming of skill events (terminal, stage,
        workspace_update, browser_stream) instead of waiting for the entire
        tool to complete and extracting them from the JSON result.
        """
        event_name = event.get("name", "")
        if event_name != "skill_event":
            return

        data = event.get("data")
        if not isinstance(data, dict):
            return

        event_type = data.get("type")
        if event_type not in (
            "stage",
            "terminal_command",
            "terminal_output",
            "terminal_error",
            "terminal_complete",
            "browser_stream",
            "workspace_update",
            "plan_overview",
            "plan_step_completed",
        ):
            return

        if self._should_emit_subevent(data, "skill_event"):
            logger.debug(
                "custom_skill_event_emitted",
                event_type=event_type,
            )
            yield data

    async def _handle_chain_error(
        self, node_name: str, event: dict
    ) -> AsyncGenerator[dict[str, Any], None]:
        error = (event.get("data") or {}).get("error")
        if error:
            yield agent_events.error(
                error_msg=str(error),
                name=node_name,
            )
            logger.error(
                "subagent_error",
                node=node_name,
                error=str(error),
                thread_id=self.thread_id,
            )
