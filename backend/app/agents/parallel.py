"""Parallel multi-agent execution for wide research and general-purpose tasks.

Supports two modes:
1. Research parallel: Decomposes research queries into parallel web search sub-queries
2. General parallel: Decomposes any complex task into independent sub-tasks with full tool access

Both modes follow: decompose → execute concurrently → synthesize results.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents import events
from app.ai.llm import extract_text_from_content, llm_service
from app.ai.model_tiers import ModelTier
from app.core.logging import get_logger

logger = get_logger(__name__)

# Configuration
MAX_PARALLEL_AGENTS = 10
DEFAULT_PARALLEL_AGENTS = 5
DECOMPOSITION_TIMEOUT = 30  # seconds
SUB_QUERY_TIMEOUT = 120  # seconds per sub-query
GENERAL_TASK_TIMEOUT = 300  # seconds per general sub-task (more complex)


@dataclass
class SubTask:
    """A decomposed sub-task for parallel execution."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str = ""
    focus_area: str = ""
    task_type: str = "research"  # "research" or "general"
    tools_hint: list[str] = field(default_factory=list)  # Suggested tools for this sub-task
    status: str = "pending"  # pending, running, completed, failed
    result: str = ""
    error: str | None = None
    duration_ms: int = 0
    shared_findings: dict[str, Any] = field(default_factory=dict)  # Inter-agent shared state


@dataclass
class ParallelExecutionResult:
    """Result from parallel execution."""
    sub_tasks: list[SubTask]
    synthesis: str = ""
    total_duration_ms: int = 0
    successful_count: int = 0
    failed_count: int = 0
    shared_findings: dict[str, Any] = field(default_factory=dict)  # Merged findings from all sub-tasks


DECOMPOSITION_PROMPT = """You are a research query decomposition assistant.

Given a complex research query, break it down into {max_tasks} independent sub-queries
that can be researched in parallel. Each sub-query should:
1. Be self-contained and answerable independently
2. Cover a distinct aspect of the original query
3. Together, they should comprehensively cover the original query

Respond with a JSON array of objects, each with:
- "query": The specific sub-query to research
- "focus_area": Brief label for this aspect (2-4 words)

Return ONLY valid JSON array. Example:
[
  {{"query": "What are the latest advances in transformer architectures for NLP in 2024?", "focus_area": "Architecture advances"}},
  {{"query": "How do modern LLMs handle multi-modal inputs?", "focus_area": "Multi-modal capabilities"}}
]"""


SYNTHESIS_PROMPT = """You are a research synthesis assistant.

Original research query: {original_query}

Below are research findings from {count} parallel sub-queries. Synthesize these into
a comprehensive, well-structured response that:
1. Integrates findings from all sub-queries
2. Identifies common themes and connections
3. Highlights key insights and conclusions
4. Notes any contradictions or gaps
5. Provides a cohesive narrative

Sub-query results:
{results}

Write a comprehensive synthesis. Be thorough but concise."""


GENERAL_DECOMPOSITION_PROMPT = """You are a task decomposition assistant.

Given a complex task, break it down into {max_tasks} independent sub-tasks
that can be executed in parallel. Each sub-task should:
1. Be self-contained and executable independently
2. Not depend on the output of other sub-tasks
3. Cover a distinct aspect of the original task
4. Together, they should fully accomplish the original task

Respond with a JSON array of objects, each with:
- "query": The specific sub-task description
- "focus_area": Brief label for this aspect (2-4 words)
- "tools_hint": Array of suggested tool names (e.g., ["web_search", "execute_code", "file_write"])

Return ONLY valid JSON array. Example:
[
  {{"query": "Research and implement the data model for user profiles", "focus_area": "Data model", "tools_hint": ["execute_code", "file_write"]}},
  {{"query": "Create the API endpoint for user registration", "focus_area": "API endpoint", "tools_hint": ["execute_code", "file_write"]}}
]"""


GENERAL_SYNTHESIS_PROMPT = """You are a task synthesis assistant.

Original task: {original_query}

Below are results from {count} parallel sub-tasks. Synthesize these into
a comprehensive summary that:
1. Describes what each sub-task accomplished
2. Identifies any issues or failures
3. Provides the combined output
4. Lists any remaining work

Sub-task results:
{results}

Write a clear synthesis of all completed work."""


async def decompose_query(
    query: str,
    max_tasks: int = DEFAULT_PARALLEL_AGENTS,
    provider: str | None = None,
) -> list[SubTask]:
    """Decompose a research query into parallel sub-tasks using FLASH LLM.

    Args:
        query: Original research query
        max_tasks: Maximum number of sub-tasks to create
        provider: Optional LLM provider override

    Returns:
        List of SubTask objects
    """
    llm = llm_service.get_llm_for_tier(ModelTier.LITE, provider=provider)

    try:
        async with asyncio.timeout(DECOMPOSITION_TIMEOUT):
            result = await llm.ainvoke([
                SystemMessage(content=DECOMPOSITION_PROMPT.format(max_tasks=max_tasks)),
                HumanMessage(content=f"Research query: {query}"),
            ])

        text = extract_text_from_content(result.content).strip()

        # Clean up markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        items = json.loads(text)
        if not isinstance(items, list):
            raise ValueError("Expected JSON array")

        sub_tasks = []
        for item in items[:max_tasks]:
            if isinstance(item, dict) and "query" in item:
                sub_tasks.append(SubTask(
                    query=item["query"],
                    focus_area=item.get("focus_area", "General"),
                ))

        logger.info("query_decomposed", original=query[:50], sub_task_count=len(sub_tasks))
        return sub_tasks

    except Exception as e:
        logger.warning("query_decomposition_failed", error=str(e))
        # Fallback: return the original query as a single task
        return [SubTask(query=query, focus_area="Full research")]


async def _execute_sub_task(
    sub_task: SubTask,
    provider: str | None = None,
) -> SubTask:
    """Execute a single sub-task using a lightweight research call.

    Uses PRO-tier LLM with web search context for each sub-query.

    Args:
        sub_task: The sub-task to execute
        provider: Optional LLM provider override

    Returns:
        Updated SubTask with results
    """
    import time
    start = time.time()
    sub_task.status = "running"

    try:
        # Use web_research skill for each sub-task
        from app.agents.tools.web_search import web_search

        # Do a web search for the sub-query
        search_result = await web_search.ainvoke({
            "query": sub_task.query,
            "max_results": 5,
        })

        # Use LLM to synthesize search results into a focused answer
        llm = llm_service.get_llm_for_tier(ModelTier.PRO, provider=provider)

        async with asyncio.timeout(SUB_QUERY_TIMEOUT):
            result = await llm.ainvoke([
                SystemMessage(content=(
                    "You are a research assistant. Based on the search results below, "
                    "provide a thorough, well-cited answer to the research query. "
                    "Focus on facts and key findings."
                )),
                HumanMessage(content=(
                    f"Research query: {sub_task.query}\n\n"
                    f"Search results:\n{search_result}\n\n"
                    "Provide a comprehensive answer based on these results."
                )),
            ])

        sub_task.result = extract_text_from_content(result.content)
        sub_task.status = "completed"

    except Exception as e:
        logger.warning("sub_task_failed", task_id=sub_task.id, error=str(e))
        sub_task.status = "failed"
        sub_task.error = str(e)

    sub_task.duration_ms = max(1, int((time.time() - start) * 1000))
    return sub_task


async def synthesize_results(
    original_query: str,
    sub_tasks: list[SubTask],
    provider: str | None = None,
) -> str:
    """Synthesize results from parallel sub-tasks into a comprehensive response.

    Args:
        original_query: Original research query
        sub_tasks: Completed sub-tasks with results
        provider: Optional LLM provider override

    Returns:
        Synthesized research response
    """
    successful = [t for t in sub_tasks if t.status == "completed" and t.result]

    if not successful:
        return "Unable to complete the research - all sub-queries failed."

    results_text = ""
    for i, task in enumerate(successful, 1):
        results_text += f"\n--- Sub-query {i}: {task.focus_area} ---\n"
        results_text += f"Query: {task.query}\n"
        results_text += f"Findings:\n{task.result}\n"

    llm = llm_service.get_llm_for_tier(ModelTier.PRO, provider=provider)

    try:
        result = await llm.ainvoke([
            HumanMessage(content=SYNTHESIS_PROMPT.format(
                original_query=original_query,
                count=len(successful),
                results=results_text,
            )),
        ])
        return extract_text_from_content(result.content)
    except Exception as e:
        logger.error("synthesis_failed", error=str(e))
        # Fallback: concatenate results
        return f"Research findings (synthesis unavailable):\n\n{results_text}"


class ParallelExecutor:
    """Executes research queries in parallel via task decomposition.

    Usage:
        executor = ParallelExecutor(max_agents=5)
        result = await executor.execute(
            query="Comprehensive analysis of AI agent architectures in 2024",
            provider="anthropic",
            on_progress=lambda event: print(event),
        )
    """

    def __init__(self, max_agents: int = DEFAULT_PARALLEL_AGENTS):
        self.max_agents = min(max_agents, MAX_PARALLEL_AGENTS)

    async def execute(
        self,
        query: str,
        provider: str | None = None,
        on_progress: callable | None = None,
    ) -> ParallelExecutionResult:
        """Execute a research query with parallel sub-tasks.

        Args:
            query: Research query to execute
            provider: Optional LLM provider override
            on_progress: Optional callback for progress events (receives dict events)

        Returns:
            ParallelExecutionResult with all sub-task results and synthesis
        """
        import time
        start = time.time()

        def emit(event: dict):
            if on_progress:
                on_progress(event)

        # Phase 1: Decompose
        emit(events.stage("parallel_decompose", "Decomposing query into sub-tasks...", "running"))
        sub_tasks = await decompose_query(query, max_tasks=self.max_agents, provider=provider)
        emit(events.stage("parallel_decompose", f"Created {len(sub_tasks)} sub-tasks", "completed"))

        # Emit parallel_start event with all sub-agent IDs and focus areas
        emit({
            "type": "parallel_start",
            "sub_agents": [
                {"sub_agent_id": task.id, "focus_area": task.focus_area}
                for task in sub_tasks
            ],
            "count": len(sub_tasks),
        })

        # Emit individual task events with sub_agent_id
        for task in sub_tasks:
            emit({
                "type": "parallel_task",
                "task_id": task.id,
                "sub_agent_id": task.id,
                "query": task.query,
                "focus_area": task.focus_area,
                "status": "pending",
            })

        # Phase 2: Execute in parallel
        emit(events.stage("parallel_execute", f"Researching {len(sub_tasks)} sub-queries in parallel...", "running"))

        results = await asyncio.gather(
            *[_execute_sub_task(t, provider=provider) for t in sub_tasks],
            return_exceptions=True,
        )

        # Process results (handle exceptions from gather)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                sub_tasks[i].status = "failed"
                sub_tasks[i].error = str(result)
            # Otherwise result is the updated SubTask (same object)

        # Emit completion events per task with sub_agent_id
        for task in sub_tasks:
            emit({
                "type": "parallel_task",
                "task_id": task.id,
                "sub_agent_id": task.id,
                "focus_area": task.focus_area,
                "status": task.status,
                "duration_ms": task.duration_ms,
            })

        successful = sum(1 for t in sub_tasks if t.status == "completed")
        failed = sum(1 for t in sub_tasks if t.status == "failed")
        emit(events.stage("parallel_execute", f"Completed {successful}/{len(sub_tasks)} sub-queries", "completed"))

        # Emit parallel_complete event
        emit({
            "type": "parallel_complete",
            "successful": successful,
            "failed": failed,
            "total": len(sub_tasks),
        })

        # Phase 3: Synthesize
        emit(events.stage("parallel_synthesize", "Synthesizing results...", "running"))
        synthesis = await synthesize_results(query, sub_tasks, provider=provider)
        emit(events.stage("parallel_synthesize", "Synthesis complete", "completed"))

        total_duration = max(1, int((time.time() - start) * 1000))

        return ParallelExecutionResult(
            sub_tasks=sub_tasks,
            synthesis=synthesis,
            total_duration_ms=total_duration,
            successful_count=successful,
            failed_count=failed,
        )


async def decompose_general_task(
    query: str,
    max_tasks: int = DEFAULT_PARALLEL_AGENTS,
    provider: str | None = None,
) -> list[SubTask]:
    """Decompose a general task into parallel sub-tasks using FLASH LLM.

    Unlike research decomposition, this creates sub-tasks that can use
    any available tools (code execution, file operations, etc.).

    Args:
        query: Original task description
        max_tasks: Maximum number of sub-tasks to create
        provider: Optional LLM provider override

    Returns:
        List of SubTask objects with task_type="general"
    """
    llm = llm_service.get_llm_for_tier(ModelTier.LITE, provider=provider)

    try:
        async with asyncio.timeout(DECOMPOSITION_TIMEOUT):
            result = await llm.ainvoke([
                SystemMessage(content=GENERAL_DECOMPOSITION_PROMPT.format(max_tasks=max_tasks)),
                HumanMessage(content=f"Task: {query}"),
            ])

        text = extract_text_from_content(result.content).strip()

        # Clean up markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        items = json.loads(text)
        if not isinstance(items, list):
            raise ValueError("Expected JSON array")

        sub_tasks = []
        for item in items[:max_tasks]:
            if isinstance(item, dict) and "query" in item:
                sub_tasks.append(SubTask(
                    query=item["query"],
                    focus_area=item.get("focus_area", "General"),
                    task_type="general",
                    tools_hint=item.get("tools_hint", []),
                ))

        logger.info("general_task_decomposed", original=query[:50], sub_task_count=len(sub_tasks))
        return sub_tasks

    except Exception as e:
        logger.warning("general_task_decomposition_failed", error=str(e))
        return [SubTask(query=query, focus_area="Full task", task_type="general")]


async def _execute_general_sub_task(
    sub_task: SubTask,
    provider: str | None = None,
    shared_state: dict | None = None,
) -> SubTask:
    """Execute a general sub-task using PRO-tier LLM with tool guidance.

    For general tasks, the sub-agent gets guidance on which tools to use
    but executes via LLM reasoning rather than direct tool calls.

    Args:
        sub_task: The sub-task to execute
        provider: Optional LLM provider override
        shared_state: Read-only shared state from other sub-tasks

    Returns:
        Updated SubTask with results
    """
    import time
    start = time.time()
    sub_task.status = "running"

    try:
        llm = llm_service.get_llm_for_tier(ModelTier.PRO, provider=provider)

        context_parts = [f"Sub-task: {sub_task.query}"]
        if sub_task.tools_hint:
            context_parts.append(f"Suggested tools: {', '.join(sub_task.tools_hint)}")
        if shared_state:
            context_parts.append(f"Shared context from other sub-tasks: {json.dumps(shared_state)[:1000]}")

        async with asyncio.timeout(GENERAL_TASK_TIMEOUT):
            result = await llm.ainvoke([
                SystemMessage(content=(
                    "You are a task execution assistant. Complete the given sub-task thoroughly. "
                    "Provide a detailed result of what you accomplished, including any code, "
                    "file contents, or analysis produced."
                )),
                HumanMessage(content="\n".join(context_parts)),
            ])

        sub_task.result = extract_text_from_content(result.content)
        sub_task.status = "completed"

        # Extract any findings to share with other sub-tasks
        sub_task.shared_findings = {
            "focus_area": sub_task.focus_area,
            "status": "completed",
            "summary": sub_task.result[:500],
        }

    except Exception as e:
        logger.warning("general_sub_task_failed", task_id=sub_task.id, error=str(e))
        sub_task.status = "failed"
        sub_task.error = str(e)

    sub_task.duration_ms = max(1, int((time.time() - start) * 1000))
    return sub_task


async def synthesize_general_results(
    original_query: str,
    sub_tasks: list[SubTask],
    provider: str | None = None,
) -> str:
    """Synthesize results from parallel general sub-tasks.

    Args:
        original_query: Original task description
        sub_tasks: Completed sub-tasks with results
        provider: Optional LLM provider override

    Returns:
        Synthesized task response
    """
    successful = [t for t in sub_tasks if t.status == "completed" and t.result]

    if not successful:
        return "Unable to complete the task - all sub-tasks failed."

    results_text = ""
    for i, task in enumerate(successful, 1):
        results_text += f"\n--- Sub-task {i}: {task.focus_area} ---\n"
        results_text += f"Task: {task.query}\n"
        results_text += f"Result:\n{task.result}\n"

    llm = llm_service.get_llm_for_tier(ModelTier.PRO, provider=provider)

    try:
        result = await llm.ainvoke([
            HumanMessage(content=GENERAL_SYNTHESIS_PROMPT.format(
                original_query=original_query,
                count=len(successful),
                results=results_text,
            )),
        ])
        return extract_text_from_content(result.content)
    except Exception as e:
        logger.error("general_synthesis_failed", error=str(e))
        return f"Task results (synthesis unavailable):\n\n{results_text}"


class GeneralParallelExecutor:
    """Executes any task type in parallel via decomposition.

    Extends the research-only ParallelExecutor to support general-purpose
    tasks with full tool access guidance and inter-agent shared state.

    Usage:
        executor = GeneralParallelExecutor(max_agents=3)
        result = await executor.execute(
            query="Build a REST API with user auth and a React frontend",
            provider="anthropic",
        )
    """

    def __init__(self, max_agents: int = DEFAULT_PARALLEL_AGENTS):
        self.max_agents = min(max_agents, MAX_PARALLEL_AGENTS)

    async def execute(
        self,
        query: str,
        provider: str | None = None,
        on_progress: callable | None = None,
    ) -> ParallelExecutionResult:
        """Execute a general task with parallel sub-tasks.

        Args:
            query: Task description to execute
            provider: Optional LLM provider override
            on_progress: Optional callback for progress events

        Returns:
            ParallelExecutionResult with all sub-task results and synthesis
        """
        import time
        start = time.time()

        def emit(event: dict):
            if on_progress:
                on_progress(event)

        # Phase 1: Decompose
        emit(events.stage("parallel_decompose", "Decomposing task into sub-tasks...", "running"))
        sub_tasks = await decompose_general_task(query, max_tasks=self.max_agents, provider=provider)
        emit(events.stage("parallel_decompose", f"Created {len(sub_tasks)} sub-tasks", "completed"))

        # Emit parallel_start event with all sub-agent IDs and focus areas
        emit({
            "type": "parallel_start",
            "sub_agents": [
                {"sub_agent_id": task.id, "focus_area": task.focus_area}
                for task in sub_tasks
            ],
            "count": len(sub_tasks),
        })

        for task in sub_tasks:
            emit(events.parallel_task(
                task_id=task.id,
                focus_area=task.focus_area,
                status="pending",
                query=task.query,
            ))

        # Phase 2: Execute in parallel with shared state
        emit(events.stage("parallel_execute", f"Executing {len(sub_tasks)} sub-tasks in parallel...", "running"))

        shared_state: dict[str, Any] = {}

        results = await asyncio.gather(
            *[_execute_general_sub_task(t, provider=provider, shared_state=shared_state) for t in sub_tasks],
            return_exceptions=True,
        )

        # Process results and merge shared findings
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                sub_tasks[i].status = "failed"
                sub_tasks[i].error = str(result)
            elif hasattr(result, 'shared_findings'):
                shared_state.update(result.shared_findings)

        for task in sub_tasks:
            emit(events.parallel_task(
                task_id=task.id,
                focus_area=task.focus_area,
                status=task.status,
                duration_ms=task.duration_ms,
            ))

        successful = sum(1 for t in sub_tasks if t.status == "completed")
        failed = sum(1 for t in sub_tasks if t.status == "failed")
        emit(events.stage("parallel_execute", f"Completed {successful}/{len(sub_tasks)} sub-tasks", "completed"))

        # Emit parallel_complete event
        emit({
            "type": "parallel_complete",
            "successful": successful,
            "failed": failed,
            "total": len(sub_tasks),
        })

        # Phase 3: Synthesize
        emit(events.stage("parallel_synthesize", "Synthesizing results...", "running"))
        synthesis = await synthesize_general_results(query, sub_tasks, provider=provider)
        emit(events.stage("parallel_synthesize", "Synthesis complete", "completed"))

        total_duration = max(1, int((time.time() - start) * 1000))

        return ParallelExecutionResult(
            sub_tasks=sub_tasks,
            synthesis=synthesis,
            total_duration_ms=total_duration,
            successful_count=successful,
            failed_count=failed,
            shared_findings=shared_state,
        )


def is_parallelizable_query(query: str) -> bool:
    """Heuristic check if a query would benefit from parallel execution.

    Returns True for queries that:
    - Are complex research requests
    - Ask for comprehensive/detailed analysis
    - Cover multiple aspects or comparisons
    - Request multi-source synthesis
    - Are multi-part tasks that can be decomposed

    Args:
        query: The user's query

    Returns:
        True if the query likely benefits from parallel execution
    """
    query_lower = query.lower()

    # Keywords indicating comprehensive research
    # Use stems/prefixes for better matching (e.g., "compar"
    # matches "compare", "comparing", "comparison")
    comprehensive_keywords = [
        "comprehensive", "detailed", "in-depth", "thorough",
        "analy", "compar", "contrast", "evaluat",
        "research paper", "research report", "academic",
        "multiple", "various", "different aspects",
        "pros and cons", "advantages and disadvantages",
        "state of the art", "landscape", "overview",
        "survey", "review", "synthesis",
    ]

    # Keywords indicating parallelizable general tasks
    parallel_task_keywords = [
        "and also", "additionally", "as well as",
        "build both", "create both", "implement both",
        "simultaneously", "at the same time", "in parallel",
        "multiple components", "several parts",
    ]

    # Check for keyword matches
    keyword_matches = sum(1 for kw in comprehensive_keywords if kw in query_lower)
    parallel_matches = sum(1 for kw in parallel_task_keywords if kw in query_lower)

    # Check query length (longer queries tend to be more complex)
    is_long = len(query.split()) > 20

    return keyword_matches >= 2 or parallel_matches >= 1 or (keyword_matches >= 1 and is_long)
