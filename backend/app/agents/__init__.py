"""Multi-agent system with LangGraph-based orchestration.

This module provides a supervisor-based architecture where the task agent
handles all types of queries, using skills for specialized capabilities.

Available skills (invoked by task agent):
- deep_research: Deep research with web search and analysis
- data_analysis: Full data analysis with planning, code execution, summarization
- image_generation: AI image generation
- app_builder: Build web apps with live preview
- slide_generation: Create PPTX presentations
- code_generation: Generate code snippets
- web_research: Focused web research with summarization

Usage:
    from app.agents import agent_supervisor

    # Streaming execution
    async for event in agent_supervisor.run(query="Hello"):
        print(event)

    # Non-streaming execution
    result = await agent_supervisor.invoke(query="Hello")
"""

# State definitions
# Routing
from app.agents.routing import RoutingResult, route_query
from app.agents.state import (
    AgentType,
    ResearchState,
    SupervisorState,
    TaskState,
)

# Subagents
from app.agents.subagents import (
    task_subgraph,
)

# Supervisor
from app.agents.supervisor import (
    AgentSupervisor,
    agent_supervisor,
    create_supervisor_graph,
)

__all__ = [
    # State types
    "AgentType",
    "SupervisorState",
    "TaskState",
    "ResearchState",
    # Routing
    "route_query",
    "RoutingResult",
    # Supervisor
    "AgentSupervisor",
    "agent_supervisor",
    "create_supervisor_graph",
    # Subagents
    "task_subgraph",
]
