"""Multi-agent system with LangGraph-based orchestration.

Architecture: Orchestrator + PlannerAgent + ExecutorAgent.

The Orchestrator classifies queries (simple/complex), invokes PlannerAgent
for complex tasks, dispatches ExecutorAgent per step, verifies results, and
produces final responses. Simple queries go directly to ExecutorAgent.

Usage:
    from app.agents import agent_orchestrator

    # Streaming execution
    async for event in agent_orchestrator.run(query="Hello"):
        print(event)

    # Non-streaming execution
    result = await agent_orchestrator.invoke(query="Hello")

    # Backward-compatible alias
    from app.agents import agent_supervisor  # same as agent_orchestrator
"""

# Orchestrator (canonical entry point)
from app.agents.orchestrator import (
    AgentOrchestrator,
    agent_orchestrator,
)

# Backward-compat alias: agent_supervisor points to the orchestrator
agent_supervisor = agent_orchestrator

__all__ = [
    "AgentOrchestrator",
    "agent_orchestrator",
    "agent_supervisor",
]
