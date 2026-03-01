"""Policy-backed tool risk helpers for HITL approvals."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from app.agents.policy import PolicyDecision, PolicyInput, get_policy_engine


class ToolRiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Backward-compatible exports; policy engine is authoritative now.
HIGH_RISK_TOOLS: set[str] = set()
MEDIUM_RISK_TOOLS: set[str] = set()


def get_tool_risk_level(tool_name: str) -> ToolRiskLevel:
    """Get policy-derived risk level for a tool."""
    from app.agents.tools.registry import get_tool_contract

    result = get_policy_engine().assess_risk(tool_name, get_tool_contract(tool_name))
    return ToolRiskLevel(result.value)


def get_skill_risk_level(skill_id: str) -> ToolRiskLevel:
    """Infer risk level for a skill from metadata + required tool surface."""
    try:
        from app.services.skill_registry import skill_registry

        skill = skill_registry.get_skill(skill_id)
        if not skill:
            return ToolRiskLevel.HIGH

        explicit_risk = getattr(skill.metadata, "risk_level", None)
        if explicit_risk in ("low", "medium", "high"):
            return ToolRiskLevel(explicit_risk)

        has_medium = False
        for tool_name in skill.metadata.required_tools or []:
            level = get_tool_risk_level(tool_name)
            if level == ToolRiskLevel.HIGH:
                return ToolRiskLevel.HIGH
            if level == ToolRiskLevel.MEDIUM:
                has_medium = True
        return ToolRiskLevel.MEDIUM if has_medium else ToolRiskLevel.LOW
    except Exception:
        return ToolRiskLevel.HIGH


def requires_approval_for_skill(
    skill_id: str,
    auto_approve_tools: list[str] | None = None,
    hitl_enabled: bool = True,
    risk_threshold: Literal["high", "medium", "all"] = "high",
) -> bool:
    """Check if invoking a skill should require user approval."""
    if not hitl_enabled:
        return False

    approved = set(auto_approve_tools or [])
    if "invoke_skill" in approved or f"invoke_skill:{skill_id}" in approved:
        return False

    fake_tool_args = {"skill_id": skill_id}
    result = get_policy_engine().decide(
        PolicyInput(
            tool_name="invoke_skill",
            tool_args=fake_tool_args,
            auto_approve_tools=auto_approve_tools,
            hitl_enabled=hitl_enabled,
            risk_threshold=risk_threshold,
            contract=_get_contract("invoke_skill"),
            is_skill_invocation=True,
        )
    )

    # Elevate invoke_skill decision using resolved skill risk.
    if result.decision == PolicyDecision.ALLOW:
        skill_risk = get_skill_risk_level(skill_id)
        if risk_threshold == "all":
            return True
        if risk_threshold == "medium" and skill_risk in {
            ToolRiskLevel.MEDIUM,
            ToolRiskLevel.HIGH,
        }:
            return True
        if risk_threshold == "high" and skill_risk == ToolRiskLevel.HIGH:
            return True
        return False
    return result.decision == PolicyDecision.REQUIRE_APPROVAL


def requires_approval(
    tool_name: str,
    auto_approve_tools: list[str] | None = None,
    hitl_enabled: bool = True,
    risk_threshold: Literal["high", "medium", "all"] = "high",
) -> bool:
    """Check if a tool requires approval under policy engine."""
    result = get_policy_engine().decide(
        PolicyInput(
            tool_name=tool_name,
            tool_args={},
            auto_approve_tools=auto_approve_tools,
            hitl_enabled=hitl_enabled,
            risk_threshold=risk_threshold,
            contract=_get_contract(tool_name),
        )
    )
    return result.decision == PolicyDecision.REQUIRE_APPROVAL


def _get_contract(tool_name: str):
    from app.agents.tools.registry import get_tool_contract

    return get_tool_contract(tool_name)


def get_tool_approval_message(tool_name: str, args: dict) -> tuple[str, str]:
    """Generate approval title/message for a tool."""
    risk_level = get_tool_risk_level(tool_name)

    if tool_name in ("browser_navigate", "computer_tool"):
        url = args.get("url", args.get("action", {}).get("text", "unknown"))
        return (
            "Browser Navigation",
            f"The agent wants to navigate to:\n\n**{url}**\n\nThis will open a browser and access external content.",
        )
    if tool_name in ("browser_click", "browser_type"):
        target = args.get("selector", args.get("text", "element"))
        action = "click" if tool_name == "browser_click" else "type into"
        return (
            f"Browser {action.title()}",
            f"The agent wants to {action} **{target}** in the browser.\n\nThis may trigger actions on the current page.",
        )
    if tool_name in ("execute_code", "code_interpreter", "python_repl", "sandbox_execute"):
        code_preview = args.get("code", "")[:200]
        if len(args.get("code", "")) > 200:
            code_preview += "..."
        return (
            "Code Execution",
            f"The agent wants to execute code:\n\n```\n{code_preview}\n```\n\nThis code will run in a sandboxed environment.",
        )
    if tool_name in ("sandbox_file", "file_write", "file_delete", "file_str_replace"):
        path = args.get("path", args.get("filename", "unknown"))
        operation = "modify" if "delete" not in tool_name else "delete"
        return (
            f"File {operation.title()}",
            f"The agent wants to {operation} the file:\n\n**{path}**",
        )
    return (
        f"Tool Approval: {tool_name}",
        f"The agent wants to use **{tool_name}**.\n\nRisk level: **{risk_level.value}**",
    )
