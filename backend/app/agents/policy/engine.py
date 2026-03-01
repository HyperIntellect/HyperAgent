"""Policy engine v1 for tool and skill execution decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from app.agents.policy.contracts import CapabilityContract, SideEffectLevel


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PolicyInput:
    tool_name: str
    tool_args: dict
    auto_approve_tools: list[str] | None = None
    hitl_enabled: bool = True
    risk_threshold: Literal["high", "medium", "all"] = "high"
    contract: CapabilityContract | None = None
    is_skill_invocation: bool = False


@dataclass
class PolicyResult:
    decision: PolicyDecision
    reason_code: str
    risk_level: RiskLevel


class PolicyEngine:
    """Simple rule-based policy engine for runtime governance."""

    def assess_risk(self, tool_name: str, contract: CapabilityContract | None = None) -> RiskLevel:
        if tool_name.startswith("browser_"):
            return RiskLevel.HIGH
        if tool_name in {
            "execute_code",
            "sandbox_file",
            "file_write",
            "file_str_replace",
            "shell_exec",
            "shell_kill",
            "deploy_expose_port",
            "deploy_to_production",
            "execute_sql",
        }:
            return RiskLevel.HIGH
        if tool_name in {
            "file_read",
            "http_request",
            "send_notification",
            "deploy_get_url",
            "shell_view",
            "shell_wait",
            "invoke_skill",
        }:
            return RiskLevel.MEDIUM

        if contract and contract.side_effect_level in {SideEffectLevel.HIGH, SideEffectLevel.MEDIUM}:
            return RiskLevel.HIGH if contract.side_effect_level == SideEffectLevel.HIGH else RiskLevel.MEDIUM
        return RiskLevel.LOW

    def decide(self, policy_input: PolicyInput) -> PolicyResult:
        tool_name = policy_input.tool_name
        approved = set(policy_input.auto_approve_tools or [])
        risk = self.assess_risk(tool_name, policy_input.contract)

        if not policy_input.hitl_enabled:
            return PolicyResult(PolicyDecision.ALLOW, "hitl_disabled", risk)

        if tool_name in approved:
            return PolicyResult(PolicyDecision.ALLOW, "auto_approved_tool", risk)

        if (
            policy_input.is_skill_invocation
            and policy_input.tool_args.get("skill_id")
            and f"invoke_skill:{policy_input.tool_args.get('skill_id')}" in approved
        ):
            return PolicyResult(PolicyDecision.ALLOW, "auto_approved_skill", risk)

        if policy_input.risk_threshold == "all":
            return PolicyResult(PolicyDecision.REQUIRE_APPROVAL, "risk_threshold_all", risk)

        if policy_input.risk_threshold == "medium" and risk in {RiskLevel.HIGH, RiskLevel.MEDIUM}:
            return PolicyResult(PolicyDecision.REQUIRE_APPROVAL, "risk_threshold_medium", risk)

        if policy_input.risk_threshold == "high" and risk == RiskLevel.HIGH:
            return PolicyResult(PolicyDecision.REQUIRE_APPROVAL, "risk_threshold_high", risk)

        return PolicyResult(PolicyDecision.ALLOW, "allowed_by_policy", risk)


_policy_engine = PolicyEngine()


def get_policy_engine() -> PolicyEngine:
    return _policy_engine

