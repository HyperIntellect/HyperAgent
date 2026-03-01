"""Capability contracts for tools and skills."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SideEffectLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DataSensitivity(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"


class NetworkScope(str, Enum):
    NONE = "none"
    SANDBOX_ONLY = "sandbox_only"
    EXTERNAL = "external"


@dataclass(frozen=True)
class CapabilityContract:
    side_effect_level: SideEffectLevel
    data_sensitivity: DataSensitivity
    network_scope: NetworkScope
    idempotency_hint: bool

