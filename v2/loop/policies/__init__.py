"""Policies — the default step decomposition (design_v3 §5.1, §6.2.3)."""
from __future__ import annotations

from v2.loop.policies.base import (
    BoundaryTimestepRouting,
    ConditioningInjector,
    ExpertRouting,
    FlowShiftPolicy,
    NoRouting,
    PassthroughConditioning,
    PrecisionPolicy,
)
from v2.loop.policies.cfg import (
    AdaptiveGateCFG,
    BatchedCFG,
    CFGPolicy,
    ClassicCFG,
    EmbeddedGuidance,
    PerModalityCFG,
)

__all__ = [
    "CFGPolicy", "ClassicCFG", "BatchedCFG", "EmbeddedGuidance", "AdaptiveGateCFG", "PerModalityCFG",
    "FlowShiftPolicy", "PrecisionPolicy", "ExpertRouting", "NoRouting", "BoundaryTimestepRouting",
    "ConditioningInjector", "PassthroughConditioning",
]
