"""Loop plane — driven loops + policies (design_v3 §5)."""
from __future__ import annotations

from .._enums import LoopKind, WorkUnitKind
from .contracts import (
    CacheOp,
    CachePlan,
    Done,
    Loop,
    LoopContext,
    LoopResult,
    LoopState,
    PlacementHint,
    ResourceRequest,
    ShapeSignature,
    StepContext,
    StepResult,
    WorkPlan,
)
from .driver import LoopRunner
from .sampler import add_noise, build_flow_sigmas, flow_match_euler_step, x0_from_velocity

__all__ = [
    "Loop", "LoopContext", "LoopState", "LoopResult", "WorkPlan", "Done", "StepResult",
    "StepContext", "ShapeSignature", "ResourceRequest", "CachePlan", "CacheOp", "PlacementHint",
    "LoopRunner", "LoopKind", "WorkUnitKind",
    "build_flow_sigmas", "flow_match_euler_step", "x0_from_velocity", "add_noise",
]
