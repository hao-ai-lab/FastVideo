"""Loop plane — driven loops + policies (design_v3 §5)."""
from __future__ import annotations

from v2._enums import LoopKind, WorkUnitKind
from v2.loop.contracts import (
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
from v2.loop.driver import LoopRunner
from v2.loop.sampler import add_noise, build_flow_sigmas, flow_match_euler_step, x0_from_velocity

__all__ = [
    "Loop", "LoopContext", "LoopState", "LoopResult", "WorkPlan", "Done", "StepResult",
    "StepContext", "ShapeSignature", "ResourceRequest", "CachePlan", "CacheOp", "PlacementHint",
    "LoopRunner", "LoopKind", "WorkUnitKind",
    "build_flow_sigmas", "flow_match_euler_step", "x0_from_velocity", "add_noise",
]
