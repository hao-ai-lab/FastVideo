"""Execution plane — engine + scheduler (design_v3 §6)."""
from __future__ import annotations

from .async_engine import AsyncEngine, RequestState
from .context import RuntimeLoopContext
from .disaggregated import DisaggregatedRunner
from .engine import Engine, ProgramRunner
from .pools import DeployConfig, PoolSet, RolePool, RolePoolSpec, wan_t2v_disaggregated
from .scheduler import (
    AdmissionController,
    AdmissionInfeasible,
    BatchScheduler,
    SchedulerMetrics,
    StepTicket,
    WorkUnit,
)

__all__ = ["Engine", "ProgramRunner", "RuntimeLoopContext", "AdmissionController",
           "AdmissionInfeasible", "BatchScheduler", "SchedulerMetrics", "StepTicket", "WorkUnit",
           "AsyncEngine", "RequestState", "DisaggregatedRunner",
           "DeployConfig", "PoolSet", "RolePool", "RolePoolSpec", "wan_t2v_disaggregated"]
