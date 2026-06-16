"""The scheduler, in layers (design_v3 §6.3) — each testable on a fake pool, no GPU.

Mini-fastvideo implements the load-bearing layers:
  * ``AdmissionController`` — the reservation gate (§6.2): do not admit a WorkUnit unless its
    compute budget AND memory (resident + worst-case peak) can be reserved. Two requests that
    fit individually but jointly OOM are rejected at admission, not at step 37.
  * ``BatchScheduler`` — groups compatible WorkPlans by ``shape_sig.batch_key`` (§6.3): image
    diffusion and AR decode batch across requests; jumbo video stays batch-of-1.
  * cost currency — predicted GPU-time (§6.1), accumulated for fairness/metrics.

PlacementScheduler/TransferScheduler are single-node trivial here (one pool, in-proc edges).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..loop.contracts import WorkPlan
from ..memory.allocator import MemoryManager, Reservation


@dataclass
class WorkUnit:
    """The smallest schedulable action (design_v3 §6.1)."""
    request_id: str
    plan: WorkPlan
    priority: int = 0


@dataclass
class SchedulerMetrics:
    admitted: int = 0
    deferred: int = 0
    batches: int = 0
    batched_units: int = 0
    gpu_seconds: float = 0.0
    by_kind: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class AdmissionController:
    """Reservation before admission (design_v3 §6.2). Compute budget + memory both gate."""

    def __init__(self, memory: MemoryManager | None = None, compute_budget_seconds: float = float("inf")):
        self.memory = memory or MemoryManager()
        self.compute_budget = compute_budget_seconds
        self.compute_spent = 0.0
        self.metrics = SchedulerMetrics()

    def reserve_resident(self, tag: str, nbytes: int) -> Reservation | None:
        if nbytes <= 0:
            return None
        if not self.memory.can_reserve(tag, nbytes):
            return None
        return self.memory.reserve(tag, nbytes)

    def admit_step(self, plan: WorkPlan) -> Reservation | None | bool:
        """Reserve a step's peak activation + check the compute budget.

        Returns a Reservation (peak reserved), True (no peak to reserve but admitted),
        or None (deferred — budget/memory unavailable)."""
        c = plan.resources.compute_seconds
        if self.compute_spent + c > self.compute_budget:
            self.metrics.deferred += 1
            return None
        peak = plan.resources.peak_activation_bytes
        res: Reservation | None = None
        if peak > 0:
            if not self.memory.can_reserve(plan.loop_id, peak):
                self.metrics.deferred += 1
                return None
            res = self.memory.reserve(plan.loop_id, peak)
        self.compute_spent += c
        self.metrics.admitted += 1
        self.metrics.gpu_seconds += c
        self.metrics.by_kind[plan.kind.value] += 1
        return res if res is not None else True

    def release(self, res: Reservation | None | bool) -> None:
        if isinstance(res, Reservation):
            self.memory.release(res)


class BatchScheduler:
    """Group compatible WorkPlans across requests (design_v3 §6.3 layer 3)."""

    @staticmethod
    def group(units: list[WorkUnit]) -> list[list[WorkUnit]]:
        groups: dict[Any, list[WorkUnit]] = defaultdict(list)
        for u in units:
            groups[u.plan.shape_sig.batch_key].append(u)
        return list(groups.values())
