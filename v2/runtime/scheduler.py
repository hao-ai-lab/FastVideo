"""The scheduler — a refundable memory (OOM) admission guard.

``AdmissionController`` is the only layer: do not admit a step unless its peak-activation memory can
be reserved (resident memory is reserved once for the whole loop). Reservations are *refundable* (a
concurrency gate, not a lifetime cap); an infeasible reservation (need > pool capacity) fails fast
with ``AdmissionInfeasible`` instead of spinning. No compute/time budget and no cross-request
batching — serving is pooled run-to-completion (concurrency bounded by the pool).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from v2.core.loop.contracts import WorkPlan
from v2.runtime.memory.allocator import MemoryManager, Reservation


class AdmissionInfeasible(RuntimeError):
    """A reservation can never be satisfied (need > pool capacity) — fail fast, don't spin."""


@dataclass
class StepTicket:
    """Returned by ``admit_step``; carries the memory reservation ``release`` must refund."""
    memory_res: Reservation | None


@dataclass
class SchedulerMetrics:
    admitted: int = 0
    deferred: int = 0
    stepped_units: int = 0
    by_kind: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class AdmissionController:
    """Reservation before admission — a refundable memory (OOM) guard. No compute/time budget
    (pooled run-to-completion prices nothing; concurrency is bounded by the RolePool)."""

    def __init__(self, memory: MemoryManager | None = None):
        self.memory = memory or MemoryManager()
        self.metrics = SchedulerMetrics()

    def feasible_resident(self, nbytes: int) -> bool:
        return nbytes <= self.memory.total_bytes

    def feasible_step(self, plan: WorkPlan) -> bool:
        """Could this step EVER be admitted on an empty pool? If not, the caller fails fast."""
        return plan.resources.peak_activation_bytes <= self.memory.total_bytes

    def reserve_resident(self, tag: str, nbytes: int) -> Reservation | None:
        if nbytes <= 0:
            return None
        if not self.memory.can_reserve(tag, nbytes):
            return None
        return self.memory.reserve(tag, nbytes)

    def admit_step(self, plan: WorkPlan) -> StepTicket | None:
        """Reserve a step's peak activation memory. None ⇒ deferred (no free memory yet)."""
        peak = plan.resources.peak_activation_bytes
        res: Reservation | None = None
        if peak > 0:
            if not self.memory.can_reserve(plan.loop_id, peak):
                self.metrics.deferred += 1
                return None
            res = self.memory.reserve(plan.loop_id, peak)
        self.metrics.admitted += 1
        self.metrics.by_kind[plan.kind.value] += 1
        return StepTicket(res)

    def release(self, ticket: StepTicket | None) -> None:
        if ticket is None:
            return
        if ticket.memory_res is not None:
            self.memory.release(ticket.memory_res)
