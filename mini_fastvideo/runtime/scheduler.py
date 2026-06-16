"""The scheduler, in layers (design_v3 §6.3) — each testable on a fake pool, no GPU.

Mini-fastvideo implements the load-bearing layers:
  * ``AdmissionController`` — the reservation gate (§6.2): do not admit a WorkUnit unless its
    compute budget AND memory (resident + worst-case peak) can be reserved. Reservations are
    *refundable* (compute is a concurrency gate, not a lifetime cap), and infeasible reservations
    (need > pool capacity) fail fast with ``AdmissionInfeasible`` instead of spinning.
  * ``BatchScheduler`` — groups compatible WorkPlans by ``shape_sig.batch_key`` (§6.3). NOTE: in this
    single-process mini, grouping drives *accounting* (how many units would co-batch); kernels still
    execute per-plan. Real cross-request batched execution is a GPU-path concern (documented, not faked).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..loop.contracts import WorkPlan
from ..memory.allocator import MemoryManager, Reservation


class AdmissionInfeasible(RuntimeError):
    """A reservation can never be satisfied (need > pool capacity) — fail fast, don't spin."""


@dataclass
class WorkUnit:
    """The smallest schedulable action (design_v3 §6.1)."""
    request_id: str
    plan: WorkPlan
    priority: int = 0


@dataclass
class StepTicket:
    """Returned by ``admit_step``; carries what ``release`` must refund (memory + compute)."""
    memory_res: Reservation | None
    compute_seconds: float


@dataclass
class SchedulerMetrics:
    admitted: int = 0
    deferred: int = 0
    batches: int = 0
    stepped_units: int = 0
    gpu_seconds: float = 0.0
    by_kind: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class AdmissionController:
    """Reservation before admission (design_v3 §6.2). Compute budget + memory both gate; both refund."""

    def __init__(self, memory: MemoryManager | None = None, compute_budget_seconds: float = float("inf")):
        self.memory = memory or MemoryManager()
        self.compute_budget = compute_budget_seconds
        self.compute_spent = 0.0
        self.metrics = SchedulerMetrics()

    def feasible_resident(self, nbytes: int) -> bool:
        return nbytes <= self.memory.total_bytes

    def feasible_step(self, plan: WorkPlan) -> bool:
        """Could this step EVER be admitted on an empty pool? If not, the caller fails fast."""
        return (plan.resources.peak_activation_bytes <= self.memory.total_bytes
                and plan.resources.compute_seconds <= self.compute_budget)

    def reserve_resident(self, tag: str, nbytes: int) -> Reservation | None:
        if nbytes <= 0:
            return None
        if not self.memory.can_reserve(tag, nbytes):
            return None
        return self.memory.reserve(tag, nbytes)

    def admit_step(self, plan: WorkPlan) -> StepTicket | None:
        """Reserve a step's peak activation + check the compute budget. None ⇒ deferred."""
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
        return StepTicket(res, c)

    def release(self, ticket: StepTicket | None) -> None:
        if ticket is None:
            return
        if ticket.memory_res is not None:
            self.memory.release(ticket.memory_res)
        self.compute_spent = max(0.0, self.compute_spent - ticket.compute_seconds)   # refund


class BatchScheduler:
    """Group compatible WorkPlans across requests (design_v3 §6.3 layer 3)."""

    @staticmethod
    def group(units: list[WorkUnit]) -> list[list[WorkUnit]]:
        groups: dict[Any, list[WorkUnit]] = defaultdict(list)
        for u in units:
            groups[u.plan.shape_sig.batch_key].append(u)
        return list(groups.values())
