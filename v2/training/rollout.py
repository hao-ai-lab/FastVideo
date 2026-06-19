"""Rollout = serve loop + capture.

``rollout_loop`` drives the *same* ``Loop`` the engine serves, in the ROLLOUT execution profile,
through a minimal training context — so there is one numerics surface, not a second sampler.

``training`` imports ``loop``/``runtime`` here; the engine never imports ``training`` (one-way rule).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

from v2._enums import ExecutionProfile
from v2.loop.contracts import StepResult, WorkPlan
from v2.loop.driver import LoopRunner
from v2.training.behavior import TrajectoryStatus


class _TrainingContext:
    """A minimal LoopContext for in-process (library-mode) rollout. Runs the kernel thunk directly;
    optional observers attach (e.g. ParityAligner for the consistency ladder)."""

    def __init__(self, instance: Any, slots: dict[str, Any], profile: ExecutionProfile, observers: list | None = None):
        self.instance = instance
        self.slots = slots
        self.profile = profile
        self._observers = observers or []
        self.state = None

    def bind_state(self, state) -> None:
        self.state = state

    def execute(self, plan: WorkPlan) -> StepResult:
        for o in self._observers:
            o.observe("step_start", plan=plan)
        out = plan.run(self.instance, None) if plan.run is not None else {}
        result = out if isinstance(out, StepResult) else StepResult(output=dict(out))
        for o in self._observers:
            o.observe("step_complete", plan=plan, result=result)
        return result

    def emit(self, chunk) -> None:  # rollout doesn't stream to a user
        pass

    def check_cancel(self) -> None:
        pass

    def observe(self, event: str, **kw) -> None:
        for o in self._observers:
            o.observe(event, **kw)


def rollout_loop(instance: Any,
                 loop_id: str,
                 request: Any,
                 *,
                 slots: dict[str, Any] | None = None,
                 profile: ExecutionProfile = ExecutionProfile.ROLLOUT,
                 observers: list | None = None):
    """Drive the SHARED loop to completion in the given profile; return its LoopResult."""
    ctx = _TrainingContext(instance, slots or {}, profile, observers)
    runner = LoopRunner(instance.loop(loop_id), ctx, request, instance)
    return runner.run()


@dataclass
class Trajectory:
    """One rollout sample: request + seed + per-step records + reward + advantage."""
    request_id: str
    prompt: str
    seed: int
    latents: Any = None
    behavior: Any = None
    rewards: dict[str, float] = field(default_factory=dict)
    reward: float = 0.0
    advantage: float = 0.0
    weights_version: str = "v0"
    status: TrajectoryStatus = TrajectoryStatus.COMPLETED


class TrajectoryBuffer:

    def __init__(self) -> None:
        self.items: list[Trajectory] = []

    def add(self, t: Trajectory) -> None:
        self.items.append(t)

    def group_by_prompt(self) -> dict[str, list[Trajectory]]:
        groups: dict[str, list[Trajectory]] = {}
        for t in self.items:
            groups.setdefault(t.prompt, []).append(t)
        return groups

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


@dataclass
class RolloutContext:
    """What a ``RolloutFn`` receives: an engine/instance client + a seed source."""
    instance: Any
    text_encode: Callable[[Any, Any, dict], None]
    base_seed: int = 0
