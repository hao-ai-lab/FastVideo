"""Workflow — multi-MODEL composition above the engine (design_v3 §13; ``ProgramKind.WORKFLOW``).

A ``Program`` composes the loops of ONE resident model instance (the omni/MoT default — every
``ModelLoopNode`` resolves on the program's single instance). A ``Workflow`` composes *across model
instances*: each stage is a full ``engine.run`` on a (possibly different) registered card, with typed
artifacts threaded stage→stage. This is the realistic "T2I model **followed by** I2V model" pipeline
(FLUX→Wan): two distinct cards, two distinct weight sets, chained.

The split mirrors the doc's two layers exactly:
  * **Program** — single-model, step-interleaved loop composition (the hot path; unchanged).
  * **Workflow** — multi-model orchestration; a thin layer that calls ``engine.run`` per stage and
    passes artifacts forward. No change to the engine's single-instance program runner — so cross-
    model chaining costs nothing on the per-step scheduling path. The engine already holds every
    instance in its registry; the Workflow just selects which card each stage runs on.

Why not put cross-model in ``Program``? Because a step-interleaved program shares one resident
instance, one cache scope, one parity gate. Crossing instances is a *boundary* (different weights,
different caches, an artifact hand-off) — exactly a Workflow edge, not a loop step. Keeping it out of
the program runner is what preserves the interleave-parity guarantee for the single-model case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class WorkflowStage:
    """One model invocation. ``make_request(state)`` builds this stage's Request from the accumulated
    ``state`` (the initial inputs plus every prior stage's artifacts, keyed ``"<stage_label>:<name>"``
    and also under ``"prev"`` for the immediately-preceding stage's artifact dict)."""
    model_id: str
    make_request: Callable[[dict[str, Any]], Any]
    label: str = ""


@dataclass
class Workflow:
    workflow_id: str
    stages: list[WorkflowStage] = field(default_factory=list)

    def run(self, engine: Any, **initial: Any) -> Any:
        """Execute the stages in order on ``engine``; return the final stage's Output.

        Each stage's artifacts are merged into ``state`` so downstream stages can read them
        (e.g. the I2V stage reads the T2I stage's ``image`` artifact)."""
        if not self.stages:
            raise ValueError(f"workflow {self.workflow_id!r} has no stages")
        state: dict[str, Any] = dict(initial)
        out = None
        for i, stage in enumerate(self.stages):
            label = stage.label or stage.model_id
            req = stage.make_request(state)
            if req.model_id != stage.model_id:
                raise ValueError(f"workflow {self.workflow_id!r} stage {label!r}: request model "
                                 f"{req.model_id!r} != stage model {stage.model_id!r}")
            out = engine.run(req)
            state["prev"] = out.artifacts
            for name, art in out.artifacts.items():
                state[f"{label}:{name}"] = art
        return out
