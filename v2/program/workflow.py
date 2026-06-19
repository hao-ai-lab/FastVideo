"""Workflow — multi-MODEL composition above the engine (``ProgramKind.WORKFLOW``).

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
from typing import Any
from collections.abc import Callable


@dataclass
class WorkflowStage:
    """One model invocation. ``make_request(state)`` builds this stage's Request from the accumulated
    ``state`` (the initial inputs plus every prior stage's artifacts, keyed ``"<stage_label>:<name>"``
    and also under ``"prev"`` for the immediately-preceding stage's artifact dict)."""
    model_id: str
    make_request: Callable[[dict[str, Any]], Any]
    label: str = ""


def _engine_serves(engine: Any, model_id: str) -> bool:
    if hasattr(engine, "serves"):
        return engine.serves(model_id)  # AsyncEngine
    return model_id in getattr(engine, "_registry", {})  # Engine


@dataclass
class Workflow:
    """A named, registrable cross-model pipeline. ``workflow_id`` is its servable name — it lives in
    the SAME namespace as a card's ``model_id`` (so the engine, server, and fleet route to it the same
    way), and by convention is dotted/namespaced to avoid colliding with a card id (e.g.
    ``"image_video.t2i_i2v"``). ``requires`` is the set of cards it composes — declared, validated,
    never inferred."""
    workflow_id: str
    stages: list[WorkflowStage] = field(default_factory=list)

    @property
    def requires(self) -> list[str]:
        """The model_ids this workflow composes (deduped, in first-use order). The engine must serve
        all of them before the workflow can run."""
        seen: set[str] = set()
        out: list[str] = []
        for s in self.stages:
            if s.model_id not in seen:
                seen.add(s.model_id)
                out.append(s.model_id)
        return out

    def validate(self, engine: Any) -> Workflow:
        """Fail-fast: every required card must already be registered on ``engine`` (no silent skips)."""
        if not self.stages:
            raise ValueError(f"workflow {self.workflow_id!r} has no stages")
        missing = [m for m in self.requires if not _engine_serves(engine, m)]
        if missing:
            raise ValueError(f"workflow {self.workflow_id!r} requires unregistered models {missing} "
                             f"(register those cards before the workflow)")
        return self

    def run(self, engine: Any, **initial: Any) -> Any:
        """Execute the stages in order on ``engine``; return the final stage's Output.

        Each stage's artifacts are merged into ``state`` so downstream stages can read them
        (e.g. the I2V stage reads the T2I stage's ``image`` artifact)."""
        if not self.stages:
            raise ValueError(f"workflow {self.workflow_id!r} has no stages")
        state: dict[str, Any] = dict(initial)
        out = None
        for stage in self.stages:
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


def _payload(artifact: Any) -> Any:
    for attr in ("frames", "samples", "latent", "tensor", "text"):
        v = getattr(artifact, attr, None)
        if v is not None:
            return v
    return None


class ParallelWorkflow(Workflow):
    """Fan-out: run every stage on the SAME initial input (conceptually in parallel — the engine can
    interleave their steps), then merge their artifacts into one Output, namespaced by stage label.
    The non-linear shape a linear chain can't express: one prompt → N models → a merged result (e.g.
    variants, or video+audio+upscale in parallel)."""

    def run(self, engine: Any, **initial: Any) -> Any:
        if not self.stages:
            raise ValueError(f"workflow {self.workflow_id!r} has no stages")
        from v2.request.artifacts import Output
        merged: dict[str, Any] = {}
        metrics: dict[str, float] = {}
        rid = self.workflow_id
        for stage in self.stages:
            label = stage.label or stage.model_id
            req = stage.make_request(dict(initial))  # each branch sees the ORIGINAL input
            if req.model_id != stage.model_id:
                raise ValueError(f"{self.workflow_id!r} stage {label!r}: model mismatch")
            out = engine.run(req)
            rid = out.request_id
            for name, art in out.artifacts.items():
                merged[f"{label}:{name}"] = art  # namespaced — branches don't collide
            for k, v in out.metrics.items():
                metrics[f"{label}:{k}"] = v
        return Output(request_id=rid, artifacts=merged, metrics=metrics)


class BestOfNWorkflow(Workflow):
    """Inference-time scaling / rejection sampling: generate N candidates (varying seed), score each with
    a reward scorer, return the best. A feedback loop across models — generator + reward (e.g. the served
    REWARD_BATCH card) — the other non-linear shape."""

    def __init__(self, workflow_id, generator_stage, *, scorer, n: int = 4, score_key: str = "latents"):
        super().__init__(workflow_id, [generator_stage])
        self.scorer = scorer  # any .score(media, prompts) -> {"avg": ...}
        self.n = int(n)
        self.score_key = score_key

    def run(self, engine: Any, **initial: Any) -> Any:
        import numpy as np
        stage = self.stages[0]
        base_seed = int(initial.get("seed", 0))
        cands = []
        for i in range(self.n):
            req = stage.make_request({**initial, "seed": base_seed * 100 + i})
            if req.model_id != stage.model_id:
                raise ValueError(f"{self.workflow_id!r}: model mismatch")
            cands.append(engine.run(req))
        media = [_payload(c.artifacts[self.score_key]) for c in cands]
        scores = np.asarray(self.scorer.score(media, [initial.get("prompt", "")] * len(cands))["avg"])
        best = int(np.argmax(scores))
        out = cands[best]
        out.metrics["best_of_n"] = float(self.n)
        out.metrics["best_index"] = float(best)
        out.metrics["best_score"] = float(scores[best])
        return out


class WorkflowRegistry:
    """Declarative ``workflow_id → builder`` catalog — the cross-model analog of the card builders in
    ``models/__init__.py`` (cf. vllm-omni's ``pipeline_registry``). Adding a custom pipeline is one
    ``register`` call; the builder is a zero/kw-arg factory returning a ``Workflow``."""

    def __init__(self) -> None:
        self._builders: dict[str, Any] = {}

    def register(self, workflow_id: str, builder: Any) -> Any:
        if workflow_id in self._builders:
            raise ValueError(f"workflow {workflow_id!r} already registered")
        self._builders[workflow_id] = builder
        return builder

    def build(self, workflow_id: str, **kw: Any) -> Workflow:
        if workflow_id not in self._builders:
            raise KeyError(f"no workflow {workflow_id!r} (have {list(self._builders)})")
        return self._builders[workflow_id](**kw)

    def names(self) -> list[str]:
        return list(self._builders)

    def __contains__(self, workflow_id: str) -> bool:
        return workflow_id in self._builders
