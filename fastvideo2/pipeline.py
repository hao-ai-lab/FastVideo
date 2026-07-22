"""Pipelines — an ordered stage list over named slots, with enforced edges.

A pipeline composes one model instance's components and loops into a task:

    text_encode -> denoise (loop) -> vae_decode

Deliberately not a DAG, not an IR, not a workflow engine. Two kinds of stage:

* ``ComponentStage`` — a one-shot transform (encode, decode). Its function
  receives only the slots it declared in ``reads`` and must return exactly the
  slots it declared in ``writes``.
* ``LoopStage`` — binds one of the card's declared loops; the engine drives it
  step-by-step via ``LoopRunner``.

``reads``/``writes`` are enforced, not decorative: the runner hands each stage
a view restricted to its declared reads and rejects undeclared or missing
writes. Declared edges that lie fail loudly at run time, and ``validate()``
rejects reads nothing produces at declaration time.

This module is stdlib-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


class PipelineError(ValueError):
    pass


@dataclass(frozen=True)
class ComponentStage:
    """One-shot transform: ``fn(instance, inputs, request) -> {write: value}``."""
    stage_id: str
    fn: Callable[[Any, Mapping[str, Any], Any], dict]
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoopStage:
    """Binds a card loop by id. The loop's ``finalize()`` dict is written to
    the single declared write slot."""
    stage_id: str
    loop_id: str
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Pipeline:
    pipeline_id: str
    inputs: tuple[str, ...]                 # request-provided slots (seeded by the engine)
    stages: tuple[Any, ...]
    outputs: dict[str, str] = field(default_factory=dict)  # output name -> slot

    def validate(self) -> "Pipeline":
        errs: list[str] = []
        produced: set[str] = set(self.inputs)
        seen_ids: set[str] = set()
        for st in self.stages:
            if st.stage_id in seen_ids:
                errs.append(f"duplicate stage id {st.stage_id!r}")
            seen_ids.add(st.stage_id)
            for r in st.reads:
                if r not in produced:
                    errs.append(f"stage {st.stage_id!r} reads {r!r} which nothing before it "
                                f"produces (have {sorted(produced)})")
            for w in st.writes:
                if w in produced:
                    errs.append(f"stage {st.stage_id!r} rewrites slot {w!r}")
                produced.add(w)
            if isinstance(st, LoopStage) and len(st.writes) != 1:
                errs.append(f"loop stage {st.stage_id!r} must declare exactly one write slot")
        for name, slot in self.outputs.items():
            if slot not in produced:
                errs.append(f"output {name!r} reads slot {slot!r} that no stage writes")
        if errs:
            raise PipelineError(f"pipeline {self.pipeline_id!r} failed validation:\n  - "
                                + "\n  - ".join(errs))
        return self


def run_component_stage(stage: ComponentStage, instance: Any, slots: dict, request: Any) -> None:
    """Execute one component stage with enforced edges (the runner's teeth)."""
    view = {k: slots[k] for k in stage.reads}
    out = stage.fn(instance, view, request)
    if not isinstance(out, dict) or set(out) != set(stage.writes):
        raise PipelineError(
            f"stage {stage.stage_id!r} returned slots {sorted(out) if isinstance(out, dict) else type(out)}, "
            f"declared writes {sorted(stage.writes)}")
    slots.update(out)
