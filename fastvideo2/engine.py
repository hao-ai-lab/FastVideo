"""The engine — drives a pipeline over one resident instance, one request at a
time, with the identity chain attached.

Identity chain: every unit of work is named ``request/stage`` for one-shot
stages and ``request/stage/loop.step`` for loop steps. The same name goes to
(a) the returned trace (typed timings, machine-readable) and (b) NVTX ranges
when CUDA is present — so Nsight correlates kernels to model-level identity
with no extra instrumentation.

Deliberately absent (this is the one-shot MVP): queueing, admission, batching,
sessions, cancellation. Sessions with forkable state are the next consumer of
the loop contract, not a reason to grow this file now.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, replace
from typing import Any

from fastvideo2.card import ModelCard
from fastvideo2.loading import load_component, resolve_weights
from fastvideo2.loop import LoopRunner, build_loop
from fastvideo2.pipeline import ComponentStage, LoopStage, Pipeline, run_component_stage


@dataclass(frozen=True)
class Request:
    """One generation request. ``None`` fields resolve from the card's
    sampling defaults via :meth:`resolve`."""
    prompt: str
    request_id: str = "req0"
    negative_prompt: str | None = None
    seed: int = 0
    num_steps: int | None = None
    guidance_scale: float | None = None
    height: int | None = None
    width: int | None = None
    num_frames: int | None = None
    shift: float | None = None
    capture_trajectory: bool = False

    def resolve(self, card: ModelCard) -> "Request":
        d = card.sampling_defaults
        fill = {
            "negative_prompt": d.negative_prompt,
            "num_steps": d.num_steps,
            "guidance_scale": d.guidance_scale,
            "height": d.height,
            "width": d.width,
            "num_frames": d.num_frames,
            "shift": d.shift,
        }
        patch = {k: v for k, v in fill.items() if getattr(self, k) is None}
        return replace(self, **patch)


@dataclass
class Output:
    request_id: str
    outputs: dict[str, Any]
    trace: list[dict] = field(default_factory=list)   # [{label, seconds, ...meta}]

    @property
    def seconds(self) -> float:
        return sum(t["seconds"] for t in self.trace)


class Instance:
    """A resident, loaded card: components materialize lazily and are shared
    by reference; loops are built from the card's declared specs."""

    def __init__(self, card: ModelCard, root: str | None = None, device: str = "cpu"):
        self.card = card
        self.device = device
        self.root = resolve_weights(card, root)
        self._components: dict[str, Any] = {}
        self._source_roots: dict[str, str] = {}
        self._loops: dict[str, Any] = {}

    def component(self, component_id: str) -> Any:
        if component_id not in self._components:
            spec = self.card.components.get(component_id)
            if spec is None:
                raise KeyError(f"component {component_id!r} not declared on card {self.card.model_id!r}")
            root = self._source_root(spec.source) if spec.source else self.root
            self._components[component_id] = load_component(spec, root, self.device)
        return self._components[component_id]

    def _source_root(self, source: str) -> str:
        """Resolve a per-component weights source (e.g. the official-layout
        transformer repo) through the same snapshot cache as card weights."""
        if source not in self._source_roots:
            from huggingface_hub import snapshot_download
            self._source_roots[source] = snapshot_download(source)
        return self._source_roots[source]

    def loop(self, loop_id: str) -> Any:
        if loop_id not in self._loops:
            spec = self.card.loops.get(loop_id)
            if spec is None:
                raise KeyError(f"loop {loop_id!r} not declared on card {self.card.model_id!r}")
            self._loops[loop_id] = build_loop(spec)
        return self._loops[loop_id]


def load(card: ModelCard, root: str | None = None, device: str | None = None) -> Instance:
    """The public entrypoint: card + weights root -> resident instance."""
    card.validate()
    if device is None:
        device = _detect_device()
    return Instance(card, root=root, device=device)


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@contextlib.contextmanager
def _nvtx(name: str):
    """NVTX range when CUDA is live; free otherwise."""
    pushed = False
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
            pushed = True
    except ImportError:
        pass
    try:
        yield
    finally:
        if pushed:
            import torch
            torch.cuda.nvtx.range_pop()


def run(instance: Instance, pipeline: Pipeline, request: Request) -> Output:
    """Run one request through a pipeline to completion."""
    import time
    req = request.resolve(instance.card)
    trace: list[dict] = []
    slots: dict[str, Any] = {}
    for name in pipeline.inputs:  # request-provided slots, by attribute name
        slots[name] = getattr(req, name)

    for stage in pipeline.stages:
        chain = f"{req.request_id}/{stage.stage_id}"
        if isinstance(stage, ComponentStage):
            with _nvtx(chain):
                t0 = time.perf_counter()
                run_component_stage(stage, instance, slots, req)
                trace.append({"label": chain, "seconds": time.perf_counter() - t0})
        elif isinstance(stage, LoopStage):
            loop = instance.loop(stage.loop_id)

            def observe(label: str, seconds: float, meta: dict, _chain: str = chain) -> None:
                trace.append({"label": f"{_chain}/{label}", "seconds": seconds, **meta})

            inputs = {k: slots[k] for k in stage.reads}
            with _nvtx(chain):
                runner = LoopRunner(loop, req, instance, inputs, observe=observe)
                slots[stage.writes[0]] = runner.run()
        else:
            raise TypeError(f"unknown stage kind {type(stage).__name__}")

    outputs = {name: slots[slot] for name, slot in pipeline.outputs.items()}
    return Output(request_id=req.request_id, outputs=outputs, trace=trace)
