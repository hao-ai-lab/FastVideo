"""The engine — drives Programs over resident ModelInstances (design_v3 §6).

Two execution modes share one ProgramRunner:
  * ``run_serial``      — each request's program runs to completion before the next.
  * ``run_interleaved`` — round-robin one *unit* per request per tick (a unit = one loop step,
    or one one-shot node), so the denoise steps of concurrent requests interleave. This is the
    mode the §9.3 interleave parity gate exercises: serial and interleaved must be bit-identical.

The engine owns admission (reservation before stepping), the observer bus, and the interceptor
chain (deploy scope). It never imports ``training`` (design_v3 §10 dependency rule).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._enums import ExecutionProfile
from ..extend.base import InterceptorChain, ObserverBus
from ..loop.driver import LoopRunner
from ..parity.aligner import ParityAligner  # noqa: F401  (engine can attach it as an observer)
from ..program.specs import ComponentNode, ModelLoopNode, Program
from ..request.artifacts import (
    AudioArtifact,
    LatentArtifact,
    Output,
    TensorArtifact,
    TextArtifact,
    VideoArtifact,
)
from ..request.cancel import CancelKind, CancelScope
from ..request.streams import Stream
from .context import RuntimeLoopContext
from .scheduler import (
    AdmissionController,
    AdmissionInfeasible,
    BatchScheduler,
    SchedulerMetrics,
    WorkUnit,
)


def _to_artifact(name: str, value: Any, producer: str = "") -> Any:
    if isinstance(value, dict):
        value = value.get(name, value.get("latents", next(iter(value.values()), None)))
    if name == "video":
        return VideoArtifact(name=name, producer=producer, frames=value)
    if name == "audio":
        return AudioArtifact(name=name, producer=producer, samples=value)
    if name == "text":
        return TextArtifact(name=name, producer=producer, text=value if isinstance(value, str) else "")
    if name == "latents":
        return LatentArtifact(name=name, producer=producer, latent=value)
    return TensorArtifact(name=name, producer=producer, tensor=value)


class ProgramRunner:
    """Drives one Program for one request. Advances one *unit* per ``tick`` for interleaving."""

    def __init__(self, engine: "Engine", program: Program, instance: Any, request: Any):
        self.engine = engine
        self.program = program
        self.instance = instance
        self.request = request
        self.slots: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.cancel_scope = CancelScope(kind=CancelKind.REQUEST, target_id=request.request_id)
        self.stream = Stream(request.request_id)
        profile = ExecutionProfile.ROLLOUT if getattr(request.outputs, "capture", None) and \
            request.outputs.capture.value == "behavior" else engine.profile
        self.ctx = RuntimeLoopContext(
            instance, observers=engine.observers, interceptors=engine.interceptors,
            slots=self.slots, stream=self.stream, cancel_scope=self.cancel_scope,
            profile=profile, metrics=self.metrics, request_id=request.request_id)
        self.node_idx = 0
        self.loop_runner: LoopRunner | None = None
        self._resident_res: Any = None
        self.done = False
        self.error: str | None = None

    def _merge_metrics(self, m: dict[str, float]) -> None:
        for k, v in m.items():
            self.metrics[k] = self.metrics.get(k, 0.0) + v

    def tick(self) -> bool:
        if self.done:
            return True
        self.cancel_scope.check()          # common-path cancellation (offline + all-component programs)
        nodes = self.program.nodes
        while True:
            if self.node_idx >= len(nodes):
                self.done = True
                return True
            node = nodes[self.node_idx]
            if not node.when(self.request):
                self.node_idx += 1
                continue

            if isinstance(node, ComponentNode):
                if node.fn is not None:
                    node.fn(self.instance, self.slots, self.request, self.ctx)
                self.node_idx += 1
                continue                      # one-shots run within this tick

            if isinstance(node, ModelLoopNode):
                if self.loop_runner is None:
                    loop = self.instance.loop(node.loop_id)
                    self.loop_runner = LoopRunner(loop, self.ctx, self.request, self.instance)
                    self._resident_res = None

                # reserve resident memory for the whole loop (concurrent-pressure admission)
                if self._resident_res is None:
                    plan0 = self.loop_runner.peek()
                    if plan0 is None:
                        self._finish_loop(node)
                        return False
                    need = plan0.resources.resident_bytes
                    if need > 0:
                        if not self.engine.admission.feasible_resident(need):
                            raise AdmissionInfeasible(
                                f"{node.loop_id}: resident {need}B exceeds pool capacity "
                                f"{self.engine.admission.memory.total_bytes}B")
                        res = self.engine.admission.reserve_resident(node.loop_id, need)
                        if res is None:
                            self.engine.metrics.deferred += 1
                            return False      # defer: cannot fit resident yet (jointly-OOM guard)
                        self._resident_res = res
                    else:
                        self._resident_res = True

                plan = self.loop_runner.peek()
                if plan is None:
                    self._finish_loop(node)
                    return False
                if not self.engine.admission.feasible_step(plan):
                    raise AdmissionInfeasible(
                        f"{node.loop_id}: step needs peak {plan.resources.peak_activation_bytes}B / "
                        f"{plan.resources.compute_seconds:.4g}s — exceeds pool/budget capacity")
                step_res = self.engine.admission.admit_step(plan)
                if step_res is None:
                    return False              # deferred this tick (will retry; refundable)
                self.loop_runner.step()
                self.engine.admission.release(step_res)
                self.engine.metrics.stepped_units += 1
                if self.loop_runner.done:
                    self._finish_loop(node)
                return False                  # yielded one loop step → return for interleaving
            # unknown node kind
            self.node_idx += 1

    def _finish_loop(self, node: ModelLoopNode) -> None:
        if self.loop_runner is not None and self.loop_runner.result is not None:
            self.slots[node.output_slot] = self.loop_runner.result.outputs
            self._merge_metrics(self.loop_runner.result.metrics)
        if self._resident_res not in (None, True):
            self.engine.admission.memory.release(self._resident_res)   # resident is a plain Reservation
        self._resident_res = None
        self.loop_runner = None
        self.node_idx += 1

    @property
    def _progress(self) -> tuple:
        """A token that changes iff the runner advanced (node, loop step, or resident reservation)."""
        sidx = self.loop_runner.state.step_idx if self.loop_runner is not None else -1
        return (self.node_idx, sidx, self._resident_res is not None)

    def run_to_completion(self) -> None:
        stuck = 0
        while not self.done:
            p0 = self._progress
            self.tick()
            if not self.done and self._progress == p0:
                stuck += 1
                if stuck > 3:                       # fail fast, not a 10M busy-spin
                    raise AdmissionInfeasible(
                        f"program {self.program.program_id} made no progress for several ticks "
                        "(unsatisfiable reservation or deadlock)")
            else:
                stuck = 0

    def output(self) -> Output:
        artifacts: dict[str, Any] = {}
        for name, slot in self.program.output_artifacts.items():
            artifacts[name] = _to_artifact(name, self.slots.get(slot), producer=slot)
        return Output(request_id=self.request.request_id, artifacts=artifacts,
                      metrics=dict(self.metrics), error=self.error)


@dataclass
class Engine:
    """A single-pool engine (design_v3 §6). Pools are single-node; fleet scale is multiple pools."""
    profile: ExecutionProfile = ExecutionProfile.SERVE
    observers: ObserverBus = field(default_factory=ObserverBus)
    interceptors: InterceptorChain = field(default_factory=InterceptorChain)
    admission: AdmissionController = field(default_factory=AdmissionController)
    metrics: SchedulerMetrics = field(default_factory=SchedulerMetrics)
    _registry: dict[str, tuple[Any, Program]] = field(default_factory=dict)
    _workflows: dict[str, Any] = field(default_factory=dict)   # workflow_id -> validated Workflow
    _wf_running: set = field(default_factory=set)              # workflow ids in-flight (nested cycle guard)

    def register(self, model_id: str, instance: Any, program: Program) -> None:
        self._registry[model_id] = (instance, program)

    def register_workflow(self, workflow: Any) -> Any:
        """Register a cross-model Workflow as a servable, addressable by ``workflow.workflow_id`` in
        the SAME namespace as model ids. Validates its required cards are already registered."""
        workflow.validate(self)
        if workflow.workflow_id in self._registry:
            raise ValueError(f"workflow id {workflow.workflow_id!r} collides with a registered model")
        self._workflows[workflow.workflow_id] = workflow
        return workflow

    def serves(self, name: str) -> bool:
        return name in self._registry or name in self._workflows

    def _resolve(self, request: Any) -> tuple[Any, Program]:
        if request.model_id not in self._registry:
            raise KeyError(f"no model registered for {request.model_id!r} (have {list(self._registry)})")
        return self._registry[request.model_id]

    def _make_runner(self, request: Any) -> ProgramRunner:
        instance, program = self._resolve(request)
        return ProgramRunner(self, program, instance, request)

    # --- offline single-request (the VideoGenerator path) -------------------- #
    def run(self, request: Any) -> Output:
        if request.model_id in self._workflows:                # a workflow servable → orchestrate stages
            wf = self._workflows[request.model_id]
            if request.model_id in self._wf_running:           # nested workflows: cycle guard (§9.13)
                raise ValueError(f"workflow cycle: {request.model_id!r} is already running "
                                 f"(a workflow cannot transitively invoke itself)")
            seed = request.diffusion.seed if request.diffusion.seed is not None else 0
            self._wf_running.add(request.model_id)
            try:
                return wf.run(self, prompt=request.prompt(), seed=seed, request=request)
            finally:
                self._wf_running.discard(request.model_id)
        r = self._make_runner(request)
        r.run_to_completion()
        return r.output()

    # --- serial vs interleaved (the interleave-parity gate drives both) ------- #
    def run_serial(self, requests: list[Any]) -> dict[str, Output]:
        out: dict[str, Output] = {}
        for req in requests:
            r = self._make_runner(req)
            r.run_to_completion()
            out[req.request_id] = r.output()
        return out

    def run_interleaved(self, requests: list[Any]) -> dict[str, Output]:
        runners = [self._make_runner(req) for req in requests]
        while not all(r.done for r in runners):
            # accounting: how many ready WorkPlans would co-batch this round (BatchScheduler, §6.3)
            units = [WorkUnit(r.request.request_id, r.loop_runner.peek())
                     for r in runners if not r.done and r.loop_runner is not None
                     and r.loop_runner.peek() is not None]
            if units:
                self.metrics.batches += len(BatchScheduler.group(units))
            progressed = False
            for r in runners:
                if not r.done:
                    p0 = r._progress
                    r.tick()
                    if r.done or r._progress != p0:
                        progressed = True
            if not progressed:                      # nobody could be admitted → real deadlock, fail fast
                raise AdmissionInfeasible(
                    "interleaved scheduling deadlocked: no request could be admitted this round "
                    "(jointly infeasible — would require step-boundary preemption, §6.4)")
        return {r.request.request_id: r.output() for r in runners}
