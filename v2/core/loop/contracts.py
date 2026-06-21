"""The driven-loop contract — a serializable state machine the runtime drives:

    state = loop.init(req, model_state, ctx)
    while True:
        plan = loop.next(state)              # describe next step; NO GPU kernels
        if isinstance(plan, Done): break
        result = ctx.execute(plan)           # ← THE INVERSION POINT (runtime owns it)
        state = loop.advance(state, result)  # fold result in; decide what's next
        for chunk in plan.emits: ctx.emit(chunk)
    return loop.finalize(state)

Two properties this contract buys:
  * content-adaptive steps are natural — ``next`` reads ``state``, which already folded
    in the previous ``StepResult`` via ``advance``;
  * cross-request state safety is *structural* — all per-request mutable state lives in
    ``LoopState`` (incl. ``plugin_state`` per request/CFG-branch), never module globals,
    so interleaving requests through one ModelInstance cannot smear state.

This module is pure stdlib (tensors typed as ``TensorLike``) so it imports no backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from v2.core.enums import ExecutionProfile, WorkUnitKind
from v2.core.types import TensorLike
from v2.core.request.streams import StreamChunk


# --------------------------------------------------------------------------- #
# Shape / resources / cache plan / placement                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ShapeSignature:
    """Batch-compatibility + graph-capture key."""
    kind: WorkUnitKind
    dims: tuple[int, ...] = ()
    dtype: str = "float32"
    extra: tuple[tuple[str, Any], ...] = ()  # e.g. (("cfg","classic"),("expert","e0"))

    @property
    def work_units(self) -> int:
        u = 1
        for d in self.dims:
            u *= max(int(d), 1)
        return u

    @property
    def batch_key(self) -> tuple:
        """Two WorkPlans batch together iff their batch_keys are equal."""
        return (self.kind, self.dims, self.dtype, self.extra)


@dataclass
class ResourceRequest:
    """Memory the pool's OOM guard reserves for a step (no compute/time accounting — pooled
    run-to-completion serving prices nothing)."""
    resident_bytes: int = 0
    peak_activation_bytes: int = 0
    cache_blocks: dict[str, int] = field(default_factory=dict)
    transfer_bytes: int = 0
    graph_capture_size: int = 0


@dataclass
class CacheOp:
    cache_class: str
    key: Any  # CacheKey
    nbytes: int = 0


@dataclass
class CachePlan:
    reads: list[CacheOp] = field(default_factory=list)
    writes: list[CacheOp] = field(default_factory=list)


@dataclass
class PlacementHint:
    pool: str = "default"
    role: str = "denoise"
    device: str = "cpu"


# --------------------------------------------------------------------------- #
# WorkPlan / Done / StepResult / LoopResult                                   #
# --------------------------------------------------------------------------- #
@dataclass
class WorkPlan:
    """A typed description of the next step.

    ``run`` is the kernel thunk built by ``next()`` but NOT called there (kernel-free
    planning). The runtime's ``ctx.execute`` calls it — possibly after batching with
    other compatible plans — preserving 'the runtime owns iteration'.
    """
    loop_id: str
    instance_id: str
    kind: WorkUnitKind
    shape_sig: ShapeSignature
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    cache: CachePlan = field(default_factory=CachePlan)
    placement: PlacementHint = field(default_factory=PlacementHint)
    emits: list[StreamChunk] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)  # inspectable inputs (debug/parity)
    # the kernel thunk: run(model_instance, override=None) -> StepResult | dict.
    # Built by next() but NOT called there (kernel-free planning). ``override`` is an optional
    # interceptor-supplied forward result (e.g. a cached prediction); the step body still runs
    # the cheap solver step with it.
    run: Any = None
    label: str = ""
    # --- piecewise CUDA-graph capture ------------------------------------------------------------ #
    # ``capturable``: the model's static declaration that this step is safe to capture/replay — i.e.
    # no host RNG / data-dependent control flow inside the captured region. A stochastic step (the
    # FlowGRPO SDE rollout) sets this False, forcing the runtime to eager-break it.
    capturable: bool = True
    # ``graph_key``: extra op-structure discriminators (active CFG branch set, expert id) that change
    # the captured graph's *shape of computation*. Part of the capture key so a step with a different
    # branch set / expert never replays an incompatible graph. Empty ⇒ structure fixed by shape alone.
    graph_key: tuple = ()
    # The static-buffer capture form. A capturable step exposes its deterministic op
    # structure as ``graph_fn(model, workspace) -> StepResult`` reading EVERY per-step-varying input
    # (latent, sigmas, conditioning) from ``workspace`` — never from closure — plus ``graph_inputs``,
    # the dict of those current values. The runtime allocates address-stable buffers once per key and
    # rebinds ``graph_inputs`` into them in place each step (modeling CUDA static I/O buffers). Loops
    # that don't provide both stay on the eager path (the runtime eager-breaks them). ``run`` remains
    # the eager thunk (override / stochastic paths, and the no-capture baseline).
    graph_fn: Any = None
    graph_inputs: dict[str, Any] | None = None


@dataclass
class Done:
    """Sentinel returned by ``next`` when the loop is finished. ``finalize`` produces the
    actual LoopResult; ``result`` here is optional (kept for loops that want to carry it)."""
    result: LoopResult | None = None


@dataclass
class StepResult:
    output: dict[str, Any] = field(default_factory=dict)  # typed per-loop (e.g. {"noise_pred": ...})
    cache_writes: list[CacheOp] = field(default_factory=list)
    behavior: Any = None  # BehaviorRecord slice (rollout profile)


@dataclass
class LoopResult:
    outputs: dict[str, Any] = field(default_factory=dict)  # final latents / tokens / artifacts
    metrics: dict[str, float] = field(default_factory=dict)
    behavior: Any = None  # full BehaviorRecord (rollout)


# --------------------------------------------------------------------------- #
# LoopState — the per-request mutable container (NEVER module globals)         #
# --------------------------------------------------------------------------- #
@dataclass
class LoopState:
    """All per-request mutable state lives here (structural cross-request safety)."""
    loop_id: str
    instance_id: str
    request_id: str
    profile: ExecutionProfile = ExecutionProfile.SERVE
    step_idx: int = 0
    done: bool = False
    rng: Any = None  # seeded numpy Generator (per request)
    seed: int | None = None
    # common typed fields (resolved at init)
    latents: dict[str, TensorLike] = field(default_factory=dict)
    cond: dict[str, Any] = field(default_factory=dict)  # conditioning the loop fills from encoder slots
    timesteps: list[float] = field(default_factory=list)
    sigmas: list[float] = field(default_factory=list)
    # per-model extension (Cosmos3PackedSeq, MatrixGameState, ...) — typed by LoopSpec.extension_schema
    extension: Any = None
    # interceptor/policy state, keyed per plugin id AND per CFG branch
    plugin_state: dict[str, Any] = field(default_factory=dict)
    cache_handles: dict[str, Any] = field(default_factory=dict)
    # capture buffers (rollout): trajectory of per-step records
    trajectory: list[Any] = field(default_factory=list)
    scratch: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# StepContext — what policies read each step                                  #
# --------------------------------------------------------------------------- #
@dataclass
class StepContext:
    step_idx: int
    timestep: float
    sigma: float
    branch: str = "cond"  # current guidance branch
    active_expert_id: str | None = None  # set by ExpertRouting; observed by AdaptiveGateCFG
    sampler_coeffs: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Protocols: LoopContext (the runtime seam) and Loop (the model contract)      #
# --------------------------------------------------------------------------- #
@runtime_checkable
class LoopContext(Protocol):
    """The single seam the runtime exposes to a loop. The model never sees the
    scheduler; the scheduler never sees the model's math."""
    profile: ExecutionProfile

    def execute(self, plan: WorkPlan) -> StepResult:
        ...  # THE INVERSION POINT

    def emit(self, chunk: StreamChunk) -> None:
        ...

    def check_cancel(self) -> None:
        ...  # raises request.Cancelled at boundary

    def observe(self, event: str, **kw) -> None:
        ...  # observer bus hook


@runtime_checkable
class Loop(Protocol):
    """The model-owned control flow. Four methods; ``next`` is kernel-free."""

    def init(self, req: Any, model: Any, ctx: LoopContext) -> LoopState:
        ...

    def next(self, state: LoopState) -> WorkPlan | Done:
        ...

    def advance(self, state: LoopState, result: StepResult) -> LoopState:
        ...

    def finalize(self, state: LoopState) -> LoopResult:
        ...
