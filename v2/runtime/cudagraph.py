"""Piecewise CUDA-graph capture/replay at the driven-loop step boundary (Path A; design_v3 §6.2).

The chosen optimization path (Path A) is hand-fused kernels behind the registry + piecewise CUDA
graphs captured at the step boundary — NOT a compiler. This module is the capture/replay LIFECYCLE:
when to capture a step into a graph, when to replay it, when to eager-break (a data-dependent step a
static graph cannot hold), and when to invalidate (shape or resident-weights change).

On a real GPU box, ``capture`` records the step's kernel launches into a ``torch.cuda.CUDAGraph``
bound to static input/output buffers, and ``replay`` copies the current step's inputs into those
buffers and relaunches the graph with zero Python and zero kernel-relaunch overhead — which is
exactly what makes the long Inductor *compile* avoidable (capture is cheap warmup, ~ms; no codegen).
Here, with no GPU, replay re-executes the SAME step thunk the capture ran (CPU cannot skip compute),
so this module does not model the *speedup* — it models and TESTS the parts that are
correctness-critical and bug-prone regardless of device:

  * the **capture key** — two steps may share a graph iff (device, arch, loop, shape, resident-weight
    versions, op-structure) match. Get this wrong and a stale/incompatible graph is replayed →
    silent corruption on a real box.
  * the **eager-break** — a step that is model-declared non-capturable (host RNG / data-dependent
    branch, e.g. the FlowGRPO SDE step) or that an interceptor overrode this step never touches the
    graph cache (vLLM's ``@eager_break_during_capture``).
  * **invalidation** — a resident-weights version bump (RL weight sync) or a shape change changes the
    key → recapture, so a graph never outlives the weights/shape it was captured for. This is
    AUTOMATIC: the resident-weight version is *in* the key, so a synced component's old graphs simply
    become unreachable and the next step recaptures.

The capturer NEVER executes a *stored* thunk — it always runs the *current* step's thunk (so it
cannot smear per-request state across interleaved requests). The ``CapturedGraph`` is the keyed
record used to account capture-vs-replay and to detect a same-key/different-workspace clash.

Honest scope — what this mini does NOT yet model (these bite a real GPU, not the CPU tests):
  * **Static-buffer refactor (not done).** The wan21 step body still allocates per call
    (``np.asarray``/``.astype``/host RNG). On real CUDA a capturable region must bind its I/O and
    scratch to reused static buffers first; ``workspace_bytes`` here is a *declared number*, not a
    buffer binding. So the workspace check below is a shape-change proxy (it compares the step's
    ``peak_activation_bytes``), NOT the kernel-level capture-safety backstop a real integration needs
    — the registered per-kernel ``workspace_bytes`` is declared in the matrix but not yet consulted.
  * **Capture cost is not budgeted.** ``WorkUnitKind.GRAPH_CAPTURE`` and
    ``ResourceRequest.graph_capture_size`` are declared placeholders; admission does not yet account
    the one-time capture (warmup) cost.
  * **Single reference path.** Only wan21's ``diffusion_denoise`` opts into ``breakable_cudagraph``;
    sibling cards reusing ``WanDenoiseLoop`` stay eager pending a per-card opt-in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class CapturedGraph:
    """A keyed record standing in for a captured ``torch.cuda.CUDAGraph`` + its static buffers."""
    key: tuple
    workspace_bytes: int
    replays: int = 0


class GraphCapturer:
    """Per-instance (cross-request) piecewise graph cache. Lives on the ModelInstance because a
    captured graph is valid only for that instance's resident weights + shapes."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled                  # off ⇒ always eager (the no-cudagraph baseline)
        self._graphs: dict[tuple, CapturedGraph] = {}
        self.stats = {"captures": 0, "replays": 0, "eager_breaks": 0}

    @staticmethod
    def key(plan: Any, instance: Any) -> tuple:
        """The capture key. Same key ⇒ a step may replay the captured graph. Components:
        backend (device, arch) · loop · batch-shape signature · resident-weight versions · the
        op-structure discriminators (CFG branch set / expert) the plan declares."""
        loop = instance.card.loops.get(plan.loop_id) if instance.card else None
        resident = tuple(sorted((cid, instance.version_of(cid))
                                for cid in (loop.shared_weight_components if loop else ())))
        p = instance.platform
        return (p.device, p.arch, plan.loop_id, plan.shape_sig.batch_key,
                resident, tuple(plan.graph_key))

    def dispatch(self, plan: Any, instance: Any, override: Any,
                 eager: Callable[[Any], Any]) -> Any:
        """Run one step under the capture lifecycle. ``eager(override)`` executes the step thunk and
        returns its StepResult|dict. Returns that result unchanged on every path (capture, replay,
        eager-break) — replay re-runs the current thunk, so the result is always correct for *this*
        step; the cache only governs accounting + structural-compatibility checks."""
        if not self.enabled:
            return eager(override)
        # Eager-break: a model-declared non-capturable step (host RNG / data-dependent) or a step an
        # interceptor overrode this iteration never enters the graph cache.
        if not getattr(plan, "capturable", True) or override is not None:
            self.stats["eager_breaks"] += 1
            return eager(override)

        key = self.key(plan, instance)
        ws = int(plan.resources.peak_activation_bytes)
        g = self._graphs.get(key)
        if g is None:
            # MISS → capture (cheap warmup): record the keyed graph, then run the step once.
            self._graphs[key] = CapturedGraph(key=key, workspace_bytes=ws)
            self.stats["captures"] += 1
            return eager(None)

        # HIT → replay. Safety net: everything under one key must need the same static workspace; a
        # mismatch means the key is insufficient (it would replay an incompatible graph on a GPU).
        if g.workspace_bytes != ws:
            raise RuntimeError(
                f"cudagraph key collision: key {key} captured a {g.workspace_bytes}-byte workspace "
                f"but this step needs {ws} — the capture key is insufficient and would corrupt a "
                f"real graph. This is a keying bug, not a runtime condition.")
        g.replays += 1
        self.stats["replays"] += 1
        return eager(None)

    def invalidate(self, component_ids: Any = None) -> int:
        """Evict captured graphs whose resident weights just changed (RL weight sync), mirroring
        ``CacheManager.invalidate_components``. Without this, a synced component's old graphs become
        unreachable (the version is in the key) but linger — a real GPU memory leak across FlowGRPO's
        per-iteration syncs. With ``component_ids=None``, clears all. Returns the count dropped."""
        if component_ids is None:
            n = len(self._graphs)
            self._graphs.clear()
            return n
        ids = set(component_ids)
        # key layout: (device, arch, loop_id, batch_key, resident=((cid, ver), ...), graph_key)
        drop = [k for k in self._graphs if any(cid in ids for cid, _ in k[4])]
        for k in drop:
            del self._graphs[k]
        return len(drop)

    # --- introspection (tests / diagnostics) --------------------------------- #
    @property
    def captured(self) -> int:
        return len(self._graphs)

    def clear(self) -> None:
        self._graphs.clear()
