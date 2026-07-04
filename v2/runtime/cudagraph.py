"""Piecewise CUDA-graph capture/replay at the driven-loop step boundary.

The chosen optimization path is hand-fused kernels behind the registry + piecewise CUDA graphs
captured at the step boundary — NOT a compiler. This module is the capture/replay LIFECYCLE:
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
  * **invalidation** — a resident-weights version bump or a shape change changes the
    key → recapture, so a graph never outlives the weights/shape it was captured for. This is
    AUTOMATIC: the resident-weight version is *in* the key, so a synced component's old graphs simply
    become unreachable and the next step recaptures.

**Static buffers (the capture-safety discipline).** A captured graph references FIXED memory
addresses — replay copies the current step's inputs *into* those buffers in place, then relaunches;
only the contents change, never the op-structure or the addresses. So a capturable step must expose
its computation as ``graph_fn(model, workspace)`` that reads EVERY per-step-varying input (latent,
sigmas, conditioning) from the workspace — never from closure — and must not allocate fresh I/O each
call. ``StaticWorkspace`` models those buffers: ``alloc`` reserves them once per key; ``bind`` copies
the current step's values in via ``np.copyto``, which RAISES on a shape/dtype mismatch — so an
ill-fitting input cannot silently replay an incompatible graph (a real key-soundness backstop, not
the weak ``peak_activation_bytes`` proxy). Because all per-step/per-request inputs flow through the
workspace, the SAME captured buffers are correctly reused across interleaved requests; the capturer
re-invokes the *current* step's ``graph_fn`` (CPU can't bake op-structure into a replayable object),
which is sound because the key guarantees every step under it is structurally identical.

The per-key workspace is SHARED across requests (it lives on the instance). That is correct under
this engine's synchronous, one-step-at-a-time execution: ``dispatch`` rebinds then runs ``graph_fn``
atomically within a single step, and returns a copy of the output buffer, so an interleaved request's
next step overwrites the buffers only after the prior step fully consumed them. (The batch-of-N
interleave-parity gate exercises two same-key requests sharing this workspace and stays bit-identical.)
A truly concurrent / multi-stream executor would instead need a per-stream workspace pool.

Honest scope — what this mini still does NOT model (bites a real GPU, not the CPU tests):
  * **Capture cost is not budgeted.** ``WorkUnitKind.GRAPH_CAPTURE`` and
    ``ResourceRequest.graph_capture_size`` are declared placeholders; admission does not yet account
    the one-time capture (warmup) cost.
  * **Single reference path.** Only wan21's ``diffusion_denoise`` opts into ``breakable_cudagraph``;
    sibling cards reusing ``WanDenoiseLoop`` stay eager pending a per-card opt-in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

import numpy as np


class StaticWorkspace:
    """Address-stable buffers for a captured region (models CUDA static I/O buffers). Allocated once
    per capture key; ``bind`` updates contents IN PLACE (``np.copyto``), raising on a shape mismatch
    so an ill-fitting input can't replay an incompatible graph."""

    def __init__(self, buffers: dict[str, Any]):
        self._b = buffers

    @staticmethod
    def _mk(v: Any):
        return None if v is None else np.array(v)  # scalars become 0-d arrays; arrays are copied

    @classmethod
    def alloc(cls, inputs: dict[str, Any]) -> StaticWorkspace:
        b = {k: cls._mk(v) for k, v in inputs.items()}
        x = inputs.get("x")
        if x is not None:
            b["out"] = np.array(x)  # static output buffer, latent-shaped
        return cls(b)

    def bind(self, inputs: dict[str, Any]) -> None:
        for k, v in inputs.items():
            cur = self._b.get(k)
            if cur is None:
                if v is not None:
                    raise ValueError(f"workspace[{k!r}] was None at capture but has a value on replay")
                continue
            np.copyto(cur, v)  # in place; raises if shapes/dtypes don't fit

    def __getitem__(self, k: str):
        return self._b[k]


@dataclass
class CapturedGraph:
    """A keyed record standing in for a captured ``torch.cuda.CUDAGraph`` + its static buffers."""
    key: tuple
    workspace: Any  # StaticWorkspace (the reused fixed buffers), or None
    workspace_bytes: int
    replays: int = 0


class GraphCapturer:
    """Per-instance (cross-request) piecewise graph cache. Lives on the ModelInstance because a
    captured graph is valid only for that instance's resident weights + shapes."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled  # off ⇒ always eager (the no-cudagraph baseline)
        self._graphs: dict[tuple, CapturedGraph] = {}
        self.stats = {"captures": 0, "replays": 0, "eager_breaks": 0}

    @staticmethod
    def key(plan: Any, instance: Any) -> tuple:
        """The capture key. Same key ⇒ a step may replay the captured graph. Components:
        backend (device, arch) · loop · batch-shape signature · resident-weight versions · the
        op-structure discriminators (CFG branch set / expert) the plan declares."""
        loop = instance.card.loops.get(plan.loop_id) if instance.card else None
        resident = tuple(
            sorted((cid, instance.version_of(cid)) for cid in (loop.shared_weight_components if loop else ())))
        p = instance.platform
        return (p.device, p.arch, plan.loop_id, plan.shape_sig.batch_key, resident, tuple(plan.graph_key))

    def dispatch(self, plan: Any, instance: Any, override: Any, eager: Callable[[Any], Any]) -> Any:
        """Run one step under the capture lifecycle. ``eager(override)`` runs the eager step thunk
        (override / stochastic / no-capture paths). A capturable step runs its ``graph_fn`` against
        the keyed ``StaticWorkspace``; replay rebinds the current inputs into the SAME buffers and
        re-invokes the current ``graph_fn`` (sound because the key fixes the op-structure)."""
        if not self.enabled:
            return eager(override)
        # Eager-break: a model-declared non-capturable step (host RNG / data-dependent), an
        # interceptor override this iteration, or a step with no static graph form never enters the
        # graph cache.
        has_static = plan.graph_fn is not None and plan.graph_inputs is not None
        if not getattr(plan, "capturable", True) or override is not None or not has_static:
            self.stats["eager_breaks"] += 1
            return eager(override)

        key = self.key(plan, instance)
        ws_bytes = int(plan.resources.peak_activation_bytes)
        g = self._graphs.get(key)
        if g is None:
            # MISS → capture (cheap warmup): allocate the static buffers, bind inputs, run once.
            ws = StaticWorkspace.alloc(plan.graph_inputs)
            self._graphs[key] = CapturedGraph(key=key, workspace=ws, workspace_bytes=ws_bytes)
            self.stats["captures"] += 1
            return plan.graph_fn(instance, ws)

        # HIT → replay. peak-workspace proxy guard (declared static workspace must match)...
        if g.workspace_bytes != ws_bytes:
            raise RuntimeError(f"cudagraph key collision: key {key} captured a {g.workspace_bytes}-byte workspace "
                               f"but this step needs {ws_bytes} — the capture key is insufficient and would corrupt "
                               f"a real graph. This is a keying bug, not a runtime condition.")
        # ...and the real backstop: rebinding into the fixed buffers raises if shapes don't fit.
        g.workspace.bind(plan.graph_inputs)
        g.replays += 1
        self.stats["replays"] += 1
        return plan.graph_fn(instance, g.workspace)

    def invalidate(self, component_ids: Any = None) -> int:
        """Evict captured graphs whose resident weights just changed, mirroring
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
