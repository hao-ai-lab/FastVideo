"""The two tuple-keyed backend registries — the dispatch membrane.

Everything device/arch-specific resolves through one of two registries, keyed by a tuple:

  * ``COMPONENTS`` — keyed ``(kind, device, variant)``. Resolves which *weight-bearing* component
    implementation to materialize (the toy numpy module, a torch adapter on a GPU box, …). This is
    the generalization of ``ComponentSpec.factory``: the card's ``factory`` is simply the terminal
    ``(kind, "cpu", "numpy")`` rung that every other backend falls back to.
  * ``KERNELS`` — keyed ``(op, device, arch, variant)``. Resolves which *stateless primitive* kernel
    implements an op (the flow-match solver step, an SDE step, a gemm/norm on a real box). Arch is a
    fallback axis (sm100→sm90→…→generic), with the numpy reference as the terminal rung.

Two properties this module guarantees (the kernel-colocation answer made concrete):

  1. **Location is decoupled from dispatch.** A registration carries a key, not a path. Where the .cu
     (or .py) lives on disk — unified ``fastvideo-kernel/`` or co-located with a model — is invisible
     here; both call the same ``register_*``.
  2. **The matrix is enumerable without a GPU and without importing torch.** ``manifest()`` lists
     every declared cell — including ones whose backend is *unavailable* here (e.g. a torch/cuda cell
     on a CPU box, gated by a ``find_spec``-based ``available`` predicate that never imports torch) —
     so the parity/fallback matrix can be dumped on a laptop. ``resolve`` skips unavailable cells;
     ``manifest`` lists them.

     Scope caveat (honest): in this in-tree mini the cells are populated by a fixed backend import
     list (``Platform.ensure_backends_loaded`` imports ``backends/{cpu,accel,torch_cuda}``), not by
     setuptools entry points. So the matrix is complete for the backends in that list, but an
     *out-of-tree* or *model-co-located* backend would not self-enumerate until added to it — full
     entry-point discovery is the real-package mechanism and is deliberately not wired here.

This is a pure leaf module: stdlib only. The actual backend registrations live in ``backends/`` and
are imported lazily (``Platform.ensure_backends_loaded``) to keep this import-cycle-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable, Iterable

# --------------------------------------------------------------------------- #
# Op names — the stable identifiers a loop dispatches on (not free functions). #
# --------------------------------------------------------------------------- #
FLOW_MATCH_STEP = "flow_match_step"  # deterministic flow-match Euler solver step (ODE serve)
FLOW_SDE_STEP = "flow_sde_step"  # FlowGRPO stochastic step + log-prob (RL rollout)


def _always_available() -> bool:
    return True


@dataclass
class Registration:
    """One cell in a registry: a key, the implementation, and an availability predicate.

    ``available`` lets a cell be *declared* (so it shows up in ``manifest()``) while being
    *unresolvable* in this environment — e.g. a torch/cuda kernel on a CPU box declares
    ``available=lambda: torch.cuda.is_available()`` and is listed-but-skipped here.
    """
    key: tuple
    fn: Callable[..., Any]
    available: Callable[[], bool] = _always_available
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        try:
            return bool(self.available())
        except Exception:
            return False


class TupleRegistry:
    """A registry keyed by an ordered field tuple, with availability-aware resolution."""

    def __init__(self, name: str, key_fields: tuple[str, ...]):
        self.name = name
        self.key_fields = key_fields
        self._reg: dict[tuple, Registration] = {}

    def put(self,
            key: tuple,
            fn: Callable[..., Any],
            *,
            available: Callable[[], bool] | None = None,
            source: str = "",
            meta: dict[str, Any] | None = None) -> None:
        if len(key) != len(self.key_fields):
            raise ValueError(f"{self.name}: key {key} does not match fields {self.key_fields}")
        self._reg[key] = Registration(key=key,
                                      fn=fn,
                                      available=available or _always_available,
                                      source=source,
                                      meta=dict(meta or {}))

    def lookup(self, key: tuple) -> Registration | None:
        """Exact lookup, ignoring availability (used by ``manifest``/diagnostics)."""
        return self._reg.get(key)

    def resolve_first(self, candidate_keys: Iterable[tuple]) -> Registration | None:
        """First candidate key that is both registered AND available wins (the fallback chain)."""
        for key in candidate_keys:
            reg = self._reg.get(key)
            if reg is not None and reg.is_available():
                return reg
        return None

    def manifest(self) -> list[dict[str, Any]]:
        """Every declared cell as a flat dict — enumerable without importing/calling the kernels.

        This is what makes the parity matrix dumpable on a CPU box: it lists torch/cuda cells too,
        each with ``available=False`` here, so coverage is never silently truncated to 'what imported'.
        """
        rows: list[dict[str, Any]] = []
        for key, reg in sorted(self._reg.items(), key=lambda kv: tuple(map(str, kv[0]))):
            row = dict(zip(self.key_fields, key, strict=False))
            row["available"] = reg.is_available()
            row["source"] = reg.source
            row.update(reg.meta)  # surface declared metadata (e.g. workspace_bytes)
            rows.append(row)
        return rows


# The two registries, populated by ``backends/`` modules (lazily imported).
COMPONENTS = TupleRegistry("components", ("kind", "device", "variant"))
KERNELS = TupleRegistry("kernels", ("op", "device", "arch", "variant"))


def register_component(kind: str,
                       fn: Callable[..., Any],
                       *,
                       device: str,
                       variant: str = "default",
                       available: Callable[[], bool] | None = None,
                       source: str = "") -> None:
    """Register a component builder. ``fn(spec, instance, platform) -> live component``."""
    COMPONENTS.put((kind, device, variant), fn, available=available, source=source)


def register_kernel(op: str,
                    fn: Callable[..., Any],
                    *,
                    device: str,
                    arch: str,
                    variant: str = "default",
                    available: Callable[[], bool] | None = None,
                    source: str = "",
                    workspace_bytes: int = 0) -> None:
    """Register a stateless kernel. ``fn(*args, **kwargs)`` — same signature as its numpy reference.

    ``workspace_bytes`` is the kernel's static scratch requirement — the capture-safety contract: a
    kernel used inside a captured CUDA graph must draw its workspace from the pool-provided static
    buffer, not malloc per call, or replay corrupts. Declared here so the requirement is enumerable
    in the matrix; the numpy reference needs none (0)."""
    KERNELS.put((op, device, arch, variant),
                fn,
                available=available,
                source=source,
                meta={"workspace_bytes": int(workspace_bytes)})
