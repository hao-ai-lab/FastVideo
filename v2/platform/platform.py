"""Platform — the detected ``(device, arch)`` that resolves both registries (design_v3 §17).

One ``Platform`` per pool. It owns:
  * the **device fallback chain** (e.g. ``accel → cpu``) — so a component/kernel a backend hasn't
    implemented falls back to the numpy reference (the terminal rung + parity oracle);
  * the **arch fallback chain** per device (e.g. cuda ``sm100→sm90→sm80→ptx→generic``) — so a kernel
    built for an older arch still resolves on a newer card;
  * ``build_component(spec, instance)`` — the single materialization seam (replaces a bare
    ``spec.factory`` call); and
  * ``kernels`` — a per-platform ``KernelTable`` that resolves an op once and caches the callable.

``Platform.detect()`` is honest: it returns a CPU/numpy platform unless torch+CUDA are actually
present (they are not in this environment). The (device, arch) machinery is fully exercised on the
CPU box by the pure-python ``accel`` stand-in backend — proving the dispatch is generic without
faking a GPU.
"""
from __future__ import annotations

from typing import Any, Iterator

from v2.platform.registry import COMPONENTS, KERNELS

_BACKENDS_LOADED = False


def ensure_backends_loaded() -> None:
    """Import the backend modules once so they self-register. Lazy → import-cycle-free."""
    global _BACKENDS_LOADED
    if _BACKENDS_LOADED:
        return
    _BACKENDS_LOADED = True
    # imported for side effects (each module calls register_component/register_kernel at import)
    from v2.platform.backends import accel, cpu, torch_cuda  # noqa: F401


# Arch fallback chains (newest → oldest → portable terminal). Resolution walks these in order.
_CUDA_ARCH_FALLBACK = ("sm100", "sm90", "sm80", "ptx", "generic")
_ACCEL_ARCH_FALLBACK = ("sm90", "sm80", "generic")
_PORTABLE_ARCHS = ("ptx", "generic")   # vendor-neutral terminals — valid on any device


def _arch_rank(arch: str) -> int:
    """``smXY`` → integer ``XY`` for comparison; portable terminals (ptx/generic/unknown) → -1."""
    if arch.startswith("sm") and arch[2:].isdigit():
        return int(arch[2:])
    return -1


def _suffix_from(chain: tuple[str, ...], arch: str) -> tuple[str, ...]:
    """The arch fallback tail for ``arch``: itself, then ONLY older real archs from ``chain``, then
    portable terminals. Never falls back to a *newer* arch — a kernel built for sm90 is binary-
    incompatible on an sm70 device, so resolution must only ever degrade to older/more-portable code.
    (A newer device, e.g. sm120, may still run older-arch kernels — those are kept.)
    """
    rank = _arch_rank(arch)
    tail: list[str] = [arch]
    for a in chain:
        if a in tail:
            continue
        ar = _arch_rank(a)
        if ar < 0:                              # portable terminal (ptx/generic): always valid
            tail.append(a)
        elif rank >= 0 and ar < rank:           # strictly-older real arch: a safe fallback
            tail.append(a)
        # else: newer-or-equal real arch → skip (would be binary-incompatible on this device)
    return tuple(tail)


class KernelTable:
    """A platform's resolved op→callable map. Resolves each op once (over the fallback chain),
    then caches — so the hot loop pays the registry walk at most once per (op, variant)."""

    def __init__(self, platform: "Platform"):
        self.platform = platform
        self._cache: dict[tuple[str, str], Any] = {}

    def get(self, op: str, variant: str = "default") -> Any:
        ck = (op, variant)
        if ck not in self._cache:
            reg = self.platform.resolve_kernel(op, variant)
            if reg is None:
                raise KeyError(
                    f"no kernel for op={op!r} variant={variant!r} on device chain "
                    f"{self.platform.device_chain} (and no numpy terminal registered)")
            self._cache[ck] = reg.fn
        return self._cache[ck]

    def resolved_source(self, op: str, variant: str = "default") -> str:
        """Which registered cell an op resolves to — used by parity/diagnostics tests."""
        reg = self.platform.resolve_kernel(op, variant)
        return "" if reg is None else f"{reg.key}@{reg.source}"


class Platform:
    def __init__(self, device: str, arch: str, device_chain: tuple[str, ...],
                 arch_chains: dict[str, tuple[str, ...]]):
        self.device = device
        self.arch = arch
        self.device_chain = tuple(device_chain)
        self.arch_chains = dict(arch_chains)
        self._kernels: KernelTable | None = None

    # --- the kernel table (lazy, cached) ------------------------------------- #
    @property
    def kernels(self) -> KernelTable:
        if self._kernels is None:
            ensure_backends_loaded()
            self._kernels = KernelTable(self)
        return self._kernels

    def arch_chain(self, device: str) -> tuple[str, ...]:
        return self.arch_chains.get(device, ("generic",))

    # --- candidate-key generators (the fallback chains, in priority order) --- #
    def _component_keys(self, kind: str, variant: str) -> Iterator[tuple]:
        for dev in self.device_chain:
            yield (kind, dev, variant)
            if variant != "default":
                yield (kind, dev, "default")

    def _kernel_keys(self, op: str, variant: str) -> Iterator[tuple]:
        for dev in self.device_chain:
            for arch in self.arch_chain(dev):
                yield (op, dev, arch, variant)
                if variant != "default":
                    yield (op, dev, arch, "default")

    # --- the two resolution seams -------------------------------------------- #
    def build_component(self, spec: Any, instance: Any, variant: str = "default") -> Any:
        """Materialize a component. Registry first (device chain), then ``spec.factory`` as the
        terminal numpy rung — so existing cards (factory-only) are unchanged, while a GPU/accel
        backend overrides by registering ``(kind, device, …)``."""
        ensure_backends_loaded()
        reg = COMPONENTS.resolve_first(self._component_keys(spec.kind, variant))
        if reg is not None:
            return reg.fn(spec, instance, self)
        if getattr(spec, "factory", None) is not None:
            return spec.factory(instance)          # the (kind, "cpu", "numpy") terminal rung
        raise RuntimeError(
            f"component kind={spec.kind!r} ({getattr(spec, 'component_id', '?')!r}) has no impl on "
            f"device chain {self.device_chain} and no factory terminal")

    def resolve_kernel(self, op: str, variant: str = "default"):
        ensure_backends_loaded()
        return KERNELS.resolve_first(self._kernel_keys(op, variant))

    # --- constructors -------------------------------------------------------- #
    @classmethod
    def cpu(cls) -> "Platform":
        return cls("cpu", "numpy", ("cpu",), {"cpu": ("numpy",)})

    @classmethod
    def accel(cls, arch: str = "sm90") -> "Platform":
        """A pure-python stand-in accelerator (CPU-resident) used to prove cross-device dispatch,
        arch fallback, and the parity oracle without a real GPU."""
        return cls("accel", arch, ("accel", "cpu"),
                   {"accel": _suffix_from(_ACCEL_ARCH_FALLBACK, arch), "cpu": ("numpy",)})

    @classmethod
    def cuda(cls, arch: str = "sm90") -> "Platform":
        """A real CUDA platform (resolves torch adapters + fastvideo-kernel cells). Only usable on a
        box where those cells are ``available``; declared here so detection has a target."""
        return cls("cuda", arch, ("cuda", "cpu"),
                   {"cuda": _suffix_from(_CUDA_ARCH_FALLBACK, arch), "cpu": ("numpy",)})

    @classmethod
    def detect(cls) -> "Platform":
        """Honest detection: CUDA iff torch+CUDA are actually importable, else CPU/numpy."""
        ensure_backends_loaded()
        import importlib.util
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    major, minor = torch.cuda.get_device_capability()
                    return cls.cuda(f"sm{major}{minor}")
            except Exception:
                pass
        return cls.cpu()

    def __repr__(self) -> str:
        return f"Platform(device={self.device!r}, arch={self.arch!r}, chain={self.device_chain})"


# --------------------------------------------------------------------------- #
# Matrix dumps — enumerate the full (op/kind × device × arch × variant) grid,  #
# including unavailable backends, without importing torch or touching a GPU.   #
# --------------------------------------------------------------------------- #
def kernel_matrix() -> list[dict[str, Any]]:
    ensure_backends_loaded()
    return KERNELS.manifest()


def component_matrix() -> list[dict[str, Any]]:
    ensure_backends_loaded()
    return COMPONENTS.manifest()
