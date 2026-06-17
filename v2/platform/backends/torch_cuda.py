"""``cuda`` — the real torch/GPU backend, *declared but unavailable* on this box (design_v3 §17).

This environment has no torch and no GPU, so these cells cannot run here. But they are registered
anyway, gated by an ``available`` predicate (a ``find_spec`` check that never imports torch), so the
backend matrix is **enumerable without a GPU and without importing torch**: ``kernel_matrix()`` /
``component_matrix()`` list these rows with ``available=False``. (Registration is via the fixed
backend import list, not setuptools entry points — see the scope caveat in ``registry.py``; full
entry-point discovery is the real-package mechanism, not wired in this mini.)

On a real box (torch + CUDA present) ``available`` flips to ``True`` and ``Platform.detect()``
returns a ``cuda`` platform that resolves these instead of the numpy/accel rungs. The bodies below
are deliberately not implemented — they exist to anchor the matrix and to fail loudly if ever
resolved on a box that lacks the toolchain.
"""
from __future__ import annotations

import importlib.util
from typing import Any

from ..registry import FLOW_MATCH_STEP, FLOW_SDE_STEP, register_component, register_kernel


def _torch_present() -> bool:
    return importlib.util.find_spec("torch") is not None


def _cuda_available() -> bool:
    if not _torch_present():
        return False
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _unavailable(what: str):
    def _fn(*_a, **_k):
        raise RuntimeError(
            f"{what} requires a GPU box with torch + the fastvideo-kernel toolchain; "
            f"this cell is declared for matrix enumeration only")
    return _fn


# Component: the real DiT torch adapter (loads `spec.load_id`, places weights on the device).
register_component("dit", _unavailable("torch DiT adapter"), device="cuda",
                   available=_cuda_available, source="fastvideo.models.dits (torch adapter)")

# Kernels: the real fused solver primitives from fastvideo-kernel, at the Hopper rung.
register_kernel(FLOW_MATCH_STEP, _unavailable("cuda flow_match kernel"), device="cuda", arch="sm90",
                available=_cuda_available, source="fastvideo-kernel:flow_match (cuda)",
                workspace_bytes=1 << 16)
register_kernel(FLOW_SDE_STEP, _unavailable("cuda flow_sde kernel"), device="cuda", arch="sm90",
                available=_cuda_available, source="fastvideo-kernel:flow_sde (cuda)",
                workspace_bytes=1 << 17)
