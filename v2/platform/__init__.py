"""The multi-backend dispatch substrate (design_v3 §17).

Two tuple-keyed registries (``COMPONENTS``, ``KERNELS``) + a ``Platform`` that detects ``(device,
arch)`` and resolves both — with the numpy reference as the terminal fallback rung and parity
oracle. This package is the membrane that lets CPU, GPU, and other backends coexist: a model's
loops, policies, scheduler, caches, and training code never name a device; they call through the
component/kernel seams and the resolved ``Platform`` decides the implementation.

Importing this package is cheap and cycle-free (registry + platform classes only). The concrete
backend registrations in ``backends/`` are imported lazily on first ``Platform`` use.
"""
from __future__ import annotations

from .platform import (
    KernelTable,
    Platform,
    component_matrix,
    ensure_backends_loaded,
    kernel_matrix,
)
from .registry import (
    COMPONENTS,
    FLOW_MATCH_STEP,
    FLOW_SDE_STEP,
    KERNELS,
    register_component,
    register_kernel,
)

__all__ = [
    "Platform",
    "KernelTable",
    "COMPONENTS",
    "KERNELS",
    "register_component",
    "register_kernel",
    "ensure_backends_loaded",
    "kernel_matrix",
    "component_matrix",
    "FLOW_MATCH_STEP",
    "FLOW_SDE_STEP",
]
