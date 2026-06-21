"""Policy base classes — the *default* step decomposition.

Policies (CFG, expert routing, precision, flow-shift) delete duplication for
the families that fit; they are never required. A family whose math is braided ships a custom
``next``/``advance`` and uses these as a library (LTX-2's multi-pass guidance does exactly
that). Policy *bindings* resolve at build; policy *state* is per-request in ``LoopState``
(the adaptive-gate cached delta).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from v2.core.loop.contracts import StepContext
from v2.core.loop.sampler import build_flow_sigmas


# --------------------------------------------------------------------------- #
# FlowShiftPolicy — resolution-bucket shift + sigma schedule                  #
# --------------------------------------------------------------------------- #
class FlowShiftPolicy:
    """config-driven flow-shift lookup (e.g. Wan 480p shift=3.0, 720p=5.0)."""

    def __init__(self, shift: float = 3.0, bucket_lookup: dict[int, float] | None = None):
        self.shift = shift
        self.bucket_lookup = bucket_lookup or {}

    def shift_for(self, height: int = 0, width: int = 0) -> float:
        return self.bucket_lookup.get(height * width, self.shift)

    def build_schedule(self,
                       num_steps: int,
                       height: int = 0,
                       width: int = 0,
                       sigmas: list[float] | None = None) -> np.ndarray:
        if sigmas is not None:  # explicit distilled schedule (LTX-2)
            return np.asarray(sigmas, dtype=np.float64)
        return build_flow_sigmas(num_steps, shift=self.shift_for(height, width))


# --------------------------------------------------------------------------- #
# PrecisionPolicy — autocast / scheduler-step dtype control                   #
# --------------------------------------------------------------------------- #
class PrecisionPolicy:
    """Replaces ``prefix=='Flux'`` autocast hacks + ``scheduler_step_in_fp32``."""

    def __init__(self, compute_dtype: str = "float32", scheduler_step_in_fp32: bool = True):
        self.compute_dtype = compute_dtype
        self.scheduler_step_in_fp32 = scheduler_step_in_fp32

    def cast(self, arr: Any) -> Any:
        # Array-preserving: a device (torch) tensor is cast in place on its device — never pulled to
        # host. numpy on CPU is unchanged. (torch is imported lazily, only on the tensor path.)
        if isinstance(arr, np.ndarray):
            return np.asarray(arr, dtype=np.dtype(self.compute_dtype))
        import torch
        return arr.to(getattr(torch, self.compute_dtype, torch.float32))

    @property
    def scheduler_dtype(self):
        return np.float32 if self.scheduler_step_in_fp32 else np.dtype(self.compute_dtype)


# --------------------------------------------------------------------------- #
# ExpertRouting — Wan2.2 boundary-timestep transformer switch                 #
# --------------------------------------------------------------------------- #
class ExpertRouting(ABC):

    @abstractmethod
    def expert_for(self, ctx: StepContext) -> str:
        ...


class NoRouting(ExpertRouting):
    """Single-expert models (Wan2.1 1.3B): always the same component."""

    def __init__(self, component_id: str = "transformer"):
        self.component_id = component_id

    def expert_for(self, ctx: StepContext) -> str:
        return self.component_id


class BoundaryTimestepRouting(ExpertRouting):
    """Wan2.2 ``boundary_ratio`` switch between two transformers."""

    def __init__(self, high_noise: str, low_noise: str, boundary: float = 0.5):
        self.high_noise = high_noise
        self.low_noise = low_noise
        self.boundary = boundary

    def expert_for(self, ctx: StepContext) -> str:
        return self.high_noise if ctx.sigma >= self.boundary else self.low_noise
