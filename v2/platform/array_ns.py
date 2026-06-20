"""Array namespace — numpy on a CPU box, torch-on-device on a GPU box.

The denoise loop's math is already array-agnostic: CFG combine (``uncond + s·(cond−uncond)``) and the
flow-match Euler step (``x + (σ_next−σ_t)·v``) are pure arithmetic that runs identically on numpy and
torch. This namespace supplies the few NON-arithmetic helpers the loop still needs — birthing the
latent, dtype casts, and the single host round-trip at the request boundary — so on a GPU box the
latent stays resident on-device for the whole loop (no per-step host<->device copy) while the CPU/toy
path stays pure numpy (torch-free; the parity mini is unchanged).

The latent is still *seeded* with numpy (``rng.standard_normal``) and uploaded once via ``from_host``,
so a GPU run is bit-identical to the pre-on-device path (the noise values are the same; upload is
lossless). ``Platform.xp`` picks the namespace from the platform's device.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class NumpyNS:
    """The CPU/toy namespace: every op is a numpy identity (no torch dependency)."""

    device = "cpu"

    def from_host(self, a: Any) -> np.ndarray:
        return np.asarray(a, dtype=np.float32)

    def to_host(self, a: Any) -> np.ndarray:
        return np.asarray(a)

    def to_f32(self, a: Any) -> np.ndarray:
        return np.asarray(a, dtype=np.float32)

    def is_native(self, a: Any) -> bool:
        return isinstance(a, np.ndarray)


class TorchNS:
    """The GPU namespace: keeps arrays as device tensors; marshals host<->device only on demand."""

    def __init__(self, device: str = "cuda") -> None:
        import torch
        self._torch = torch
        self.device = device

    def from_host(self, a: Any) -> Any:
        t = self._torch
        if t.is_tensor(a):
            return a.to(self.device)
        return t.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)

    def to_host(self, a: Any) -> np.ndarray:
        t = self._torch
        if t.is_tensor(a):
            return a.detach().to("cpu", t.float32).numpy()
        return np.asarray(a)

    def to_f32(self, a: Any) -> Any:
        t = self._torch
        return a.float() if t.is_tensor(a) else np.asarray(a, dtype=np.float32)

    def is_native(self, a: Any) -> bool:
        return bool(self._torch.is_tensor(a))


def get_array_ns(device: str) -> Any:
    """The array namespace for a platform device: torch-on-device for cuda, numpy otherwise."""
    return TorchNS(device) if device == "cuda" else NumpyNS()
