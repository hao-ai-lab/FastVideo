# SPDX-License-Identifier: Apache-2.0
"""CPU offload for frozen inference modules without the device-to-host copy.

``module.to(device)`` / ``module.to("cpu")`` round-trips every parameter and
buffer through *pageable* host memory: the host-to-device copy of a pageable
tensor is staged and slow, and the device-to-host copy allocates fresh
pageable storage each time, so a pinned copy made at load time is lost on the
first offload. For a frozen module the weights on the device never change, so
copying them back is pure waste.

``load`` keeps one pinned host copy per tensor (made once, the first time) and
copies it to the device with an asynchronous, stream-ordered ``copy_``.
``unload`` simply points the parameters back at the pinned host copies and lets
the device storage go. Net: one pinned H2D per request, no D2H at all, and the
module lives on the host between requests exactly as before.

Numerics are untouched: the device tensors are byte copies of the same weights.
"""
from __future__ import annotations

import torch
from torch import nn

from fastvideo.logger import init_logger
from fastvideo.utils import is_pin_memory_available

logger = init_logger(__name__)

_HOST_ATTR = "_pinned_offload_host"


def _tensors(module: nn.Module):
    for name, p in module.named_parameters(recurse=True):
        yield name, p
    for name, b in module.named_buffers(recurse=True):
        yield "buffer:" + name, b


def _host_copies(module: nn.Module, pin: bool) -> dict[str, torch.Tensor]:
    host = getattr(module, _HOST_ATTR, None)
    if host is not None:
        return host
    host = {}
    total = 0
    for name, t in _tensors(module):
        src = t.data
        if src.device.type != "cpu":
            src = src.to("cpu")
        if pin and not src.is_pinned():
            src = src.pin_memory()
        host[name] = src
        total += src.numel() * src.element_size()
    setattr(module, _HOST_ATTR, host)
    logger.info("pinned_offload: kept %.2f GB of %s on the host (%s)", total / 1e9, type(module).__name__,
                "pinned" if pin else "pageable")
    return host


def load(module: nn.Module, device: torch.device, pin: bool = True) -> nn.Module:
    """Put ``module`` on ``device``, copying from the pinned host copies."""
    if device.type != "cuda":
        return module.to(device)
    pin = pin and is_pin_memory_available()
    host = _host_copies(module, pin)
    for name, t in _tensors(module):
        if t.data.device == device:
            continue
        src = host[name]
        dst = torch.empty_like(src, device=device)
        dst.copy_(src, non_blocking=src.is_pinned())
        t.data = dst
    return module


def unload(module: nn.Module) -> nn.Module:
    """Point ``module`` back at its host copies; no device-to-host copy."""
    host = getattr(module, _HOST_ATTR, None)
    if host is None:
        return module.to("cpu")
    for name, t in _tensors(module):
        if t.data.device.type == "cpu":
            continue
        t.data = host[name]
    return module
