# SPDX-License-Identifier: Apache-2.0
"""CPU offload for frozen inference modules, without the device-to-host copy.

``module.to("cpu")`` allocates fresh host storage and copies every parameter
and buffer back over PCIe. For a module that is only ever run under
``torch.no_grad()`` the weights on the device are byte-identical to the ones
that were copied in, so that copy is pure waste: ``unload`` points each
tensor's ``.data`` back at the host copy it came from and lets the device
storage go.

``load`` keeps one host copy per tensor, made the first time the module is
loaded, and copies it to the device. Net: one host-to-device copy per request
and no device-to-host copy at all, with the module living on the host between
requests exactly as before. Numerics are untouched -- the device tensors are
byte copies of the same weights -- and peak host memory can only go down,
because one buffer is reused instead of a new one being allocated per cycle.

The host copies are pinned when ``pin`` is set (``--pin-cpu-memory``, on by
default), which is what lets the remaining host-to-device copy run at full
PCIe speed instead of being staged through a bounce buffer. That trades the
module's size in non-pageable host memory for the bandwidth; ``pin=False``
keeps the copies pageable and everything else the same.
"""
from __future__ import annotations

import torch
from torch import nn

from fastvideo.logger import init_logger
from fastvideo.utils import is_pin_memory_available

logger = init_logger(__name__)

_HOST_ATTR = "_frozen_offload_host"


def _tensors(module: nn.Module):
    yield from module.named_parameters(recurse=True)
    for name, buf in module.named_buffers(recurse=True):
        yield "buffer:" + name, buf


def _host_copies(module: nn.Module, pin: bool) -> dict[str, torch.Tensor]:
    host = getattr(module, _HOST_ATTR, None)
    if host is not None:
        return host
    host, total = {}, 0
    for name, t in _tensors(module):
        src = t.data if t.data.device.type == "cpu" else t.data.to("cpu")
        if pin and not src.is_pinned():
            src = src.pin_memory()
        host[name] = src
        total += src.numel() * src.element_size()
    setattr(module, _HOST_ATTR, host)
    logger.info("frozen offload: holding %.2f GB of %s on the host (%s)", total / 1e9, type(module).__name__,
                "pinned" if pin else "pageable")
    return host


def load(module: nn.Module, device: torch.device, pin: bool = True) -> nn.Module:
    """Put a frozen ``module`` on ``device``, copying from its host copies."""
    if device.type != "cuda":
        return module.to(device)
    missing = [(name, t) for name, t in _tensors(module) if t.data.device != device]
    if not missing:
        # Already resident, which is what ``vae_cpu_offload=False`` leaves. Taking
        # host copies here would pull the whole module back over PCIe once and keep
        # a pinned mirror of it for the life of the process, for a module that never
        # leaves the device.
        return module
    host = _host_copies(module, pin and is_pin_memory_available())
    for name, t in missing:
        src = host[name]
        dst = torch.empty_like(src, device=device)
        dst.copy_(src, non_blocking=src.is_pinned())
        t.data = dst
    return module


def unload(module: nn.Module) -> nn.Module:
    """Point a frozen ``module`` back at its host copies; no device-to-host copy."""
    host = getattr(module, _HOST_ATTR, None)
    if host is None:
        return module.to("cpu")
    for name, t in _tensors(module):
        if t.data.device.type != "cpu":
            t.data = host[name]
    return module
