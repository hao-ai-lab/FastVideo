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
"""
from __future__ import annotations

import torch
from torch import nn

from fastvideo.logger import init_logger

logger = init_logger(__name__)

_HOST_ATTR = "_frozen_offload_host"


def _tensors(module: nn.Module):
    yield from module.named_parameters(recurse=True)
    for name, buf in module.named_buffers(recurse=True):
        yield "buffer:" + name, buf


def _host_copies(module: nn.Module) -> dict[str, torch.Tensor]:
    host = getattr(module, _HOST_ATTR, None)
    if host is not None:
        return host
    host, total = {}, 0
    for name, t in _tensors(module):
        src = t.data if t.data.device.type == "cpu" else t.data.to("cpu")
        host[name] = src
        total += src.numel() * src.element_size()
    setattr(module, _HOST_ATTR, host)
    logger.info("frozen offload: holding %.2f GB of %s on the host", total / 1e9, type(module).__name__)
    return host


def load(module: nn.Module, device: torch.device) -> nn.Module:
    """Put a frozen ``module`` on ``device``, copying from its host copies."""
    if device.type != "cuda":
        return module.to(device)
    host = _host_copies(module)
    for name, t in _tensors(module):
        if t.data.device == device:
            continue
        src = host[name]
        dst = torch.empty_like(src, device=device)
        dst.copy_(src)
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
