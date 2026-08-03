# SPDX-License-Identifier: Apache-2.0
"""FastVideo device and CPU-offload lifecycle for H3 components."""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.distributed import get_local_torch_device


def module_device(module: Any, fallback: torch.device | None = None) -> torch.device:
    parameters = getattr(module, "parameters", None)
    parameter = None if parameters is None else next(parameters(), None)
    if parameter is not None:
        return torch.device(parameter.device)
    configured = getattr(module, "_fastvideo_input_device", None)
    if configured is not None:
        return torch.device(configured)
    return torch.device("cpu") if fallback is None else fallback


def move_module_to_local_device(module: Any) -> tuple[Any, torch.device, bool]:
    """Move a CPU-parked component to FastVideo's execution device."""
    target_device = get_local_torch_device()
    moved_for_forward = module_device(module) != target_device
    if moved_for_forward:
        module = module.to(target_device)
    return module, module_device(module, fallback=target_device), moved_for_forward


def maybe_offload_module(module: Any, enabled: bool) -> Any:
    """Return a component to CPU after its last forward in the request."""
    return module.to("cpu") if enabled else module


__all__ = ["maybe_offload_module", "module_device", "move_module_to_local_device"]
