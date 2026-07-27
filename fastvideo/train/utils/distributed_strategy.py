# SPDX-License-Identifier: Apache-2.0
"""Distributed model strategies for the modular trainer."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, TypeVar
from collections.abc import Generator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from fastvideo.logger import init_logger

logger = init_logger(__name__)

_ModelT = TypeVar("_ModelT", bound=torch.nn.Module)
SUPPORTED_DISTRIBUTED_STRATEGIES = frozenset({"fsdp", "ddp"})


def normalize_distributed_strategy(value: str | None) -> str:
    """Normalize and validate a modular-training strategy name."""
    strategy = str(value or "fsdp").strip().lower()
    if strategy not in SUPPORTED_DISTRIBUTED_STRATEGIES:
        choices = ", ".join(sorted(SUPPORTED_DISTRIBUTED_STRATEGIES))
        raise ValueError(f"training.distributed.strategy must be one of {{{choices}}}, "
                         f"got {strategy!r}")
    return strategy


def is_ddp_strategy(training_config: Any) -> bool:
    distributed = getattr(training_config, "distributed", None)
    return normalize_distributed_strategy(getattr(distributed, "strategy", "fsdp")) == "ddp"


@contextmanager
def _fork_model_rng(
    device: torch.device,
    seed: int,
) -> Generator[None, None, None]:
    fork_devices: list[int] = []
    if device.type == "cuda":
        fork_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        yield


def build_replicated_model_from_scratch(
    model_cls: type[_ModelT],
    init_params: dict[str, Any],
    *,
    device: torch.device,
    default_dtype: torch.dtype,
    seed: int,
) -> _ModelT:
    """Initialize one deterministic, full model replica on the local device.

    Model plugins call :func:`wrap_module_ddp` only after applying their
    trainable/frozen parameter policy. This is required because DDP builds its
    gradient reducer from the parameters that require gradients at wrap time.
    """
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        # Official MMAudio constructs on CPU and then calls ``.cuda()`` before
        # DDP. Preserve that RNG/initialization order for scratch parity.
        with _fork_model_rng(device, seed), torch.device("cpu"):
            model = model_cls(**init_params)
    finally:
        torch.set_default_dtype(old_dtype)
    return model.to(device=device)


class DelegatingDistributedDataParallel(DistributedDataParallel):
    """DDP wrapper that preserves a model plugin's public attributes.

    FastVideo model plugins access architecture attributes and helper methods
    such as ``latent_seq_len`` and ``normalize`` through ``transformer``.
    Native PyTorch DDP exposes these on ``module`` only; this adapter delegates
    unknown attributes while retaining the standard DDP forward/reducer.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = super().__getattr__("module")
            return getattr(module, name)


def wrap_module_ddp(
    module: _ModelT,
    *,
    device: torch.device,
    broadcast_buffers: bool = False,
) -> DelegatingDistributedDataParallel:
    """Wrap a full model replica like official MMAudio's DDP runner."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("DDP strategy requires an initialized process group")
    if not any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError("DDP strategy requires at least one trainable parameter")

    kwargs: dict[str, Any] = {
        "broadcast_buffers": bool(broadcast_buffers),
    }
    if device.type == "cuda":
        if device.index is None:
            raise ValueError("CUDA DDP requires a concrete local device index")
        kwargs.update({
            "device_ids": [device.index],
            "output_device": device.index,
        })
    wrapped = DelegatingDistributedDataParallel(module, **kwargs)
    logger.info(
        "Wrapped %s with native DDP on %s (broadcast_buffers=%s)",
        type(module).__name__,
        device,
        broadcast_buffers,
    )
    return wrapped


def unwrap_ddp_module(module: _ModelT) -> torch.nn.Module:
    """Return the underlying module for DDP, otherwise return the input."""
    if isinstance(module, DistributedDataParallel):
        return module.module
    return module


__all__ = [
    "DelegatingDistributedDataParallel",
    "SUPPORTED_DISTRIBUTED_STRATEGIES",
    "build_replicated_model_from_scratch",
    "is_ddp_strategy",
    "normalize_distributed_strategy",
    "unwrap_ddp_module",
    "wrap_module_ddp",
]
