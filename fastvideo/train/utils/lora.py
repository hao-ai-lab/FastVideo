# SPDX-License-Identifier: Apache-2.0
"""Training-side LoRA utilities for ``fastvideo.train`` model plugins."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from fastvideo.distributed import get_local_torch_device
from fastvideo.layers.lora.linear import (
    BaseLayerWithLoRA,
    get_lora_layer,
    replace_submodule,
)
from fastvideo.logger import init_logger

logger = init_logger(__name__)

DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "to_qkv",
    "to_gate_compress",
]


def _is_target_layer(
    module_name: str,
    target_modules: Sequence[str],
) -> bool:
    return any(target_name in module_name for target_name in target_modules)


def _is_excluded_layer(
    module_name: str,
    excluded_modules: Sequence[str],
) -> bool:
    return any(excluded in module_name for excluded in excluded_modules)


def _replicate_lora_parameters(
    transformer: torch.nn.Module,
) -> None:
    """Wrap LoRA params in replicated DTensors when distributed is active.

    The training loaders shard the base transformer with FSDP/HSDP before the
    model plugin sees it. Newly-added LoRA parameters therefore need to be
    explicit replicated DTensors so optimizers/checkpointing can treat them the
    same way across ranks.
    """

    if not dist.is_available() or not dist.is_initialized():
        return

    device = get_local_torch_device()
    if device.type != "cuda":
        return

    device_mesh = init_device_mesh(
        device.type,
        (dist.get_world_size(), 1),
        mesh_dim_names=["fake", "replicate"],
    )

    for module in transformer.modules():
        if not isinstance(module, BaseLayerWithLoRA):
            continue

        module.base_layer.requires_grad_(False)

        for attr_name in ("lora_A", "lora_B"):
            param = getattr(module, attr_name, None)
            if param is None:
                continue
            param.requires_grad_(True)
            if isinstance(param, DTensor):
                continue
            replicated = DTensor.from_local(
                param.detach(),
                device_mesh=device_mesh,
            )
            setattr(module, attr_name, nn.Parameter(replicated))


def enable_lora_training(
    transformer: torch.nn.Module,
    *,
    lora_rank: int,
    lora_alpha: int | None = None,
    lora_target_modules: Sequence[str] | None = None,
) -> int:
    """Replace supported linear layers with trainable LoRA wrappers.

    Returns the number of layers converted to LoRA.
    """

    rank = int(lora_rank)
    if rank <= 0:
        raise ValueError(f"lora_rank must be > 0, got {lora_rank!r}")

    alpha = int(lora_alpha) if lora_alpha is not None else rank
    target_modules = list(lora_target_modules or DEFAULT_LORA_TARGET_MODULES)
    arch_config = getattr(
        getattr(transformer, "config", None),
        "arch_config",
        None,
    )
    excluded_modules = list(
        getattr(arch_config, "exclude_lora_layers", []),
    )

    transformer.requires_grad_(False)

    replacements: list[tuple[str, BaseLayerWithLoRA]] = []
    for module_name, module in transformer.named_modules():
        if not module_name:
            continue
        if not _is_target_layer(module_name, target_modules):
            continue
        if _is_excluded_layer(module_name, excluded_modules):
            continue

        lora_layer = get_lora_layer(
            module,
            lora_rank=rank,
            lora_alpha=alpha,
            training_mode=True,
        )
        if lora_layer is None:
            continue
        replacements.append((module_name, lora_layer))

    if not replacements:
        raise ValueError(
            "No LoRA-compatible layers were found for the requested "
            f"target modules: {target_modules}"
        )

    for module_name, lora_layer in replacements:
        replace_submodule(transformer, module_name, lora_layer)

    _replicate_lora_parameters(transformer)
    transformer.train()

    logger.info(
        "Enabled LoRA training with rank=%d alpha=%d on %d layers",
        rank,
        alpha,
        len(replacements),
    )
    return len(replacements)
