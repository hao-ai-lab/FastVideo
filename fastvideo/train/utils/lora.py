# SPDX-License-Identifier: Apache-2.0
"""Training-side LoRA utilities for ``fastvideo.train`` model plugins."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate

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

_LORA_CONFIG_KEYS = ("enable", "rank", "alpha", "target_modules")


@dataclass
class LoraConfig:
    """Structured LoRA settings for one ``fastvideo.train`` model role.

    Parsed from the nested ``models.<role>.lora`` YAML block::

        lora:
          enable: true                       # default false
          rank: 16
          alpha: 32                          # defaults to rank when omitted
          target_modules: [to_q, to_k, to_v, to_out]

    ``enable`` is an explicit on/off switch so a config states its intent
    plainly: the presence of ``rank`` alone never silently flips a run into
    LoRA-only training.  When ``enable`` is false a still-present ``rank`` is
    ignored (with an INFO log), so a configured-but-off block is valid.
    """

    enable: bool = False
    rank: int | None = None
    alpha: int | None = None
    target_modules: list[str] | None = None

    def __post_init__(self) -> None:
        if self.rank is not None:
            self.rank = int(self.rank)
        if self.alpha is not None:
            self.alpha = int(self.alpha)
        if self.target_modules is not None:
            self.target_modules = list(self.target_modules)

        if self.enable:
            if self.rank is None:
                raise ValueError("models.<role>.lora.enable is true but lora.rank is unset "
                                 "— an explicit positive rank is required to enable LoRA")
            if self.rank <= 0:
                raise ValueError(f"models.<role>.lora.rank must be > 0, got {self.rank!r}")
        elif self.rank is not None:
            logger.info(
                "models.<role>.lora.rank=%s is set but lora.enable is false — "
                "LoRA will NOT be applied (model trains on its normal "
                "trainable path).", self.rank)

    @classmethod
    def coerce(
        cls,
        obj: LoraConfig | dict[str, Any] | None,
    ) -> LoraConfig | None:
        """Normalize a raw YAML mapping (or existing config) into a LoraConfig.

        Returns ``None`` when no ``lora`` block was given, which callers treat
        as "LoRA not configured" — identical in effect to ``enable: false``.
        """
        if obj is None:
            return None
        if isinstance(obj, LoraConfig):
            return obj
        if not isinstance(obj, dict):
            raise TypeError("models.<role>.lora must be a mapping or LoraConfig, got "
                            f"{type(obj).__name__}")
        unknown = set(obj) - set(_LORA_CONFIG_KEYS)
        if unknown:
            logger.warning("LoraConfig: ignoring unrecognized lora keys %s "
                           "(valid keys: %s)", sorted(unknown), list(_LORA_CONFIG_KEYS))
        return cls(
            enable=bool(obj.get("enable", False)),
            rank=obj.get("rank"),
            alpha=obj.get("alpha"),
            target_modules=obj.get("target_modules"),
        )


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


def _replicate_lora_parameters(transformer: torch.nn.Module, ) -> None:
    """Wrap LoRA params in replicated DTensors when distributed is active.

    The training loaders shard the base transformer with FSDP/HSDP before the
    model plugin sees it. Newly-added LoRA parameters therefore need to be
    explicit replicated DTensors so optimizers/checkpointing can treat them the
    same way across ranks.

    The mesh is reused from the FSDP-wrapped base_layer parameters rather than
    rebuilt via ``init_device_mesh`` — building a parallel mesh with a different
    topology than the one FSDP already registered can conflict with the
    existing mesh init.  ``placements=[Replicate()] * mesh.ndim`` is passed
    explicitly so the local tensor is treated as a replicated copy across all
    mesh dimensions (instead of falling back to a default Shard layout).
    """

    if not dist.is_available() or not dist.is_initialized():
        return

    device = get_local_torch_device()
    if device.type != "cuda":
        return

    # Look up the mesh that FSDP/HSDP already attached to a base_layer
    # parameter. Non-FSDP runs (e.g. single-GPU / non-distributed) won't have
    # any DTensor params here; in that case we leave LoRA params as plain
    # tensors, which is the correct local-only behavior.
    mesh: DeviceMesh | None = None
    for module in transformer.modules():
        if not isinstance(module, BaseLayerWithLoRA):
            continue
        for p in module.base_layer.parameters():
            if isinstance(p, DTensor):
                mesh = p.device_mesh
                break
        if mesh is not None:
            break

    if mesh is None:
        return

    placements = [Replicate()] * mesh.ndim

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
                device_mesh=mesh,
                placements=placements,
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
    excluded_modules = list(getattr(arch_config, "exclude_lora_layers", []), )

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
        raise ValueError("No LoRA-compatible layers were found for the requested "
                         f"target modules: {target_modules}")

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
