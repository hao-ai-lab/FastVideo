# SPDX-License-Identifier: Apache-2.0
"""Training-side LoRA utilities for ``fastvideo.train`` model plugins."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
import hashlib
import math
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

# Private loader attributes intentionally live on the transformed model.  The
# component loader constructs the model on ``meta`` before FSDP sees it, so the
# structural transform and the checkpoint-loading plan have to travel together.
LORA_CHECKPOINT_KEY_ALIASES_ATTR = "_fastvideo_checkpoint_key_aliases"
LORA_MISSING_PARAMETER_INITIALIZER_ATTR = "_fastvideo_missing_parameter_initializer"


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


def _initialize_lora_parameter(
    name: str,
    shape: torch.Size,
    dtype: torch.dtype,
    *,
    parameter_names: frozenset[str],
    seed: int,
) -> torch.Tensor | None:
    """Build one deterministic full LoRA tensor for the FSDP loader.

    Every rank constructs the same full tensor and lets the loader distribute
    it according to the parameter's FSDP placement.  Deriving a seed from the
    fully-qualified name keeps initialization independent of state-dict/set
    iteration order.
    """
    if name not in parameter_names:
        return None

    if name.endswith(".lora_B"):
        return torch.zeros(tuple(shape), dtype=dtype, device="cpu")
    if not name.endswith(".lora_A") or len(shape) != 2:
        raise ValueError(f"Unsupported LoRA parameter initialization request: {name} {tuple(shape)}")

    name_seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")
    generator = torch.Generator(device="cpu").manual_seed((int(seed) + name_seed) % (2**63 - 1))
    # nn.init.kaiming_uniform_(a=sqrt(5)) simplifies to +/- 1/sqrt(fan_in).
    bound = 1.0 / math.sqrt(int(shape[1]))
    value = torch.empty(tuple(shape), dtype=torch.float32, device="cpu")
    value.uniform_(-bound, bound, generator=generator)
    return value.to(dtype=dtype)


def finalize_lora_training(transformer: torch.nn.Module) -> int:
    """Freeze base weights and enable only adapter parameters after loading."""
    transformer.requires_grad_(False)
    count = 0
    for module in transformer.modules():
        if not isinstance(module, BaseLayerWithLoRA):
            continue
        if module.lora_A is None or module.lora_B is None:
            raise RuntimeError("Training LoRA wrapper is missing adapter parameters")
        module.base_layer.requires_grad_(False)
        module.lora_A.requires_grad_(True)
        module.lora_B.requires_grad_(True)
        count += 1
    if count == 0:
        raise ValueError("No training LoRA wrappers were found after checkpoint loading")
    transformer.train()
    return count


def enable_lora_training(
    transformer: torch.nn.Module,
    *,
    lora_rank: int,
    lora_alpha: int | None = None,
    lora_target_modules: Sequence[str] | None = None,
    prepare_for_fsdp: bool = False,
    initialization_seed: int = 0,
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
    checkpoint_key_aliases: dict[str, str] = {}
    lora_parameter_names: set[str] = set()
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
        for state_name in module.state_dict():
            checkpoint_key_aliases[f"{module_name}.{state_name}"] = f"{module_name}.base_layer.{state_name}"
        lora_parameter_names.update({f"{module_name}.lora_A", f"{module_name}.lora_B"})

    if not replacements:
        raise ValueError("No LoRA-compatible layers were found for the requested "
                         f"target modules: {target_modules}")

    for module_name, lora_layer in replacements:
        replace_submodule(transformer, module_name, lora_layer)

    if prepare_for_fsdp:
        # The FSDP loader consumes these aliases after the normal HF-to-native
        # mapping, then initializes adapter tensors that are intentionally absent
        # from the base checkpoint.  FSDP therefore owns base and adapter params
        # in the same parameter groups from the outset.
        setattr(transformer, LORA_CHECKPOINT_KEY_ALIASES_ATTR, checkpoint_key_aliases)
        setattr(
            transformer,
            LORA_MISSING_PARAMETER_INITIALIZER_ATTR,
            partial(
                _initialize_lora_parameter,
                parameter_names=frozenset(lora_parameter_names),
                seed=int(initialization_seed),
            ),
        )
    else:
        # Retain the legacy post-load path for existing model plugins.  New
        # distributed integrations must use ``prepare_for_fsdp=True``.
        _replicate_lora_parameters(transformer)
        finalize_lora_training(transformer)

    logger.info(
        "Enabled LoRA training with rank=%d alpha=%d on %d layers",
        rank,
        alpha,
        len(replacements),
    )
    return len(replacements)
