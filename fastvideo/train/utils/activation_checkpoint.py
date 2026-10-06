# SPDX-License-Identifier: Apache-2.0
"""Activation checkpointing policies for the modular training framework.

The modular trainer owns these policies under ``fastvideo.train``, which keeps
model plugins within one training package.
"""

from enum import Enum
from typing import Any

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (CheckpointWrapper, checkpoint_wrapper)

from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Model families expose transformer layers under these stable attributes. The
# shared policy discovers them without importing each model implementation.
_TRANSFORMER_BLOCK_NAMES = [
    "blocks",
    "double_blocks",
    "single_blocks",
    "transformer_blocks",
    "temporal_transformer_blocks",
    "transformer_double_blocks",
    "transformer_single_blocks",
    "text_transformer_blocks",
    "visual_transformer_blocks",
]


class CheckpointType(str, Enum):
    """Supported activation checkpointing policies."""

    FULL = "full"
    OPS = "ops"
    BLOCK_SKIP = "block_skip"


# Names rather than the op objects: the fastvideo ops register only when their
# backend module is imported, so torch.ops.fastvideo... would raise here on any
# build that has not loaded that backend.
_SELECTIVE_ACTIVATION_CHECKPOINTING_OP_NAMES = {
    "aten::_scaled_dot_product_flash_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_cudnn_attention",
    "fastvideo::_flash_attn_default_forward",
    "fastvideo::_flash_attn_cute_forward",
    "fastvideo::_flash_attn_cute_varlen_forward",
    "fastvideo::_flash_attn_cute_fp4_forward",
    "fastvideo::_flash_attn_no_pad_forward",
    "fastvideo::_flash_attn_varlen_qk_no_pad_forward",
    # VSA dispatches block_sparse_attn; video_sparse_attn is its Python entry
    # point, not an op, and naming that here would match nothing.
    "fastvideo_kernel::block_sparse_attn_sm90",
    "fastvideo_kernel::block_sparse_attn_sm100a",
    "fastvideo_kernel::block_sparse_attn_triton",
}

# No collective is listed. The replaced set named
# _c10d_functional::reduce_scatter_tensor, which is correct in torchtitan, where
# Megatron-style sequence parallelism reduce-scatters inside the forward.
# FSDP2's reduce-scatter runs in the post-backward hook, outside the region, and
# retaining the parameter all-gather that runs inside a block would keep every
# checkpointed block's unsharded weights resident at once, which is what FSDP
# exists to avoid. Ulysses sequence parallelism moves q/k/v and the attention
# output with all-to-alls inside each block; those run through
# torch.autograd.Function or in-place c10d collectives that no policy entry can
# retain, so they run again during recompute.

# Math SDPA is decomposed before this policy sees it. These paths run attention
# inside a torch.autograd.Function rather than a dispatcher op, so no policy
# entry can retain their outputs and they get full recomputation: VMoBA, SLA,
# ATTN_QAT_TRAIN, every FA3 path, FA4 masked self-attention and FA4 below sm90
# (both served by flash-attn 2's library functions), and CuTe VSA with 128- or
# 256-token blocks (FASTVIDEO_VSA_CUTEDSL=1). A block that retains nothing logs
# a one-time warning, which also catches a renamed op in the set above.
_warned_nothing_retained = False


def resolve_checkpointing_type(
    checkpointing_type: str | None,
    training_config: Any,
) -> str | None:
    """Return a role's checkpointing type, falling back to ``training.model``."""
    return checkpointing_type or getattr(
        getattr(training_config, "model", None),
        "enable_gradient_checkpointing_type",
        None,
    )


def is_activation_checkpointed(module: torch.nn.Module) -> bool:
    """Return whether any submodule runs inside an activation checkpoint."""
    return any(isinstance(submodule, CheckpointWrapper) for submodule in module.modules())


def apply_activation_checkpointing(
    module: torch.nn.Module,
    checkpointing_type: str = CheckpointType.FULL,
    n_layer: int = 1,
) -> torch.nn.Module:
    """Apply the selected activation checkpointing policy to a module."""
    if checkpointing_type == CheckpointType.FULL:
        module = _apply_activation_checkpointing_blocks(module)
    elif checkpointing_type == CheckpointType.OPS:
        # Wrapping each block, not the transformer root, keeps one block's
        # activations live during recompute instead of the whole model's.
        module = _apply_activation_checkpointing_blocks(
            module,
            context_fn=_selective_checkpointing_context_fn,
        )
    elif checkpointing_type == CheckpointType.BLOCK_SKIP:
        module = _apply_activation_checkpointing_blocks(module, n_layer)
    else:
        raise ValueError(f"Checkpointing type '{checkpointing_type}' not supported. "
                         f"Supported types are {CheckpointType.__members__.keys()}")
    return module


def _apply_activation_checkpointing_blocks(
    module: torch.nn.Module,
    n_layer: int | None = None,
    **checkpoint_kwargs: Any,
) -> torch.nn.Module:
    """Checkpoint every block or every nth block when ``n_layer`` is set."""
    applied = False
    for transformer_block_name in _TRANSFORMER_BLOCK_NAMES:
        blocks: torch.nn.Module | None = getattr(module, transformer_block_name, None)
        if blocks is None:
            continue
        for index, (layer_id, block) in enumerate(blocks.named_children()):
            if n_layer is None or index % n_layer == 0:
                # The wrapped transformer blocks contain no stochastic masks
                # that must replay during recomputation.
                checkpointed_block = checkpoint_wrapper(block, preserve_rng_state=False, **checkpoint_kwargs)
                blocks.register_module(layer_id, checkpointed_block)
        applied = True
    if not applied:
        raise ValueError("Activation checkpointing is not applied successfully")
    return module


def _selective_checkpointing_context_fn():
    """Retain selected expensive operations for one checkpointed block call."""
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

    retained = False

    def _custom_policy(ctx, func, *args, **kwargs):
        nonlocal retained
        # OpOverload.name() is e.g. "aten::_scaled_dot_product_flash_attention".
        to_save = func.name() in _SELECTIVE_ACTIVATION_CHECKPOINTING_OP_NAMES
        if not ctx.is_recompute:
            retained = retained or to_save
        elif not retained:
            _warn_nothing_retained()
        return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(_custom_policy)


def _warn_nothing_retained() -> None:
    global _warned_nothing_retained
    if _warned_nothing_retained:
        return
    _warned_nothing_retained = True
    logger.warning("Activation checkpointing 'ops' retained no operation output in a checkpointed block, so "
                   "that block recomputes in full, as under 'full'. Its attention backend does not reach a "
                   "retained dispatcher op; see fastvideo/train/utils/activation_checkpoint.py for the paths "
                   "'ops' can retain.")
