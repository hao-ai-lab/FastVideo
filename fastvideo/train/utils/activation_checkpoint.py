# SPDX-License-Identifier: Apache-2.0
"""Activation checkpointing policies for the modular training framework.

The modular trainer owns these policies under ``fastvideo.train``, which keeps
model plugins within one training package.
"""

from enum import Enum

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

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
    "aten::_scaled_dot_product_attention_math",
    "fastvideo::_flash_attn_default_forward",
    "fastvideo::_flash_attn_cute_forward",
    "fastvideo::_flash_attn_cute_varlen_forward",
    "fastvideo::_flash_attn_cute_fp4_forward",
    "fastvideo::_flash_attn_no_pad_forward",
    "fastvideo::_flash_attn_varlen_qk_no_pad_forward",
    # VSA dispatches block_sparse_attn; video_sparse_attn is its Python entry
    # point, not an op, and naming that here would match nothing.
    "fastvideo_kernel::block_sparse_attn_sm90",
    "fastvideo_kernel::block_sparse_attn_triton",
}

# The replaced set also named _c10d_functional::reduce_scatter_tensor, which
# never fires: under FSDP2 the reduce-scatter runs in the post-backward hook,
# outside the region. What does appear inside it is the parameter all-gather
# (c10d::_allgather_base_), and retaining that would hold every checkpointed
# block's unsharded weights at once, so it is deliberately not listed.

# VMoBA and the FA3 training path go through torch.autograd.Function rather than
# the dispatcher, so no policy can reach them; they get full recomputation.


def apply_activation_checkpointing(
    module: torch.nn.Module,
    checkpointing_type: str = CheckpointType.FULL,
    n_layer: int = 1,
) -> torch.nn.Module:
    """Apply the selected activation checkpointing policy to a module."""
    if checkpointing_type == CheckpointType.FULL:
        module = _apply_activation_checkpointing_blocks(module)
    elif checkpointing_type == CheckpointType.OPS:
        module = _apply_activation_checkpointing_ops(module)
    elif checkpointing_type == CheckpointType.BLOCK_SKIP:
        module = _apply_activation_checkpointing_blocks(module, n_layer)
    else:
        raise ValueError(f"Checkpointing type '{checkpointing_type}' not supported. "
                         f"Supported types are {CheckpointType.__members__.keys()}")
    return module


def _apply_activation_checkpointing_blocks(
    module: torch.nn.Module,
    n_layer: int | None = None,
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
                checkpointed_block = checkpoint_wrapper(block, preserve_rng_state=False)
                blocks.register_module(layer_id, checkpointed_block)
        applied = True
    if not applied:
        raise ValueError("Activation checkpointing is not applied successfully")
    return module


def _apply_activation_checkpointing_ops(module: torch.nn.Module) -> torch.nn.Module:
    """Checkpoint every block while retaining selected operation outputs."""
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

    def selective_checkpointing_context_fn():
        """Retain selected expensive operations during recomputation."""

        def _custom_policy(ctx, func, *args, **kwargs):
            # OpOverload.name() is e.g. "aten::_scaled_dot_product_flash_attention".
            to_save = func.name() in _SELECTIVE_ACTIVATION_CHECKPOINTING_OP_NAMES
            return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

        return create_selective_checkpoint_contexts(_custom_policy)

    applied = False
    for transformer_block_name in _TRANSFORMER_BLOCK_NAMES:
        blocks: torch.nn.Module | None = getattr(module, transformer_block_name, None)
        if blocks is None:
            continue
        for layer_id, block in blocks.named_children():
            # Selective checkpointing wraps modules without stochastic masks that
            # must replay during recomputation.
            checkpointed_block = checkpoint_wrapper(block,
                                                    context_fn=selective_checkpointing_context_fn,
                                                    preserve_rng_state=False)
            blocks.register_module(layer_id, checkpointed_block)
        applied = True
    if not applied:
        raise ValueError("Activation checkpointing is not applied successfully")
    return module
