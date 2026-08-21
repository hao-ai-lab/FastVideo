# SPDX-License-Identifier: Apache-2.0
"""Activation checkpointing policies for the modular training framework.

The modular trainer owns these policies under ``fastvideo.train``, which keeps
model plugins within one training package.
"""

import collections
from enum import Enum
from typing import Any

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


_SELECTIVE_ACTIVATION_CHECKPOINTING_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}


def apply_activation_checkpointing(
    module: torch.nn.Module,
    checkpointing_type: str = CheckpointType.FULL,
    n_layer: int = 1,
) -> torch.nn.Module:
    """Apply the selected activation checkpointing policy to a module."""
    if checkpointing_type == CheckpointType.FULL:
        module = _apply_activation_checkpointing_blocks(module)
    elif checkpointing_type == CheckpointType.OPS:
        module = _apply_activation_checkpointing_ops(
            module,
            _SELECTIVE_ACTIVATION_CHECKPOINTING_OPS,
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


def _apply_activation_checkpointing_ops(
    module: torch.nn.Module,
    ops: set[Any],
) -> torch.nn.Module:
    """Checkpoint a module while retaining selected operation outputs."""
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

    def _get_custom_policy(meta: dict[str, int]):
        """Build a policy that alternates matrix-multiply output retention."""

        def _custom_policy(ctx, func, *args, **kwargs):
            """Retain selected expensive operations during recomputation."""
            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func == torch.ops.aten.mm.default:
                meta[mm_count_key] += 1
            # Retain compute outputs except every second matrix multiplication.
            to_save = func in ops and not (func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0)
            return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

        return _custom_policy

    def selective_checkpointing_context_fn():
        """Create independent operation counters for one checkpointed call."""
        meta: dict[str, int] = collections.defaultdict(int)
        return create_selective_checkpoint_contexts(_get_custom_policy(meta))

    # Selective checkpointing wraps modules without stochastic masks that must
    # replay during recomputation.
    return checkpoint_wrapper(
        module,
        context_fn=selective_checkpointing_context_fn,
        preserve_rng_state=False,
    )
