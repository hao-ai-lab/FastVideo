from enum import Enum

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (checkpoint_wrapper)

TRANSFORMER_BLOCK_NAMES = [
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
    FULL = "full"
    OPS = "ops"
    BLOCK_SKIP = "block_skip"


# Attention and collectives are the outputs worth keeping: recomputing attention
# is quadratic in sequence length, and recomputing a collective re-issues
# communication. Matched by name because the fastvideo ops below do not exist
# until their backend module is imported, so naming them by identity here would
# force every backend to load and still miss the env-gated FA4 ones.
_SAVE_OP_PATTERNS = (
    "_scaled_dot_product",  # aten SDPA: flash / efficient / cudnn / math
    "flash_attn",  # fastvideo::_flash_attn_{default,cute,varlen,...}
    "video_sparse_attn",  # VSA
    "moba_attn",  # VMoBA
    "sage_attn",  # SageAttention 2 / 3
    "reduce_scatter",
    "all_gather",
)


def apply_activation_checkpointing(module: torch.nn.Module,
                                   checkpointing_type: str = CheckpointType.FULL,
                                   n_layer: int = 1) -> torch.nn.Module:
    if checkpointing_type == CheckpointType.FULL:
        module = _apply_activation_checkpointing_blocks(module)
    elif checkpointing_type == CheckpointType.OPS:
        module = _apply_activation_checkpointing_ops(module)
    elif checkpointing_type == CheckpointType.BLOCK_SKIP:
        module = _apply_activation_checkpointing_blocks(module, n_layer)
    else:
        raise ValueError(
            f"Checkpointing type '{checkpointing_type}' not supported. Supported types are {CheckpointType.__members__.keys()}"
        )
    return module


def _apply_activation_checkpointing_blocks(module: torch.nn.Module, n_layer: int | None = None) -> torch.nn.Module:
    applied = False
    for transformer_block_name in TRANSFORMER_BLOCK_NAMES:
        blocks: torch.nn.Module = getattr(module, transformer_block_name, None)
        if blocks is None:
            continue
        for index, (layer_id, block) in enumerate(blocks.named_children()):
            if n_layer is None or index % n_layer == 0:
                block = checkpoint_wrapper(block, preserve_rng_state=False)
                blocks.register_module(layer_id, block)
        applied = True
    if not applied:
        raise ValueError("Activation checkpointing is not applied successfully")
    return module


def _apply_activation_checkpointing_ops(module: torch.nn.Module) -> torch.nn.Module:
    from torch.utils.checkpoint import (CheckpointPolicy, create_selective_checkpoint_contexts)

    def selective_checkpointing_context_fn():

        def _custom_policy(ctx, func, *args, **kwargs):
            # OpOverload.name() is e.g. "aten::_scaled_dot_product_flash_attention".
            to_save = any(pattern in func.name() for pattern in _SAVE_OP_PATTERNS)
            return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

        return create_selective_checkpoint_contexts(_custom_policy)

    applied = False
    for transformer_block_name in TRANSFORMER_BLOCK_NAMES:
        blocks: torch.nn.Module = getattr(module, transformer_block_name, None)
        if blocks is None:
            continue
        for layer_id, block in blocks.named_children():
            block = checkpoint_wrapper(block, context_fn=selective_checkpointing_context_fn, preserve_rng_state=False)
            blocks.register_module(layer_id, block)
        applied = True
    if not applied:
        raise ValueError("Activation checkpointing is not applied successfully")
    return module
