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


# Mirrors fastvideo/train/utils/activation_checkpoint.py, which documents which
# attention paths this set can and cannot retain. The two stacks may not import
# each other, so a unit test keeps the copies equal.
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
    "fastvideo_kernel::block_sparse_attn_sm90",
    "fastvideo_kernel::block_sparse_attn_sm100a",
    "fastvideo_kernel::block_sparse_attn_triton",
}


def apply_activation_checkpointing(module: torch.nn.Module,
                                   checkpointing_type: str = CheckpointType.FULL,
                                   n_layer: int = 1) -> torch.nn.Module:
    if checkpointing_type == CheckpointType.FULL:
        module = _apply_activation_checkpointing_blocks(module)
    elif checkpointing_type == CheckpointType.OPS:
        # Wrapping each block, not the transformer root, keeps one block's
        # activations live during recompute instead of the whole model's.
        module = _apply_activation_checkpointing_blocks(module, context_fn=_selective_checkpointing_context_fn)
    elif checkpointing_type == CheckpointType.BLOCK_SKIP:
        module = _apply_activation_checkpointing_blocks(module, n_layer)
    else:
        raise ValueError(
            f"Checkpointing type '{checkpointing_type}' not supported. Supported types are {CheckpointType.__members__.keys()}"
        )
    return module


def _apply_activation_checkpointing_blocks(module: torch.nn.Module,
                                           n_layer: int | None = None,
                                           **checkpoint_kwargs) -> torch.nn.Module:
    applied = False
    for transformer_block_name in TRANSFORMER_BLOCK_NAMES:
        blocks: torch.nn.Module = getattr(module, transformer_block_name, None)
        if blocks is None:
            continue
        for index, (layer_id, block) in enumerate(blocks.named_children()):
            if n_layer is None or index % n_layer == 0:
                block = checkpoint_wrapper(block, preserve_rng_state=False, **checkpoint_kwargs)
                blocks.register_module(layer_id, block)
        applied = True
    if not applied:
        raise ValueError("Activation checkpointing is not applied successfully")
    return module


def _selective_checkpointing_context_fn():
    from torch.utils.checkpoint import (CheckpointPolicy, create_selective_checkpoint_contexts)

    def _custom_policy(ctx, func, *args, **kwargs):
        to_save = func.name() in _SELECTIVE_ACTIVATION_CHECKPOINTING_OP_NAMES
        return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(_custom_policy)
