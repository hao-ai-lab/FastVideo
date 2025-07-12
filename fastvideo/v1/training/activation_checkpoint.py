import collections
from enum import Enum

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper)

TRANSFORMER_BLOCK_NAMES = [
    "blocks",
    "double_blocks",
    "single_blocks",
    "transformer_blocks",
    "temporal_transformer_blocks",
    "transformer_double_blocks",
    "transformer_single_blocks",
]


class CheckpointType(str, Enum):
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
        n_layer: int = 1) -> torch.nn.Module:
    if checkpointing_type == CheckpointType.FULL:
        module = _apply_activation_checkpointing_blocks(module)
    elif checkpointing_type == CheckpointType.OPS:
        module = _apply_activation_checkpointing_ops(
            module, _SELECTIVE_ACTIVATION_CHECKPOINTING_OPS)
    elif checkpointing_type == CheckpointType.BLOCK_SKIP:
        module = _apply_activation_checkpointing_blocks(module, n_layer)
    else:
        raise ValueError(
            f"Checkpointing type '{checkpointing_type}' not supported. Supported types are {CheckpointType.__members__.keys()}"
        )
    return module


def _apply_activation_checkpointing_blocks(module: torch.nn.Module,
                                           n_layer: int | None = None
                                           ) -> torch.nn.Module:
    for transformer_block_name in TRANSFORMER_BLOCK_NAMES:
        blocks: torch.nn.Module = getattr(module, transformer_block_name, None)
        if blocks is None:
            continue
        for index, (layer_id, block) in enumerate(blocks.named_children()):
            if n_layer is None or index % n_layer == 0:
                block = checkpoint_wrapper(block, preserve_rng_state=False)
                blocks.register_module(layer_id, block)
    return module


def _apply_activation_checkpointing_ops(module: torch.nn.Module,
                                        ops) -> torch.nn.Module:
    from torch.utils.checkpoint import (CheckpointPolicy,
                                        create_selective_checkpoint_contexts)

    def _get_custom_policy(meta: dict[str, int]) -> CheckpointPolicy:

        def _custom_policy(ctx, func, *args, **kwargs):
            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func == torch.ops.aten.mm.default:
                meta[mm_count_key] += 1
            # Saves output of all compute ops, except every second mm
            to_save = func in ops and not (func == torch.ops.aten.mm.default
                                           and meta[mm_count_key] % 2 == 0)
            return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

        return _custom_policy

    def selective_checkpointing_context_fn():
        meta: dict[str, int] = collections.defaultdict(int)
        return create_selective_checkpoint_contexts(_get_custom_policy(meta))

    return checkpoint_wrapper(module,
                              context_fn=selective_checkpointing_context_fn,
                              preserve_rng_state=False)
