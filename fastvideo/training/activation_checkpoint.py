import collections
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
]


class CheckpointType(str, Enum):
    FULL = "full"
    OPS = "ops"
    BLOCK_SKIP = "block_skip"
    ATTN_ONLY = "attn_only"


_SELECTIVE_ACTIVATION_CHECKPOINTING_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}


def apply_activation_checkpointing(module: torch.nn.Module,
                                   checkpointing_type: str = CheckpointType.FULL,
                                   n_layer: int = 1) -> torch.nn.Module:
    if checkpointing_type == CheckpointType.FULL:
        module = _apply_activation_checkpointing_blocks(module)
    elif checkpointing_type == CheckpointType.OPS:
        module = _apply_activation_checkpointing_ops(module, _SELECTIVE_ACTIVATION_CHECKPOINTING_OPS)
    elif checkpointing_type == CheckpointType.BLOCK_SKIP:
        module = _apply_activation_checkpointing_blocks(module, n_layer)
    elif checkpointing_type == CheckpointType.ATTN_ONLY:
        module = _apply_activation_checkpointing_attn_only(module)
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


def _apply_activation_checkpointing_ops(module: torch.nn.Module, ops) -> torch.nn.Module:
    from torch.utils.checkpoint import (CheckpointPolicy, create_selective_checkpoint_contexts)

    def _get_custom_policy(meta: dict[str, int]) -> CheckpointPolicy:

        def _custom_policy(ctx, func, *args, **kwargs):
            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func == torch.ops.aten.mm.default:
                meta[mm_count_key] += 1
            # Saves output of all compute ops, except every second mm
            to_save = func in ops and not (func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0)
            return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

        return _custom_policy

    def selective_checkpointing_context_fn():
        meta: dict[str, int] = collections.defaultdict(int)
        return create_selective_checkpoint_contexts(_get_custom_policy(meta))

    return checkpoint_wrapper(module, context_fn=selective_checkpointing_context_fn, preserve_rng_state=False)


def _is_attention_forward(func) -> bool:
    """True for the (expensive-to-recompute) attention forward op, across
    backends: flash_attn lib FA2 (`flash_attn::_flash_attn_forward`), FA3
    (`flash_attn_3::fwd`), FastVideo custom ops, CuTe variants, and aten SDPA.
    Matched by op name so we don't depend on which op object is registered at
    import time (the active backend varies at runtime). Backward ops are
    excluded so only the forward output is saved."""
    s = str(func)
    if "backward" in s:
        return False
    # FA2 names the op "...forward", FA3 (flash_attn_interface) names it "...fwd".
    return ("flash_attn" in s and ("forward" in s or "fwd" in s)) or "_scaled_dot_product" in s


# Decision cache keyed by op: the set of distinct ops in a training step is tiny,
# so the string matching below runs once per unique op instead of once per call.
_ATTN_ONLY_SAVE_CACHE: dict = {}


def _attn_only_must_save(func) -> bool:
    """MUST_SAVE the attention forward output and any functional collective
    (`_c10d_functional.*` — e.g. FSDP2's all_gather_into_tensor / reduce_scatter).
    Recomputing a collective in backward re-issues communication: expensive, and
    a cross-rank ordering hazard that can deadlock. Everything else (the GEMM/FFN
    mm) is cheap to recompute and is not saved."""
    try:
        if func in _ATTN_ONLY_SAVE_CACHE:
            return _ATTN_ONLY_SAVE_CACHE[func]
    except TypeError:
        # Unhashable func: decide live, don't cache.
        return _is_attention_forward(func) or "_c10d_functional" in str(func)
    res = _is_attention_forward(func) or "_c10d_functional" in str(func)
    _ATTN_ONLY_SAVE_CACHE[func] = res
    return res


def _apply_activation_checkpointing_attn_only(module: torch.nn.Module) -> torch.nn.Module:
    """Per-block selective checkpointing that MUST_SAVE only the attention
    forward output (small — ~221 MB/block at seq 72k) and any collective output
    (don't recompute comm), and recomputes everything else (the GEMM/FFN mm
    intermediates are huge but cheap to recompute — saving them is what makes
    the stock 'ops' mode OOM). Eliminates the attention forward recompute that
    FULL mode pays (flash_fwd runs 2x under FULL) while staying within memory.
    Orthogonal to torch.compile."""
    from torch.utils.checkpoint import (CheckpointPolicy, create_selective_checkpoint_contexts)

    def _attn_only_policy(ctx, func, *args, **kwargs):
        return (CheckpointPolicy.MUST_SAVE if _attn_only_must_save(func) else CheckpointPolicy.PREFER_RECOMPUTE)

    def _ctx_fn():
        return create_selective_checkpoint_contexts(_attn_only_policy)

    applied = False
    for transformer_block_name in TRANSFORMER_BLOCK_NAMES:
        blocks: torch.nn.Module = getattr(module, transformer_block_name, None)
        if blocks is None:
            continue
        for layer_id, block in blocks.named_children():
            block = checkpoint_wrapper(block, context_fn=_ctx_fn, preserve_rng_state=False)
            blocks.register_module(layer_id, block)
        applied = True
    if not applied:
        raise ValueError("Activation checkpointing (attn_only) is not applied successfully")
    return module
