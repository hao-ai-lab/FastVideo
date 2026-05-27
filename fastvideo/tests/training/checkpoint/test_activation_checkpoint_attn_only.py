"""Unit tests for the ``attn_only`` selective activation-checkpointing policy.

CPU-only: exercises the op classifier and the per-block wrapping structure
without running a forward/backward (so no GPU or distributed init needed).
"""
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (CheckpointWrapper)

from fastvideo.training.activation_checkpoint import (CheckpointType, _attn_only_must_save,
                                                      _is_attention_forward, apply_activation_checkpointing)


def test_checkpoint_type_has_attn_only():
    assert CheckpointType.ATTN_ONLY == "attn_only"
    assert CheckpointType.ATTN_ONLY.value == "attn_only"


def test_is_attention_forward_matches_flash_and_sdpa_forward():
    # FA2 names the op "...forward", FA3 (flash_attn_interface) names it "...fwd",
    # aten exposes "_scaled_dot_product_*". The matcher keys on the op name so it
    # is robust to whichever backend is registered at runtime.
    assert _is_attention_forward("flash_attn::_flash_attn_forward")
    assert _is_attention_forward("flash_attn_3::fwd")
    assert _is_attention_forward(torch.ops.aten._scaled_dot_product_flash_attention.default)
    assert _is_attention_forward(torch.ops.aten._scaled_dot_product_efficient_attention.default)


def test_is_attention_forward_excludes_backward_and_unrelated_ops():
    # Backward ops must NOT be saved (we only want the forward output).
    assert not _is_attention_forward("flash_attn::_flash_attn_backward")
    assert not _is_attention_forward("flash_attn_3::bwd")
    assert not _is_attention_forward("aten::_scaled_dot_product_flash_attention_backward")
    # Unrelated compute (the FFN/QKV GEMM we deliberately recompute).
    assert not _is_attention_forward(torch.ops.aten.mm.default)
    assert not _is_attention_forward("aten::addmm")


def test_attn_only_must_save_attention_and_functional_collectives():
    # Attention forward -> saved.
    assert _attn_only_must_save("flash_attn::_flash_attn_forward")
    assert _attn_only_must_save(torch.ops.aten._scaled_dot_product_flash_attention.default)
    # Functional collectives (FSDP2 all-gather / reduce-scatter) -> saved, so a
    # collective is never re-issued during the backward recompute.
    assert _attn_only_must_save(torch.ops._c10d_functional.all_gather_into_tensor.default)
    assert _attn_only_must_save(torch.ops._c10d_functional.reduce_scatter_tensor.default)


def test_attn_only_must_save_recomputes_gemm_and_excludes_backward():
    assert not _attn_only_must_save(torch.ops.aten.mm.default)
    assert not _attn_only_must_save("aten::addmm")
    assert not _attn_only_must_save("flash_attn::_flash_attn_backward")


def test_attn_only_must_save_is_cached_and_consistent():
    # Repeated lookups (the hot path: called per op every step) return the same
    # decision; the cache must not flip a result.
    op = torch.ops.aten._scaled_dot_product_flash_attention.default
    first = _attn_only_must_save(op)
    assert first is True
    assert _attn_only_must_save(op) is first
    mm = torch.ops.aten.mm.default
    assert _attn_only_must_save(mm) is False
    assert _attn_only_must_save(mm) is False


class _DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):  # pragma: no cover - never executed in these tests
        return self.proj(x)


class _DummyTransformer(nn.Module):
    def __init__(self, n_blocks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(n_blocks)])


def test_attn_only_wraps_each_transformer_block():
    model = _DummyTransformer(n_blocks=3)
    returned = apply_activation_checkpointing(model, checkpointing_type=CheckpointType.ATTN_ONLY)

    # The module is wrapped in place and returned.
    assert returned is model
    assert len(model.blocks) == 3
    for block in model.blocks:
        assert isinstance(block, CheckpointWrapper)
        # The original block is preserved under the wrapper.
        assert isinstance(block._checkpoint_wrapped_module, _DummyBlock)


def test_attn_only_raises_when_no_transformer_blocks_found():
    class _NoBlocks(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

    try:
        apply_activation_checkpointing(_NoBlocks(), checkpointing_type=CheckpointType.ATTN_ONLY)
    except ValueError as exc:
        assert "attn_only" in str(exc)
    else:
        raise AssertionError("expected ValueError when no transformer blocks are present")


def test_apply_activation_checkpointing_rejects_unknown_type():
    try:
        apply_activation_checkpointing(_DummyTransformer(), checkpointing_type="not_a_real_type")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unsupported checkpointing type")
