"""Unit tests for the ``attn_only`` selective activation-checkpointing policy.

CPU-only (no GPU or distributed init): the op classifier, per-block wrapping,
and a real SDPA forward/backward parity check against none/full.
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


def test_is_attention_forward_matches_fastvideo_cute_ops():
    # FastVideo CuTe ops carry "flash_attn" + "forward" -> flash branch.
    assert _is_attention_forward("fastvideo::_flash_attn_default_forward")
    assert _is_attention_forward("fastvideo::_flash_attn_cute_forward")
    assert _is_attention_forward("fastvideo::_flash_attn_cute_varlen_forward")
    assert _is_attention_forward("fastvideo::_flash_attn_cute_fp4_forward")


def test_is_attention_forward_matches_video_sparse_attn():
    # VSA forward op; without this, attn_only recomputes VSA attention.
    assert _is_attention_forward("fastvideo_kernel::block_sparse_attn_triton")
    assert _is_attention_forward("fastvideo_kernel::block_sparse_attn_sm90")
    # VSA backward stays excluded.
    assert not _is_attention_forward("fastvideo_kernel::block_sparse_attn_backward_triton")
    assert not _is_attention_forward("fastvideo_kernel::block_sparse_attn_backward_sm90")


def test_is_attention_forward_matches_vmoba_flash_varlen():
    # vMoBA's MixedAttention issues _flash_attn_varlen_forward -> covered.
    assert _is_attention_forward("flash_attn::_flash_attn_varlen_forward")
    assert not _is_attention_forward("flash_attn::_flash_attn_varlen_backward")


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


# --- forward/backward numerical-parity coverage (CPU/SDPA, no GPU/dist) ---


class _AttnMLPBlock(nn.Module):
    """Real attention (aten SDPA, CPU) + MLP, so the policy sees a
    `_scaled_dot_product_*` forward to save and a GEMM to recompute."""

    def __init__(self, dim: int = 16, heads: int = 2):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):  # x: [B, S, D]
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.heads, d // self.heads)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))  # [B, H, S, Dh]
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, s, d)
        x = x + self.proj(attn)
        x = x + self.mlp(x)
        return x


class _AttnTransformer(nn.Module):
    def __init__(self, n_blocks: int = 2, dim: int = 16):
        super().__init__()
        self.blocks = nn.ModuleList([_AttnMLPBlock(dim) for _ in range(n_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _run_fwd_bwd(checkpointing_type, dim: int = 16, n_blocks: int = 2):
    """Fixed-seed fwd/bwd; returns loss, input grad, and parameter grads."""
    torch.manual_seed(0)
    model = _AttnTransformer(n_blocks=n_blocks, dim=dim)
    if checkpointing_type is not None:
        apply_activation_checkpointing(model, checkpointing_type=checkpointing_type)

    torch.manual_seed(1)
    x = torch.randn(2, 8, dim, requires_grad=True)
    out = model(x)
    loss = out.float().pow(2).mean()
    loss.backward()

    # named_parameters strips the checkpoint-wrapper prefix, so names align
    # across the wrapped and unwrapped models.
    # Drop the checkpoint-wrapper prefix so names align across wrapped/unwrapped.
    grads = {
        name.replace("._checkpoint_wrapped_module", ""): p.grad.detach().clone()
        for name, p in model.named_parameters()
    }
    return loss.detach().clone(), x.grad.detach().clone(), grads


def test_attn_only_matches_no_checkpointing_fwd_bwd():
    # attn_only == uncheckpointed path: loss, input grad, every param grad.
    ref_loss, ref_xgrad, ref_grads = _run_fwd_bwd(None)
    loss, xgrad, grads = _run_fwd_bwd(CheckpointType.ATTN_ONLY)

    torch.testing.assert_close(loss, ref_loss)
    torch.testing.assert_close(xgrad, ref_xgrad)
    assert grads.keys() == ref_grads.keys()
    for name in ref_grads:
        torch.testing.assert_close(grads[name], ref_grads[name], msg=f"grad mismatch for {name}")


def test_attn_only_matches_full_checkpointing_fwd_bwd():
    # attn_only and full save vs recompute differently -> identical grads.
    ref_loss, ref_xgrad, ref_grads = _run_fwd_bwd(CheckpointType.FULL)
    loss, xgrad, grads = _run_fwd_bwd(CheckpointType.ATTN_ONLY)

    torch.testing.assert_close(loss, ref_loss)
    torch.testing.assert_close(xgrad, ref_xgrad)
    for name in ref_grads:
        torch.testing.assert_close(grads[name], ref_grads[name], msg=f"grad mismatch for {name}")
