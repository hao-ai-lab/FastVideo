# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for capacity tails and the native VSA training dispatch.

Only the sparse kernel and device-support probe are substituted. The actual
tiling, sparse selection, gated compression and output reconstruction run.
These tests do not qualify the CUDA kernel's numerical accuracy.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import fastvideo.attention.backends.video_sparse_attn_h3 as h3
from fastvideo import envs


def _metadata(pad_tokens=0, tile_size=64):
    return h3.MiniMaxH3VSAMetadataBuilder().build(
        current_timestep=0,
        patch_size=(1, 2, 2),
        VSA_sparsity=0.5,
        packed_segments=(7, (3, 8, 10), 5, (5, 8, 12)),
        ref_keep_rate=0.5,
        tile_size=tile_size,
        device=torch.device("cpu"),
        pad_tokens=pad_tokens,
    )


def _sparse(q, k, v, mask, sizes):
    tile_size = q.shape[2] // sizes.numel()
    valid = torch.arange(tile_size)[None, :] < sizes[:, None]
    allowed = mask.repeat_interleave(tile_size, -2).repeat_interleave(tile_size, -1)
    allowed = allowed & valid.flatten()[None, None, None, :]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed), None


def _indices(mask):
    indices = torch.argsort(mask.to(torch.int32), dim=-1, descending=True, stable=True).to(torch.int32)
    return indices, mask.sum(-1, dtype=torch.int32)


def _native(q, k, v, indices, counts, sizes):
    selected = torch.arange(indices.shape[-1]) < counts[..., None]
    mask = torch.zeros_like(indices, dtype=torch.bool).scatter_(-1, indices.long(), selected)
    return _sparse(q, k, v, mask, sizes)


def _run(impl, values, metadata):
    tiled = impl.preprocess_qkv(values, metadata)
    q, k, v, gate = tiled.chunk(4, dim=0)
    output = impl.forward(q, k, v, gate, metadata)
    return impl.postprocess_output(output, metadata)


@pytest.mark.parametrize("tile_size", [64, 128])
def test_capacity_preserves_real_attention_and_gradients(monkeypatch, env_overrides, tile_size):
    env_overrides.enter_context(envs.FASTVIDEO_VSA_PTX_BLACKWELL.override(tile_size == 128))
    monkeypatch.setattr(h3, "block_sparse_attn_64_bhsd", _sparse)
    monkeypatch.setattr(h3, "_ptx_blackwell", SimpleNamespace(is_supported=lambda *_: True))
    monkeypatch.setattr(h3, "_ptx_blackwell_backward_is_supported", lambda *_: True)
    monkeypatch.setattr(h3, "block_sparse_attn_ptx_blackwell_op", _native)
    monkeypatch.setattr(h3, "map_to_index", _indices)
    real, padded = _metadata(tile_size=tile_size), _metadata(37, tile_size)
    assert real.video_tile_spans == padded.video_tile_spans
    assert real.span_sparsities == padded.span_sparsities
    assert torch.equal(real.variable_block_sizes, padded.variable_block_sizes)
    assert torch.equal(real.untile_combined_index, padded.untile_combined_index)
    assert real.variable_block_sizes.sum() == real.total_seq_length
    impl = h3.MiniMaxH3VSAImpl(num_heads=2, head_size=8, causal=False, softmax_scale=8**-0.5)
    values = torch.randn(4, real.total_seq_length, 2, 8, requires_grad=True)
    # Poison padding to prove it cannot affect pooling, selection or gradients.
    expanded = F.pad(values.detach(), (0, 0, 0, 0, 0, 37), value=float("nan")).requires_grad_()
    expected = _run(impl, values, real)
    actual = _run(impl, expanded, padded)
    torch.testing.assert_close(actual[:, :real.total_seq_length], expected, atol=0, rtol=0)
    assert torch.count_nonzero(actual[:, real.total_seq_length:]) == 0
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient[:, :real.total_seq_length])
    torch.testing.assert_close(expanded.grad[:, :real.total_seq_length], values.grad, atol=0, rtol=0)
    assert torch.isfinite(expanded.grad).all()
    assert torch.count_nonzero(expanded.grad[:, real.total_seq_length:]) == 0
    assert padded.tile_buf_holder.buffer is None


def test_native_128_grad_dispatch_never_calls_raw_forward(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("Training used a forward-only native operator")

    def native(q, k, v, indices, counts, sizes):
        calls.append(sizes.clone())
        return q + k + v, None

    monkeypatch.setattr(h3, "_ptx_blackwell", SimpleNamespace(
        is_supported=lambda *_: True, block_sparse_attn_ptx_blackwell=forbidden))
    monkeypatch.setattr(h3, "_ptx_blackwell_backward_is_supported", lambda *_: True)
    monkeypatch.setattr(h3, "block_sparse_attn_ptx_blackwell_op", native)
    monkeypatch.setattr(h3, "map_to_index", _indices)
    metadata = h3.MiniMaxH3VSAMetadataBuilder().build(
        current_timestep=0, patch_size=(1, 2, 2), VSA_sparsity=0.0,
        packed_segments=(7, 5, (4, 8, 16)), tile_size=128, device=torch.device("cpu"))
    assert metadata.variable_block_sizes.numel() % 2 == 1
    impl = h3.MiniMaxH3VSAImpl(num_heads=1, head_size=8, causal=False, softmax_scale=8**-0.5)
    leaves = [torch.randn(1, metadata.total_seq_length, 1, 8, requires_grad=True) for _ in range(3)]
    tiled = [impl.tile(leaf, metadata) for leaf in leaves]
    output = impl.postprocess_output(impl.forward(*tiled, None, metadata), metadata)
    output.sum().backward()
    assert len(calls) == 1 and calls[0][-1] == 0
    assert metadata.tile_buf_holder.buffer is None
    for leaf in leaves:
        torch.testing.assert_close(leaf.grad, torch.ones_like(leaf))


def test_native_128_training_rejects_missing_backward(monkeypatch):
    monkeypatch.setattr(h3, "_ptx_blackwell", SimpleNamespace(is_supported=lambda *_: True))
    monkeypatch.setattr(h3, "block_sparse_attn_ptx_blackwell_op", _native)
    monkeypatch.setattr(h3, "_ptx_blackwell_backward_is_supported", lambda *_: False)
    metadata = _metadata(tile_size=128)
    impl = h3.MiniMaxH3VSAImpl(num_heads=1, head_size=8, causal=False, softmax_scale=8**-0.5)
    leaf = torch.randn(1, metadata.total_seq_length, 1, 8, requires_grad=True)
    q, k, v = (impl.tile(leaf, metadata) for _ in range(3))
    with pytest.raises(RuntimeError, match="forward and backward"):
        impl.forward(q, k, v, None, metadata)


@pytest.mark.parametrize("backend_name,expected_dtype", [
    ("VIDEO_SPARSE_ATTN", torch.float32), ("VIDEO_SPARSE_ATTN_H3", torch.bfloat16),
])
def test_training_transport_cast_is_specific_to_h3(monkeypatch, backend_name, expected_dtype):
    import fastvideo.attention.layer as layer
    from fastvideo.forward_context import set_forward_context
    from fastvideo.platforms import AttentionBackendEnum

    seen = []

    def transport(value, **kwargs):
        seen.append(value.dtype)
        return value

    monkeypatch.setattr(layer, "sequence_model_parallel_all_to_all_4D", transport)
    attention = layer.DistributedAttention_VSA.__new__(layer.DistributedAttention_VSA)
    torch.nn.Module.__init__(attention)
    attention.backend = getattr(AttentionBackendEnum, backend_name)
    attention._compile_forward_enabled = True
    attention.attn_impl = SimpleNamespace(
        preprocess_qkv=lambda value, _: value,
        forward=lambda q, k, v, gate, metadata: q + k + v,
        postprocess_output=lambda value, _: value,
    )
    leaves = [torch.randn(1, 3, 2, 4, requires_grad=True) for _ in range(3)]
    with set_forward_context(current_timestep=0, attn_metadata=None):
        output, _ = attention(*leaves, original_seq_len=3)
    assert seen == [expected_dtype, expected_dtype]
    assert output.dtype == expected_dtype
    output.sum().backward()
    for leaf in leaves:
        torch.testing.assert_close(leaf.grad, torch.ones_like(leaf))


def test_compiled_selection_preserves_region_topk():
    metadata = _metadata()
    tiles = metadata.variable_block_sizes.numel()
    scores = torch.randn(1, 2, tiles, tiles)
    expected = h3._build_block_mask(scores, metadata.num_prefix_tiles, metadata.VSA_sparsity,
                                     metadata.exempt, metadata.video_tile_spans, metadata.span_sparsities)
    # CPU eager backend still requires full Dynamo capture; the custom op's
    # concrete CPU implementation executes the same region-selection code.
    compiled = torch.compile(h3.h3_region_topk_mask, backend="eager", fullgraph=True)
    actual = compiled(scores, metadata.block_mask_plan, metadata.VSA_sparsity, metadata.exempt)
    assert torch.equal(actual, expected)


def test_native_128_training_rejects_forced_triton(monkeypatch, env_overrides):
    env_overrides.enter_context(envs.FASTVIDEO_VSA_TRITON.override(True))
    monkeypatch.setattr(h3, "block_sparse_attn_ptx_blackwell_op", _native)
    metadata = _metadata(tile_size=128)
    impl = h3.MiniMaxH3VSAImpl(num_heads=1, head_size=8, causal=False, softmax_scale=8**-0.5)
    leaf = torch.randn(1, metadata.total_seq_length, 1, 8, requires_grad=True)
    q, k, v = (impl.tile(leaf, metadata) for _ in range(3))
    with pytest.raises(RuntimeError, match="Triton was forced"):
        impl.forward(q, k, v, None, metadata)
