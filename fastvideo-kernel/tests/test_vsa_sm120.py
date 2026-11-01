# SPDX-License-Identifier: Apache-2.0
"""SM120-only forward tests; no model checkpoint or distributed runtime needed."""

import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Requires an SM120 CUDA GPU",
)


def fixture(batch=1, heads=3, query_tiles=3, kv_tiles=5):
    torch.manual_seed(123)
    q = torch.randn(batch, heads, query_tiles * 64, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, heads, kv_tiles * 64, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sizes = torch.tensor(([1, 63, 64, 17, 64] * kv_tiles)[:kv_tiles], device="cuda", dtype=torch.int32)
    valid = torch.arange(kv_tiles * 64, device="cuda") % 64 < sizes.repeat_interleave(64)
    k[:, :, ~valid] = 0
    v[:, :, ~valid] = 0
    indices = torch.full((batch, heads, query_tiles, kv_tiles), -1, device="cuda", dtype=torch.int32)
    counts = torch.empty((batch, heads, query_tiles), device="cuda", dtype=torch.int32)
    mask = torch.zeros_like(indices, dtype=torch.bool)
    for b in range(batch):
        for h in range(heads):
            for row in range(query_tiles):
                count = (b + h + row) % (kv_tiles + 1)
                selected = torch.arange(kv_tiles - 1, kv_tiles - count - 1, -1, device="cuda")
                indices[b, h, row, :count] = selected.int()
                counts[b, h, row] = count
                mask[b, h, row, selected] = True
    return q, k, v, indices, counts, sizes, mask


def dense_reference(q, k, v, sizes, mask):
    valid = torch.arange(k.shape[2], device=k.device) % 64 < sizes.repeat_interleave(64)
    allowed = mask.repeat_interleave(64, -2).repeat_interleave(64, -1) & valid
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    scores.masked_fill_(~allowed, -float("inf"))
    lse = torch.logsumexp(scores, -1) / math.log(2)
    probabilities = torch.softmax(scores, -1).nan_to_num()
    values = v.float().masked_fill(~valid[None, None, :, None], 0)
    return (probabilities @ values).to(q.dtype), lse


@pytest.mark.parametrize("batch,heads,query_tiles,kv_tiles", [(1, 1, 1, 1), (2, 3, 3, 5), (1, 28, 2, 3), (1, 56, 2, 3)])
@torch.no_grad()
def test_native_and_independent_reference(batch, heads, query_tiles, kv_tiles):
    from fastvideo_kernel.block_sparse_attn import block_sparse_attn_triton
    from fastvideo_kernel.block_sparse_attn_sm120 import block_sparse_attn_sm120

    q, k, v, indices, counts, sizes, mask = fixture(batch, heads, query_tiles, kv_tiles)
    inputs = (q, k, v, indices, counts, sizes)
    saved = [tensor.clone() for tensor in inputs]
    actual = block_sparse_attn_sm120(*inputs)
    native = block_sparse_attn_triton(*inputs)
    independent = dense_reference(q, k, v, sizes, mask)
    for expected in (native, independent):
        torch.testing.assert_close(actual[0], expected[0], atol=0.02, rtol=0.02)
        torch.testing.assert_close(actual[1], expected[1], atol=0.005, rtol=0.001)
    repeated = block_sparse_attn_sm120(*inputs)
    assert all(torch.equal(a, b) for a, b in zip(actual, repeated))
    assert all(torch.equal(a, b) for a, b in zip(inputs, saved))
    # A smaller selected-list capacity and int64 metadata must retain ordering.
    capacity = max(1, int(counts.max()))
    compact = block_sparse_attn_sm120(q, k, v, indices[..., :capacity].contiguous().long(), counts.long(), sizes.long())
    assert all(torch.equal(a, b) for a, b in zip(actual, compact))


@torch.no_grad()
def test_empty_zero_valid_tiles_and_poisoned_value_padding():
    from fastvideo_kernel.block_sparse_attn_sm120 import block_sparse_attn_sm120

    q, k, v, indices, counts, sizes, mask = fixture(query_tiles=3, kv_tiles=3)
    sizes[0] = 0
    counts.fill_(3)
    indices.copy_(torch.arange(3, device="cuda").expand_as(indices))
    mask.fill_(True)
    counts[:, :, 0] = 0
    mask[:, :, 0] = False
    valid = torch.arange(k.shape[2], device="cuda") % 64 < sizes.repeat_interleave(64)
    v[:, :, ~valid] = float("nan")
    actual = block_sparse_attn_sm120(q, k, v, indices, counts, sizes)
    expected = dense_reference(q, k, v, sizes, mask)
    torch.testing.assert_close(actual[0], expected[0], atol=0.02, rtol=0.02)
    torch.testing.assert_close(actual[1], expected[1], atol=0.005, rtol=0.001)
    sizes.zero_()
    out, lse = block_sparse_attn_sm120(q, k, v, indices, counts, sizes)
    assert torch.count_nonzero(out) == 0
    assert torch.isneginf(lse).all()


def test_contract_errors(monkeypatch):
    from fastvideo_kernel.block_sparse_attn_sm120 import block_sparse_attn_sm120

    q, k, v, indices, counts, sizes, _ = fixture()
    with pytest.raises(ValueError, match="contiguous BF16"):
        block_sparse_attn_sm120(q.float(), k, v, indices, counts, sizes)
    with pytest.raises(ValueError, match="contiguous BF16"):
        block_sparse_attn_sm120(q.transpose(2, 3), k, v, indices, counts, sizes)
    with pytest.raises(ValueError, match="metadata"):
        block_sparse_attn_sm120(q, k, v, indices, counts, sizes[:-1])
    with pytest.raises(ValueError, match="int32/int64"):
        block_sparse_attn_sm120(q, k, v, indices.float(), counts, sizes)
    with pytest.raises(RuntimeError, match="forward-only"):
        block_sparse_attn_sm120(q.requires_grad_(), k, v, indices, counts, sizes)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    with pytest.raises(RuntimeError, match="compute capability 12.0"):
        block_sparse_attn_sm120(q, k, v, indices, counts, sizes)


@pytest.mark.parametrize("gated", [False, True])
@torch.no_grad()
def test_h3_dispatch_preserves_mask_and_compression(gated, monkeypatch):
    import fastvideo.envs as envs
    import fastvideo.attention.backends.video_sparse_attn_h3 as backend

    q, k, v, _, _, sizes, _ = fixture(query_tiles=3, kv_tiles=3)
    metadata = SimpleNamespace(
        tile_elems=64, variable_block_sizes=sizes, dense_layers=(), VSA_sparsity=0.5,
        num_prefix_tiles=1, exempt=True, video_tile_spans=((1, 3),), span_sparsities=(0.5,),
    )
    query, key, value = (tensor.transpose(1, 2).contiguous() for tensor in (q, k, v))
    gate = torch.randn_like(query) * 0.1 if gated else None
    monkeypatch.setattr(backend, "probe_enabled", lambda: None)
    with envs.FASTVIDEO_H3_VSA_SM120.override(False), envs.FASTVIDEO_VSA_PTX_BLACKWELL.override(False):
        native = backend.MiniMaxH3VSAImpl(3, 128, False, 128**-0.5)
        expected = native.forward(query, key, value, gate, metadata)
        with envs.FASTVIDEO_H3_VSA_SM120.override(True):
            candidate = backend.MiniMaxH3VSAImpl(3, 128, False, 128**-0.5)
            actual = candidate.forward(query, key, value, gate, metadata)
            with pytest.raises(RuntimeError, match="eager execution"):
                candidate.prepare_for_compile(query.device)
            metadata.tile_elems = 128
            with pytest.raises(RuntimeError, match="tile-64"):
                candidate.forward(query, key, value, gate, metadata)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


def test_h3_conflicting_routes():
    import fastvideo.envs as envs
    from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAImpl

    with envs.FASTVIDEO_H3_VSA_SM120.override(True):
        with pytest.raises(ValueError, match="head dimension 128"):
            MiniMaxH3VSAImpl(3, 64, False, 64**-0.5)
        with envs.FASTVIDEO_VSA_PTX_BLACKWELL.override(True):
            with pytest.raises(ValueError, match="another fine-attention override"):
                MiniMaxH3VSAImpl(3, 128, False, 128**-0.5)
