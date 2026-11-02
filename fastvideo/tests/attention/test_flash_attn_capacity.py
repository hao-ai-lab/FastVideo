# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch proof for fixed-capacity FA4 storage and undefined tails."""

import pytest
import torch

from fastvideo.attention.backends.abstract import PackedAttentionMetadata, PackedSequenceMetadata

flash = pytest.importorskip("fastvideo.attention.backends.flash_attn", exc_type=ImportError)


@pytest.mark.parametrize("real_tokens", [3, 8])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_fa4_capacity_keeps_shape_and_discards_undefined_tail(monkeypatch, real_tokens, requires_grad):
    capacity = 8
    calls = []

    class Kernel(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v):
            out = q + k + v
            out[real_tokens:] = float("nan")
            return out

        @staticmethod
        def backward(ctx, gradient):
            result = gradient.clone()
            result[real_tokens:] = float("nan")
            return result, result, result

    def varlen(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
        calls.append((q.shape, cu_q.tolist(), cu_k.tolist(), max_q, max_k))
        return Kernel.apply(q, k, v)

    monkeypatch.setattr(flash, "fa_version", "4")
    monkeypatch.setattr(flash, "flash_attn_varlen_func_compilable", varlen)
    impl = flash.FlashAttentionImpl(num_heads=2, head_size=8, causal=False, softmax_scale=8**-0.5)
    metadata = PackedAttentionMetadata(current_timestep=0, packed_sequence=PackedSequenceMetadata.from_lengths(
        (real_tokens,), device="cpu", pad_tokens=capacity-real_tokens, sequence_capacity=capacity))
    leaves = [torch.randn(1, capacity, 2, 8, requires_grad=requires_grad) for _ in range(3)]
    with torch.set_grad_enabled(requires_grad):
        output = impl._forward_impl(*leaves, metadata)
    assert calls == [(torch.Size([capacity, 2, 8]), [0, real_tokens], [0, real_tokens], capacity, capacity)]
    torch.testing.assert_close(output[:, :real_tokens], sum(leaves)[:, :real_tokens])
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[:, real_tokens:]) == 0
    if requires_grad:
        output.sum().backward()
        for leaf in leaves:
            assert torch.isfinite(leaf.grad).all()
            assert torch.count_nonzero(leaf.grad[:, real_tokens:]) == 0
            torch.testing.assert_close(leaf.grad[:, :real_tokens], torch.ones_like(leaf.grad[:, :real_tokens]))
