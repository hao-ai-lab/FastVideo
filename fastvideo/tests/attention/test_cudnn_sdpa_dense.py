# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3's opt-in cuDNN SDPA route for dense self-attention.

The ATen cuDNN attention op carries no shape metadata and always reads its
inputs as ``[B, H, S, D]``; a contiguous ``[B, S, H, D]`` tensor is silently
misread. These tests pin the BHSD-strided-view contract and the opt-in flag.
"""

import math

import pytest
import torch

from fastvideo import envs
from fastvideo.attention.utils.cudnn_sdpa import cudnn_sdpa_bshd, cudnn_sdpa_supported

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuDNN SDPA requires CUDA")


def _qkv(seq_len: int, *, heads: int = 8, head_dim: int = 64, dtype=torch.bfloat16):
    generator = torch.Generator(device="cuda").manual_seed(0)
    return [
        torch.randn(1, seq_len, heads, head_dim, device="cuda", dtype=dtype, generator=generator) for _ in range(3)
    ]


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def test_cudnn_route_presents_bhsd_strided_views():
    """Contiguous BSHD is silently misread; the BHSD view is the supported form."""
    query, key, value = _qkv(257)
    scale = 1.0 / math.sqrt(query.shape[-1])
    with torch.no_grad():
        output = cudnn_sdpa_bshd(query, key, value, scale=scale)
        # The trap: hand the op the caller's contiguous BSHD tensors directly.
        misread = torch.ops.aten._scaled_dot_product_cudnn_attention.default(
            query, key, value, None, False, 0.0, False, False, scale=scale)[0]

    # Same shape survives the misread, which is why it is silent.
    assert misread.shape == output.shape
    assert _rel_l2(misread, output) > 1.0
    assert output.is_contiguous()
    assert output.dtype is torch.bfloat16
    assert torch.isfinite(output).all()


def test_cudnn_route_is_bit_identical_to_cudnn_dispatched_sdpa():
    query, key, value = _qkv(2048)
    scale = 1.0 / math.sqrt(query.shape[-1])
    # Pin the reference to cuDNN: enabling it only makes it eligible, and the dispatcher
    # prefers another backend on some GPUs (GB10), which is not bit-identical.
    with torch.no_grad(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
        reference = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scale=scale).transpose(1, 2)

    with torch.no_grad():
        output = cudnn_sdpa_bshd(query, key, value, scale=scale)
    assert torch.equal(output, reference.contiguous())


def test_cudnn_route_tracks_eager_math_sdpa():
    """Same precision contract (bf16 in, fp32 softmax, bf16 out), different reduction order."""
    query, key, value = _qkv(257)
    scale = 1.0 / math.sqrt(query.shape[-1])
    with torch.no_grad():
        output = cudnn_sdpa_bshd(query, key, value, scale=scale)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            reference = torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scale=scale).transpose(1, 2)
    assert _rel_l2(output, reference) < 5e-3
    assert (output.float() - reference.float()).abs().max().item() < 0.02


def test_cudnn_route_guard_rejects_training_unsupported_dtypes_and_shapes():
    query, key, value = _qkv(64, heads=4, head_dim=128)
    with torch.no_grad():
        assert cudnn_sdpa_supported(query, key, value)
        assert not cudnn_sdpa_supported(query.float(), key.float(), value.float())
        assert not cudnn_sdpa_supported(query.transpose(1, 2).transpose(1, 2).transpose(1, 2), key, value)
        wide, _, _ = _qkv(64, heads=2, head_dim=256)
        assert not cudnn_sdpa_supported(wide, wide, wide)
    # A grad-enabled training call stays on the FlashAttention path.
    assert torch.is_grad_enabled()
    assert not cudnn_sdpa_supported(query, key, value)


def test_dense_attention_env_default_is_flash_and_accepts_cudnn():
    assert envs.FASTVIDEO_MINIMAX_H3_DENSE_ATTN.get() == "flash"
    with envs.FASTVIDEO_MINIMAX_H3_DENSE_ATTN.override("cudnn"):
        assert envs.FASTVIDEO_MINIMAX_H3_DENSE_ATTN.get() == "cudnn"
    with pytest.raises(ValueError):
        with envs.FASTVIDEO_MINIMAX_H3_DENSE_ATTN.override("cutlass"):
            envs.FASTVIDEO_MINIMAX_H3_DENSE_ATTN.get()


def test_flash_attn_impl_routes_dense_calls_to_cudnn():
    pytest.importorskip("flash_attn")
    from fastvideo.attention.backends.flash_attn import FlashAttentionImpl

    query, key, value = _qkv(257)
    scale = 1.0 / math.sqrt(query.shape[-1])
    impl = FlashAttentionImpl(num_heads=query.shape[2],
                              head_size=query.shape[3],
                              causal=False,
                              softmax_scale=scale,
                              cudnn_sdpa=True)
    assert impl.cudnn_sdpa
    with torch.no_grad():
        routed = impl.forward(query, key, value, None)
        expected = cudnn_sdpa_bshd(query, key, value, scale=scale)
    assert torch.equal(routed, expected)

    # The cuDNN opt-in wins over FA4 packed-varlen when both are configured.
    packed = FlashAttentionImpl(num_heads=query.shape[2],
                                head_size=query.shape[3],
                                causal=False,
                                softmax_scale=scale,
                                cudnn_sdpa=True,
                                fa4_packed_varlen=True)
    with torch.no_grad():
        assert torch.equal(packed.forward(query, key, value, None), expected)

    # Default construction keeps the existing FlashAttention behavior.
    default = FlashAttentionImpl(num_heads=query.shape[2],
                                 head_size=query.shape[3],
                                 causal=False,
                                 softmax_scale=scale)
    assert not default.cudnn_sdpa


def test_flash_attn_impl_rejects_cudnn_and_nvfp4_together():
    pytest.importorskip("flash_attn")
    from fastvideo.attention.backends.flash_attn import FlashAttentionImpl

    with pytest.raises(ValueError, match="cudnn_sdpa and nvfp4_fa4"):
        FlashAttentionImpl(num_heads=4, head_size=64, causal=False, softmax_scale=0.125, cudnn_sdpa=True,
                           nvfp4_fa4=True)
