"""Test NVFP4 FA4 (FP4 quantized QK flash attention) vs BF16 FA4.

Requires: Blackwell GPU (sm100a/sm103a), flash-attention-fp4, flashinfer, cutlass-dsl.
Run: CUTE_DSL_ENABLE_TVM_FFI=1 pytest tests/local_tests/test_nvfp4_fa4.py -v -s
"""

import os
import torch
import pytest

os.environ["CUTE_DSL_ENABLE_TVM_FFI"] = "1"

# Warm up flashinfer JIT before any FA4 imports
from flashinfer.quantization import nvfp4_quantize, SfLayout
_warmup = nvfp4_quantize(
    torch.randn(4, 128, device="cuda", dtype=torch.bfloat16),
    torch.ones(1, device="cuda", dtype=torch.float32),
    sfLayout=SfLayout.layout_128x4, do_shuffle=False,
)
del _warmup


def _make_impl(nheads, headdim, nvfp4=False):
    if nvfp4:
        os.environ["FASTVIDEO_NVFP4_FA4"] = "1"
    else:
        os.environ.pop("FASTVIDEO_NVFP4_FA4", None)
    from fastvideo.attention.backends.flash_attn import FlashAttentionImpl
    return FlashAttentionImpl(
        num_heads=nheads, head_size=headdim, causal=False,
        softmax_scale=headdim ** -0.5,
    )


def _cuda_timer(fn, warmup=5, iters=20):
    """Time a function using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


@pytest.mark.skipif(
    torch.cuda.get_device_capability() not in [(10, 0), (10, 3)],
    reason="Requires Blackwell GPU (sm100a or sm103a)"
)
class TestNVFP4FA4:

    # Wan2.1-T2V-1.3B: num_attention_heads=12, attention_head_dim=128
    MODEL_NHEADS = 12
    MODEL_HEADDIM = 128
    MODEL_SEQLEN = 32760  # 480x832 video, 81 frames

    def test_quantize_fn_shapes(self):
        """_nvfp4_quantize_for_fa4 produces correct shapes and strides."""
        from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4
        batch, seqlen, nheads, headdim = 1, 256, self.MODEL_NHEADS, self.MODEL_HEADDIM
        t = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)
        fp4, sf = _nvfp4_quantize_for_fa4(t)

        assert fp4.dtype == torch.float4_e2m1fn_x2
        assert fp4.shape == (batch, seqlen, nheads, headdim // 2)
        rest_m = seqlen // 128
        rest_k = (headdim // 16) // 4
        assert sf.shape == (32, 4, rest_m, 4, rest_k, nheads, batch)
        assert sf.dtype == torch.uint8
        assert sf.stride()[3] == 1, f"SF stride[3] must be 1, got {sf.stride()[3]}"

    def test_output_shape_and_dtype(self):
        """NVFP4 FA4 output has correct shape and dtype."""
        batch, nheads, headdim = 1, self.MODEL_NHEADS, self.MODEL_HEADDIM
        seqlen = 256
        q = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)

        impl = _make_impl(nheads, headdim, nvfp4=True)
        out = impl.forward(q, k, v, attn_metadata=None)
        assert out.shape == (batch, seqlen, nheads, headdim)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any(), "Output contains NaN"

    def test_cross_attention(self):
        """NVFP4 FA4 handles cross-attention (seqlen_q != seqlen_k)."""
        nheads, headdim = self.MODEL_NHEADS, self.MODEL_HEADDIM
        q = torch.randn(1, self.MODEL_SEQLEN, nheads, headdim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 512, nheads, headdim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 512, nheads, headdim, device="cuda", dtype=torch.bfloat16)

        impl = _make_impl(nheads, headdim, nvfp4=True)
        out = impl.forward(q, k, v, attn_metadata=None)
        assert out.shape == (1, self.MODEL_SEQLEN, nheads, headdim)
        assert not torch.isnan(out).any()

    def test_vs_bf16_accuracy(self):
        """NVFP4 FA4 output is close to BF16 FA4."""
        nheads, headdim = self.MODEL_NHEADS, self.MODEL_HEADDIM
        from flash_attn.cute.interface import flash_attn_func as fa4_func
        torch.manual_seed(42)
        q = torch.randn(1, self.MODEL_SEQLEN, nheads, headdim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, self.MODEL_SEQLEN, nheads, headdim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, self.MODEL_SEQLEN, nheads, headdim, device="cuda", dtype=torch.bfloat16)

        fp4_out = _make_impl(nheads, headdim, nvfp4=True).forward(q, k, v, attn_metadata=None)
        bf16_out = fa4_func(q, k, v, softmax_scale=headdim**-0.5, causal=False)
        if isinstance(bf16_out, tuple):
            bf16_out = bf16_out[0]

        cos = torch.nn.functional.cosine_similarity(
            fp4_out.flatten().unsqueeze(0).float(),
            bf16_out.flatten().unsqueeze(0).float()
        ).item()
        print(f"cos={cos:.4f}")
        assert cos > 0.95, f"Cosine similarity too low: {cos:.4f}"

    def test_attention_speedup(self):
        """NVFP4 FA4 kernel is faster than BF16 FA4 (excluding quantization overhead)."""
        from flash_attn.cute.interface import flash_attn_func as fa4_func
        from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4

        batch, seqlen, nheads, headdim = 1, self.MODEL_SEQLEN, self.MODEL_NHEADS, self.MODEL_HEADDIM
        q = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)

        # BF16 kernel timing
        bf16_ms = _cuda_timer(lambda: fa4_func(q, k, v, softmax_scale=headdim**-0.5, causal=False))

        # FP4 kernel timing (pre-quantized, pad V to match K)
        q_fp4, q_sf = _nvfp4_quantize_for_fa4(q)
        k_fp4, k_sf = _nvfp4_quantize_for_fa4(k)
        seqlen_padded = ((seqlen + 127) // 128) * 128
        v_padded = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, seqlen_padded - seqlen)) if seqlen_padded != seqlen else v
        fp4_ms = _cuda_timer(lambda: fa4_func(
            q_fp4, k_fp4, v_padded, softmax_scale=headdim**-0.5, causal=False, mSFQ=q_sf, mSFK=k_sf,
        ))

        speedup = bf16_ms / fp4_ms
        print(f"BF16={bf16_ms:.2f}ms FP4={fp4_ms:.2f}ms speedup={speedup:.2f}x")
        assert speedup > 1.0, f"Expected speedup > 1.0, got {speedup:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
