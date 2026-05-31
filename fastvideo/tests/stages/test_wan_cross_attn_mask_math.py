"""
Math-only unit test for the cross-attention padding mask used by
batched-CFG in ``WanT2VCrossAttention.forward``.

Goal: prove that running SDPA with a per-sample length-mask on padded
K/V produces the *same* output (at the real Q positions) as running
attention on the un-padded K/V. If this test passes, the SDPA + mask
branch is mathematically correct and any remaining SSIM divergence on
the Wan inference run is somewhere else (plumbing, batch coupling,
etc.).

Runs purely on CPU with random tensors at Wan-ish shapes. No GPU, no
fastvideo install needed beyond torch.
"""
import pytest
import torch


def _python_attn(
    q: torch.Tensor,  # (B, L_q, n, d)
    k: torch.Tensor,  # (B, L_k, n, d)
    v: torch.Tensor,  # (B, L_k, n, d)
) -> torch.Tensor:
    """Reference implementation: plain python attention, no mask,
    no padding. Used as ground truth for the un-padded baseline."""
    q_t = q.transpose(1, 2)  # (B, n, L_q, d)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    d = q_t.shape[-1]
    scale = d**-0.5
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
    weights = torch.softmax(scores.float(), dim=-1).to(q_t.dtype)
    out = torch.matmul(weights, v_t)
    return out.transpose(1, 2).contiguous()  # (B, L_q, n, d)


def _sdpa_mask_branch(
    q: torch.Tensor,  # (B, L_q, n, d) — pre-transpose
    k: torch.Tensor,  # (B, L_k_padded, n, d)
    v: torch.Tensor,  # (B, L_k_padded, n, d)
    context_lens: torch.Tensor,  # (B,)
) -> torch.Tensor:
    """Mirror exactly the branch in WanT2VCrossAttention.forward."""
    kv_seq_len = k.size(1)
    seq_idx = torch.arange(kv_seq_len, device=k.device).unsqueeze(0)
    keep = seq_idx < context_lens.to(k.device).unsqueeze(1)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    attn_mask = keep.unsqueeze(1).unsqueeze(1)
    out = torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask)
    return out.transpose(1, 2).contiguous()


def _make_inputs(B: int, L_q: int, L_real_per_sample: list[int], L_padded: int, n: int, d: int,
                 dtype: torch.dtype, seed: int):
    """Build a batched Q + padded K/V + a parallel list of un-padded
    K/V per sample."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    assert len(L_real_per_sample) == B
    assert all(rl <= L_padded for rl in L_real_per_sample)
    q = torch.randn(B, L_q, n, d, dtype=dtype, generator=g)
    # Build un-padded K/V per sample first.
    unpadded_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
    k_padded = torch.zeros(B, L_padded, n, d, dtype=dtype)
    v_padded = torch.zeros(B, L_padded, n, d, dtype=dtype)
    for b, real_len in enumerate(L_real_per_sample):
        k_real = torch.randn(1, real_len, n, d, dtype=dtype, generator=g)
        v_real = torch.randn(1, real_len, n, d, dtype=dtype, generator=g)
        unpadded_kv.append((k_real, v_real))
        # Pad: positions [real_len:L_padded] stay zero. THEN simulate
        # what the real model does — the text_embedder (an MLP) runs
        # on the padded zero-positions and produces *bias-driven*
        # non-zero values. Replicate that here with a random non-zero
        # vector at padded positions, so the mask actually has to do
        # work (it's not just zeroing zeros).
        bias_noise_k = torch.randn(L_padded - real_len, n, d, dtype=dtype, generator=g) * 0.3
        bias_noise_v = torch.randn(L_padded - real_len, n, d, dtype=dtype, generator=g) * 0.3
        k_padded[b, :real_len] = k_real[0]
        k_padded[b, real_len:] = bias_noise_k
        v_padded[b, :real_len] = v_real[0]
        v_padded[b, real_len:] = bias_noise_v
    context_lens = torch.tensor(L_real_per_sample, dtype=torch.int32)
    return q, k_padded, v_padded, unpadded_kv, context_lens


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("config", [
    # (L_real_per_sample, L_padded, label)
    ([98, 98], 98, "no-padding-both-samples"),
    ([55, 98], 98, "sample0-padded-sample1-full"),
    ([98, 55], 98, "sample0-full-sample1-padded"),
    ([20, 80], 100, "both-padded-different-amounts"),
    ([1, 1], 50, "extreme-padding-both"),
])
def test_sdpa_mask_matches_unpadded_attention(dtype, config):
    L_real_per_sample, L_padded, label = config
    B = len(L_real_per_sample)
    L_q = 64  # small for speed
    n = 12  # Wan T2V 1.3B-ish
    d = 128  # Wan head_dim
    seed = 42

    q, k_padded, v_padded, unpadded_kv, context_lens = _make_inputs(
        B, L_q, L_real_per_sample, L_padded, n, d, dtype, seed)

    # Ground truth: per-sample attention on un-padded K/V.
    out_unpadded_per_sample = []
    for b, (k_real, v_real) in enumerate(unpadded_kv):
        out = _python_attn(q[b:b + 1], k_real, v_real)
        out_unpadded_per_sample.append(out)
    out_unpadded = torch.cat(out_unpadded_per_sample, dim=0)  # (B, L_q, n, d)

    # Candidate: SDPA + mask on padded K/V.
    out_masked = _sdpa_mask_branch(q, k_padded, v_padded, context_lens)

    # Compare per-sample. Tolerances loose for bf16 (FA / SDPA backend
    # numerics), tight for fp32.
    atol, rtol = (1e-2, 1e-2) if dtype is torch.bfloat16 else (1e-4, 1e-4)
    for b in range(B):
        max_abs_diff = (out_masked[b] - out_unpadded[b]).abs().max().item()
        assert torch.allclose(out_masked[b], out_unpadded[b], atol=atol, rtol=rtol), (
            f"[{label}, dtype={dtype}] sample {b} (real_len={L_real_per_sample[b]}): "
            f"max abs diff = {max_abs_diff:.6f}, atol={atol}")


def test_no_mask_path_matches_python_reference():
    """Sanity: when context_lens is None (sequential path), the
    upstream code calls flash-attn directly. We can't test that here
    without the FA wheel, but SDPA without mask should also match the
    python reference on un-padded input."""
    B, L_q, L_k, n, d = 2, 64, 98, 12, 128
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(B, L_q, n, d, dtype=torch.float32, generator=g)
    k = torch.randn(B, L_k, n, d, dtype=torch.float32, generator=g)
    v = torch.randn(B, L_k, n, d, dtype=torch.float32, generator=g)
    ref = _python_attn(q, k, v)
    sdpa_out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).contiguous()
    max_abs_diff = (ref - sdpa_out).abs().max().item()
    assert torch.allclose(ref, sdpa_out, atol=1e-4, rtol=1e-4), f"max abs diff = {max_abs_diff:.6f}"


if __name__ == "__main__":
    # Allow running as a script: `python test_wan_cross_attn_mask_math.py`
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
