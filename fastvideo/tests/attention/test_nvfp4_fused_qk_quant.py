# SPDX-License-Identifier: Apache-2.0
"""Fused NVFP4 Q/K preconditioning + quantization vs the reference path.

The fused kernel (``fastvideo.attention.utils.nvfp4_fused_qk_quant``) must produce the same FP4 codes, the same
scale-factor bytes in FlashAttention-4's layout, and therefore the same FA4 NVFP4 attention output as
``precondition_qk`` followed by ``_nvfp4_quantize_for_fa4``. GPU tests need Blackwell (sm_100a/sm_103a), flashinfer
and the FA4 NVFP4 kernel; the selection tests run anywhere.
"""
import inspect
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

import fastvideo.envs as envs  # noqa: E402
from fastvideo.attention.utils.nvfp4_qk_precondition import parse_qk_transform, precondition_qk  # noqa: E402

flash_attn_module = pytest.importorskip(
    "fastvideo.attention.backends.flash_attn",
    reason="no usable flash-attention package installed",
    exc_type=ImportError,
)
fused = pytest.importorskip("fastvideo.attention.utils.nvfp4_fused_qk_quant")

TRANSFORMS = ["none", "smooth_k", "hadamard", "smooth_k+hadamard"]


def _blackwell_fa4_fp4() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in [(10, 0), (10, 3)]:
        return False
    try:
        import flashinfer.quantization  # noqa: F401
    except ImportError:
        return False
    return flash_attn_module._FA4_FP4_AVAILABLE and fused.fused_qk_quant_available()


requires_blackwell = pytest.mark.skipif(not _blackwell_fa4_fp4(),
                                        reason="needs Blackwell, flashinfer, Triton and FA4 NVFP4")


@pytest.fixture
def tf32_matmul(request):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = request.param
    yield request.param
    torch.backends.cuda.matmul.allow_tf32 = previous


def _inputs(batch, seqlen, heads, dim, *, strided=False, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if strided:  # Q/K/V sliced out of one packed buffer, as a fused QKV projection produces them
        qkv = torch.randn(batch, seqlen, 3, heads, dim, device="cuda", dtype=torch.bfloat16, generator=gen)
        q, k, v = qkv.unbind(2)
    else:
        q, k, v = (torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16, generator=gen)
                   for _ in range(3))
    # A per-channel K offset (what smooth_k removes) and a few outlier channels (what the Hadamard spreads).
    k = k.clone()
    k[..., :4] *= 8.0
    k += 4.0 * torch.randn(batch, 1, heads, dim, device="cuda", dtype=torch.bfloat16, generator=gen)
    if strided:
        buffer = torch.empty(batch, seqlen, 2, heads, dim, device="cuda", dtype=torch.bfloat16)
        buffer[:, :, 1] = k
        k = buffer[:, :, 1]
    return q, k, v


def _reference(q, k, transform):
    smooth_k, hadamard = parse_qk_transform(transform)
    q, k = precondition_qk(q, k, smooth_k=smooth_k, hadamard=hadamard)
    return (*flash_attn_module._nvfp4_quantize_for_fa4(q), *flash_attn_module._nvfp4_quantize_for_fa4(k))


def _fused(q, k, transform):
    smooth_k, hadamard = parse_qk_transform(transform)
    return fused.nvfp4_fused_qk_quantize_fa4(q, k, smooth_k=smooth_k, hadamard=hadamard)


def _codes(fp4):
    data = fp4.view(torch.uint8)
    return torch.stack([data & 0xF, data >> 4], dim=-1)


def _attention(q4, q_sf, k4, k_sf, v):
    out = flash_attn_module.flash_attn_fp4_func(q4[:, :v.shape[1]], k4[:, :v.shape[1]], v, q_sf, k_sf,
                                                softmax_scale=v.shape[-1]**-0.5, causal=False)
    return out[0] if isinstance(out, tuple) else out


SHAPES = [
    pytest.param((1, 1000, 4, 128, False), id="odd_seqlen"),
    pytest.param((2, 300, 2, 64, False), id="batch2_d64"),
    pytest.param((1, 777, 3, 128, True), id="strided"),
]


@requires_blackwell
@pytest.mark.parametrize("tf32_matmul", [True], indirect=True)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_fused_matches_reference_bit_exactly_under_tf32(transform, shape, tf32_matmul) -> None:
    """Under TF32 matmuls (NVIDIA PyTorch containers) the fused path is bit-identical to the reference."""
    batch, seqlen, heads, dim, strided = shape
    q, k, v = _inputs(batch, seqlen, heads, dim, strided=strided)
    ref, out = _reference(q, k, transform), _fused(q, k, transform)
    for r, o in zip(ref, out, strict=True):
        assert r.shape == o.shape and r.dtype == o.dtype and r.stride() == o.stride()
        assert torch.equal(r.view(torch.uint8), o.view(torch.uint8))
    for fp4, length in ((out[0], seqlen), (out[2], seqlen)):
        assert not _codes(fp4)[:, length:].any(), "rows padding seqlen to 128 must hold zero codes"
    assert torch.equal(_attention(*out, v), _attention(*ref, v))


@requires_blackwell
@pytest.mark.parametrize("tf32_matmul", [False], indirect=True)
@pytest.mark.parametrize("transform", ["hadamard", "smooth_k+hadamard"])
def test_fused_matches_reference_under_fp32(transform, tf32_matmul) -> None:
    """With TF32 off only fp32 summation order differs from cuBLAS's fp32 GEMM: codes/scales agree to >= 99.99%."""
    q, k, v = _inputs(1, 4000, 8, 128, seed=1)
    ref, out = _reference(q, k, transform), _fused(q, k, transform)
    for r, o in ((ref[0], out[0]), (ref[2], out[2])):
        assert (_codes(r) == _codes(o)).float().mean().item() >= 0.9999
    for r, o in ((ref[1], out[1]), (ref[3], out[3])):
        assert r.stride() == o.stride()
        assert (r == o).float().mean().item() >= 0.9999


@requires_blackwell
@pytest.mark.parametrize("tf32_matmul", [False], indirect=True)
def test_hadamard_without_tf32_takes_the_reference_path(tf32_matmul) -> None:
    q, k, _ = _inputs(1, 256, 4, 128, seed=4)
    assert not fused.fused_qk_quant_supported(q, k, hadamard=True)
    assert fused.fused_qk_quant_supported(q, k, hadamard=False)


@requires_blackwell
@pytest.mark.parametrize("tf32_matmul", [True], indirect=True)
def test_forward_nvfp4_uses_fused_path_and_compiles_fullgraph(tf32_matmul) -> None:
    impl = flash_attn_module.FlashAttentionImpl(num_heads=4, head_size=128, causal=False, softmax_scale=128**-0.5,
                                                num_kv_heads=4, nvfp4_fa4=True,
                                                nvfp4_qk_transform="smooth_k+hadamard")
    assert impl.nvfp4_fused_qk_quant
    q, k, v = _inputs(1, 1000, 4, 128, seed=2)
    with patch.object(flash_attn_module, "precondition_qk", side_effect=AssertionError("unfused path taken")):
        eager = impl._forward_nvfp4(q, k, v)
        compiled = torch.compile(impl._forward_nvfp4, fullgraph=True)(q, k, v)
    assert torch.equal(eager, compiled)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU (about 4 GiB)")
@pytest.mark.parametrize("seqlen", [74898, 74899])
def test_token_mean_matches_reference_across_the_64bit_index_split(seqlen) -> None:
    """At 56 x 128 channels the fp32 copy of K needs 64-bit indexing from 74,899 tokens on (768p/15 s H3 has 113k),
    where torch reduces it in two halves; the fused path's mean must keep the reference's bits on both sides."""
    gen = torch.Generator(device="cuda").manual_seed(3)
    k = (torch.randn(1, seqlen, 56, 128, device="cuda", generator=gen) * 2 + 1).to(torch.bfloat16)
    assert len(fused._fp32_reduction_chunks(k)) == (1 if seqlen == 74898 else 2)
    assert torch.equal(fused.reference_token_mean(k), k.float().mean(dim=1))



@pytest.mark.parametrize(("shape", "reproducible"), [
    ((1, 113000, 56, 128), True),
    ((2, 1000, 56, 128), True),
    ((2, 40000, 56, 128), False),
    ((2, 80000, 56, 128), False),
])
def test_batched_inputs_that_split_the_token_mean_take_the_unfused_path(shape, reproducible) -> None:
    k = torch.empty(shape, dtype=torch.bfloat16, device="meta")
    assert fused._token_mean_reproducible(k) is reproducible
    if not reproducible:
        with pytest.raises(ValueError, match="not reproduced"):
            fused._fp32_reduction_chunks(k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_batched_token_mean_matches_reference() -> None:
    gen = torch.Generator(device="cuda").manual_seed(5)
    k = (torch.randn(2, 4096, 56, 128, device="cuda", generator=gen) * 2 + 1).to(torch.bfloat16)
    assert torch.equal(fused.reference_token_mean(k), k.float().mean(dim=1))

@pytest.fixture
def fa4_runtime_available():
    with (
        patch.object(flash_attn_module, "_FA4_FP4_AVAILABLE", True),
        patch("torch.cuda.get_device_capability", return_value=(10, 0)),
    ):
        yield


def _impl(**extra):
    return flash_attn_module.FlashAttentionImpl(num_heads=2, head_size=64, causal=False, softmax_scale=0.125,
                                                num_kv_heads=2, nvfp4_fa4=True, **extra)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_fused_path_selected_only_with_a_transform(transform, env_overrides, fa4_runtime_available) -> None:
    env_overrides.enter_context(envs.FASTVIDEO_NVFP4_FA4_FUSED_QK_QUANT.override(None))
    with patch.object(flash_attn_module, "fused_qk_quant_available", return_value=True):
        assert _impl(nvfp4_qk_transform=transform).nvfp4_fused_qk_quant == (transform != "none")


def test_fused_path_env_opt_out(env_overrides, fa4_runtime_available) -> None:
    env_overrides.enter_context(envs.FASTVIDEO_NVFP4_FA4_FUSED_QK_QUANT.override(False))
    with patch.object(flash_attn_module, "fused_qk_quant_available", return_value=True):
        assert _impl(nvfp4_qk_transform="smooth_k+hadamard").nvfp4_fused_qk_quant is False


def test_fused_path_off_for_non_default_flashinfer_recipe(env_overrides) -> None:
    env_overrides.enter_context(envs.override_external("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "1"))
    assert fused.fused_qk_quant_available() is False


def test_fp4_availability_tracks_the_installed_kernel() -> None:
    """flash_attn.cute importing is not enough to run the FP4 path.

    The repo's flash-attn-4 pin is upstream FA4, whose kernel rejects the mSFQ/mSFK scale-factor kwargs the FP4
    forward passes; only the flash-attention-fp4 fork accepts them. A build with the upstream kernel must not
    report the FP4 path available (and so must skip the GPU tests above), or its unit lane dies with TypeError.
    """
    flash_attn_cute = pytest.importorskip("fastvideo.attention.utils.flash_attn_cute",
                                          reason="no usable flash-attention package installed")
    kernel_accepts_fp4 = {"mSFQ", "mSFK"} <= set(inspect.signature(flash_attn_cute._flash_attn_fwd).parameters)
    assert flash_attn_cute.fp4_forward_available() is kernel_accepts_fp4
    assert flash_attn_module._FA4_FP4_AVAILABLE is kernel_accepts_fp4
