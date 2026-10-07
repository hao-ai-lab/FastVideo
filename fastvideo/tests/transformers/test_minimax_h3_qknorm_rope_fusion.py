# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from fastvideo.models.dits.minimax_h3 import MiniMaxH3Attention
from fastvideo.models.dits.minimax_h3_fusions.qknorm_rope import (
    HAVE_TRITON,
    fused_qknorm_rope,
)


def _rotary_tables(
    seq_len: int,
    rotary_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    angles = torch.randn(seq_len, rotary_dim // 2, dtype=torch.float32, device=device)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _eager_qknorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    normalized = F.rms_norm(x, (x.shape[-1], ), weight, eps)
    return MiniMaxH3Attention._apply_rotary_emb(normalized, (cos, sin))


def test_fused_qknorm_rope_rejects_invalid_rotary_dim() -> None:
    x = torch.randn(2, 3, 4, 128)
    weight = torch.ones(128)

    cos, sin = _rotary_tables(3, 130, dtype=x.dtype, device=x.device)
    with pytest.raises(ValueError, match="rotary_dim must not exceed head_dim"):
        fused_qknorm_rope(x, weight, cos, sin, 1e-6)

    cos = torch.randn(3, 95)
    sin = torch.randn_like(cos)
    with pytest.raises(ValueError, match="rotary_dim must be even"):
        fused_qknorm_rope(x, weight, cos, sin, 1e-6)


def test_fused_qknorm_rope_rejects_shape_mismatches() -> None:
    x = torch.randn(2, 3, 4, 128)
    weight = torch.ones(128)
    cos, sin = _rotary_tables(3, 96, dtype=x.dtype, device=x.device)

    with pytest.raises(ValueError, match="x must have shape"):
        fused_qknorm_rope(x[0], weight, cos, sin, 1e-6)
    with pytest.raises(ValueError, match="weight must have shape"):
        fused_qknorm_rope(x, weight[:-1], cos, sin, 1e-6)
    with pytest.raises(ValueError, match="sequence length"):
        fused_qknorm_rope(x, weight, cos[:-1], sin[:-1], 1e-6)
    with pytest.raises(ValueError, match="sin must match cos shape"):
        fused_qknorm_rope(x, weight, cos, sin[:, :-2], 1e-6)


@pytest.mark.parametrize("noncontiguous_input", ["weight", "cos", "sin"])
def test_fused_qknorm_rope_rejects_noncontiguous_linear_inputs(noncontiguous_input: str) -> None:
    x = torch.randn(2, 3, 4, 128)
    weight = torch.ones(256)[::2]
    cos = torch.randn(3, 192)[:, ::2]
    sin = torch.randn(3, 192)[:, ::2]
    assert not weight.is_contiguous()
    assert not cos.is_contiguous()
    assert not sin.is_contiguous()

    if noncontiguous_input != "weight":
        weight = weight.contiguous()
    if noncontiguous_input != "cos":
        cos = cos.contiguous()
    if noncontiguous_input != "sin":
        sin = sin.contiguous()

    with pytest.raises(ValueError, match="must be contiguous"):
        fused_qknorm_rope(x, weight, cos, sin, 1e-6)


def test_fused_qknorm_rope_requires_matching_precast_dtype() -> None:
    x = torch.randn(2, 3, 4, 128, dtype=torch.float32)
    weight = torch.ones(128, dtype=torch.float32)
    cos, sin = _rotary_tables(3, 96, dtype=torch.bfloat16, device=x.device)

    with pytest.raises(TypeError, match="cos dtype must match x dtype"):
        fused_qknorm_rope(x, weight, cos, sin, 1e-6)


def test_fused_qknorm_rope_does_not_accept_missing_rotary_tables() -> None:
    x = torch.randn(2, 3, 4, 128)
    weight = torch.ones(128)

    with pytest.raises(TypeError, match="cos must be a torch.Tensor"):
        fused_qknorm_rope(x, weight, None, None, 1e-6)  # type: ignore[arg-type]


def test_fused_qknorm_rope_requires_cuda() -> None:
    x = torch.randn(2, 3, 4, 128)
    weight = torch.ones(128)
    cos, sin = _rotary_tables(3, 96, dtype=x.dtype, device=x.device)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        fused_qknorm_rope(x, weight, cos, sin, 1e-6)


@pytest.mark.parametrize(
    "rotary_dim,shape,use_input_view",
    [
        pytest.param(96, (2, 11, 5, 128), False, id="partial-96-batch2-seq11-heads5"),
        pytest.param(128, (2, 7, 3, 128), True, id="full-128-noncontiguous-input-view"),
    ],
)
def test_fused_qknorm_rope_matches_eager_bf16_cuda(
    rotary_dim: int,
    shape: tuple[int, ...],
    use_input_view: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton fusion")
    if not HAVE_TRITON:
        pytest.skip("Triton is required for the fusion")

    torch.manual_seed(1)
    device = torch.device("cuda")
    if use_input_view:
        x = torch.randn(*shape[:-1], shape[-1] * 2, dtype=torch.bfloat16, device=device)[..., ::2]
        assert not x.is_contiguous()
    else:
        x = torch.randn(shape, dtype=torch.bfloat16, device=device)
    weight = (1.0 + 0.05 * torch.randn(shape[-1], dtype=torch.bfloat16, device=device)).contiguous()
    cos, sin = _rotary_tables(shape[1], rotary_dim, dtype=x.dtype, device=device)

    with torch.inference_mode():
        actual = fused_qknorm_rope(x, weight, cos, sin, 1e-6)
        expected = _eager_qknorm_rope(x, weight, cos, sin, 1e-6)

    # The fused kernel keeps RMSNorm and both RoPE products in FP32 registers
    # until its final BF16 store. Eager materializes BF16 intermediates, and
    # PyTorch/Triton reductions need not use the same summation order.
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert actual.is_contiguous()


def test_fused_qknorm_rope_matches_eager_beyond_int32_element_count() -> None:
    """Regression: kernel row offsets must be int64.

    With int32 offsets, ``row * head_dim`` wraps once the flattened input
    crosses 2**31 elements and the kernel reads/writes out of bounds (CUDA
    illegal memory access). ``(1, 8_500_000, 2, 128)`` is 2.176e9 elements,
    just past the boundary; for H3's 56 heads x 128 head_dim the equivalent
    is ``batch*seq >= 299_593`` tokens per rank, reachable at SP=1.

    GPU assumption: needs ~16 GiB free CUDA memory (input + output at
    bf16 plus the fp32 rotary-table construction); skips below 20 GiB.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton fusion")
    if not HAVE_TRITON:
        pytest.skip("Triton is required for the fusion")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 20 * 1024**3:
        pytest.skip("needs ~20 GiB free GPU memory for a >2**31-element input")

    heads, head_dim, rotary_dim = 2, 128, 96
    seq_len = 8_500_000
    assert seq_len * heads * head_dim > 2**31

    torch.manual_seed(9)
    device = torch.device("cuda")
    x = torch.randn(1, seq_len, heads, head_dim, dtype=torch.bfloat16, device=device)
    weight = (1.0 + 0.05 * torch.randn(head_dim, dtype=torch.bfloat16, device=device)).contiguous()
    cos, sin = _rotary_tables(seq_len, rotary_dim, dtype=x.dtype, device=device)

    with torch.inference_mode():
        fused = fused_qknorm_rope(x, weight, cos, sin, 1e-6)
        torch.cuda.synchronize()
        # Compare only head/tail slices against eager: a full-tensor eager
        # reference would double peak memory for no extra coverage, and the
        # tail rows are exactly the ones an int32 wrap corrupts first.
        expected_head = _eager_qknorm_rope(x[:, :8], weight, cos[:8], sin[:8], 1e-6)
        expected_tail = _eager_qknorm_rope(x[:, -8:], weight, cos[-8:], sin[-8:], 1e-6)
        torch.testing.assert_close(fused[:, :8], expected_head, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(fused[:, -8:], expected_tail, atol=2e-2, rtol=2e-2)
