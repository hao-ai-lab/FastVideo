"""Masked lanes must not influence NVFP4 scales or produce non-finite padding.

The valid region ``[:29, :23]`` leaves the second 16-lane group of every row
partially masked and the last three rows fully masked, so the test covers
mixed groups, empty groups and (with ``all_invalid``) a fully masked tile.
"""
import pytest
import torch
import triton
import triton.language as tl
from fastvideo_kernel.triton_kernels.nvfp4_utils import MXFP_BLOCK_SIZE, _compute_dequant, _compute_quant_and_scale

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")

TILE = 32
GROUPS = TILE // MXFP_BLOCK_SIZE.value
PADDINGS = (0.0, 1e4, float("nan"))


@triton.jit
def quantize_masked(X, VALID, OUT, SCALE, TILE: tl.constexpr, GROUPS: tl.constexpr, GLOBAL: tl.constexpr,
                    TWO: tl.constexpr):
    rows = tl.arange(0, TILE)
    columns = tl.arange(0, TILE)
    values = tl.load(X + rows[:, None] * TILE + columns[None, :])
    valid = tl.load(VALID + rows[:, None] * TILE + columns[None, :])
    packed, scale, decode = _compute_quant_and_scale(values, valid, use_global_sf=GLOBAL, two_level_quant_P=TWO)
    out = _compute_dequant(packed, scale, decode, TILE, TILE, tl.bfloat16)
    tl.store(OUT + rows[:, None] * TILE + columns[None, :], out)
    tl.store(SCALE + rows[:, None] * GROUPS + tl.arange(0, GROUPS)[None, :], scale.to(tl.float32))


@pytest.mark.parametrize("global_scale,two_level", [(False, False), (True, False), (False, True)],
                         ids=["per_group", "global", "two_level"])
@pytest.mark.parametrize("all_invalid", [False, True], ids=["partial", "all_invalid"])
def test_masked_values_do_not_change_valid_quantization(global_scale, two_level, all_invalid):
    torch.manual_seed(42)
    source = torch.rand((TILE, TILE), device="cuda")
    valid = torch.zeros_like(source, dtype=torch.bool)
    if not all_invalid:
        valid[:29, :23] = True
    invalid_groups = ~valid.reshape(TILE, GROUPS, MXFP_BLOCK_SIZE.value).any(-1)
    results = []
    for padding in PADDINGS:
        values = torch.where(valid, source, padding)
        output = torch.empty_like(source, dtype=torch.bfloat16)
        scales = torch.empty((TILE, GROUPS), device="cuda")
        quantize_masked[(1,)](values, valid, output, scales, TILE, GROUPS, global_scale, two_level, num_warps=4)
        assert torch.isfinite(output).all(), padding
        assert torch.isfinite(scales).all(), padding
        assert torch.count_nonzero(output[~valid]) == 0, padding
        assert torch.count_nonzero(scales[invalid_groups]) == 0, padding
        results.append((output, scales))
    reference_output, reference_scales = results[0]
    for (output, scales), padding in zip(results[1:], PADDINGS[1:]):
        assert torch.equal(output[valid], reference_output[valid]), padding
        assert torch.equal(scales, reference_scales), padding
