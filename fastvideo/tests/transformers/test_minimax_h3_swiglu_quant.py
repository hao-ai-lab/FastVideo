# SPDX-License-Identifier: Apache-2.0
"""Blackwell-only byte parity for the fused H3 SwiGLU/NVFP4 preparation."""
import pytest
import torch


@pytest.mark.parametrize("rows,width,scale", [(129, 64, 1.0), (257, 14336, 0.1395), (33, 256, 91.75)])
def test_swiglu_quantization_matches_flashinfer(rows: int, width: int, scale: float) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("native FP4 conversion requires a Blackwell GPU")
    pytest.importorskip("flashinfer")
    from fastvideo.layers.quantization.nvfp4_config import _nvfp4_quantize, _require_flashinfer
    from fastvideo.models.dits.minimax_h3_fusions.swiglu import minimax_h3_swiglu
    from fastvideo.models.dits.minimax_h3_fusions.swiglu_quant import swiglu_quantize_nvfp4

    torch.manual_seed(321)
    x = torch.randn((1, rows, width * 2), device="cuda", dtype=torch.bfloat16) * 3
    x[:, 0] = 0
    global_scale = torch.tensor(scale, device="cuda", dtype=torch.float32)
    layout, _, _ = _require_flashinfer()
    with torch.inference_mode():
        expected_codes, expected_scales = _nvfp4_quantize(
            minimax_h3_swiglu(x).view(rows, width), global_scale, sfLayout=layout.layout_128x4,
        )
        codes, scales, returned_scale = swiglu_quantize_nvfp4(x, global_scale)
    assert torch.equal(codes, expected_codes)
    assert torch.equal(scales, expected_scales)
    assert returned_scale is global_scale
