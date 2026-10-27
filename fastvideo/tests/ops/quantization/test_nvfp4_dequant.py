# SPDX-License-Identifier: Apache-2.0
"""Fused weight expansion against the independent serialized Torch decoder."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton NVFP4 decoder")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("global_scale", [1.0, 2.7, 438.912])
def test_fused_nvfp4_expands_all_codes_and_scale_tiles(dtype, global_scale):
    from fastvideo.layers.quantization.nvfp4_dequant import dequantize_nvfp4_cuda
    from fastvideo.models.encoders.minimax_h3_checkpoint_nvfp4 import dequantize_serialized_nvfp4

    torch.manual_seed(23)
    # Multiple row/column tiles distinguish the swizzle from a row-major decoder.
    packed = torch.arange(256, device="cuda", dtype=torch.uint8).repeat(256, 1)
    scales = torch.randint(0, 127, (256, 32), device="cuda", dtype=torch.uint8)
    reference = dequantize_serialized_nvfp4(packed, scales, global_scale, dtype)
    actual = dequantize_nvfp4_cuda(packed, scales, global_scale, dtype)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
