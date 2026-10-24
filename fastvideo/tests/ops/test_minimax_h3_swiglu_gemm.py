# SPDX-License-Identifier: Apache-2.0
"""GB10 native epilogue parity, including padded-scheduler store bounds."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one SM121 GPU")


@pytest.mark.parametrize("rows", [1, 127, 128, 129, 1025, 7430])
@pytest.mark.parametrize("magnitude", [1.0, 8.0])
@torch.inference_mode()
def test_epilogue_matches_bf16_and_quantization_boundaries(rows, magnitude):
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("native SM121 epilogue")
    native = pytest.importorskip('_fastvideo_h3_swiglu_gemm')
    import flashinfer
    from flashinfer import SfLayout, nvfp4_quantize
    from fastvideo.models.dits.minimax_h3_fusions.swiglu import minimax_h3_swiglu
    from fastvideo.models.dits.minimax_h3_fusions.swiglu_gemm import pair_weight_columns
    from fastvideo.models.dits.minimax_h3_fusions.swiglu_quant import swiglu_quantize_nvfp4

    torch.manual_seed(rows)
    n, k = 512, 256
    unit = torch.ones((), device='cuda', dtype=torch.float32)
    a = torch.randn(rows, k, device='cuda', dtype=torch.bfloat16) * magnitude
    w = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    qa, sa = nvfp4_quantize(a, unit, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    qw, sw = nvfp4_quantize(w, unit, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    bf16 = flashinfer.mm_fp4(qa, qw.T, sa, sw.T, unit, torch.bfloat16, backend='cudnn')
    expected_q, expected_s, _ = swiglu_quantize_nvfp4(bf16, unit)
    expected_y = minimax_h3_swiglu(bf16)
    expected_amax = expected_y.float().abs().reshape(rows, n // 32, 16).amax(-1)
    paired_w, paired_s = pair_weight_columns(qw, sw)
    qn, sn = expected_q.numel(), expected_s.numel()
    qstorage = torch.full((qn + 1048576,), 165, device='cuda', dtype=torch.uint8)
    sstorage = torch.full((sn + 1048576,), 165, device='cuda', dtype=torch.uint8)
    q = qstorage[:qn].view_as(expected_q)
    s = sstorage[:sn].view_as(expected_s)
    amax = torch.empty_like(expected_s, dtype=torch.float32)
    y = torch.empty_like(expected_y)
    for raster, swizzle in [(0, 1), (0, 4), (2, 8)]:
        native.run(qa, paired_w, sa, paired_s, unit, unit, q, s, amax, y, raster=raster, swizzle=swizzle)
        assert torch.equal(q, expected_q)
        assert torch.equal(s, expected_s)
        assert torch.equal(y.view(torch.int16), expected_y.view(torch.int16))
        assert torch.equal(amax[:rows], expected_amax)
        assert bool((qstorage[qn:] == 165).all())
        assert bool((sstorage[sn:] == 165).all())


@torch.inference_mode()
def test_omni_eight_chunks_preserve_tail_scales_and_output_guards():
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("native SM121 epilogue")
    native = pytest.importorskip('_fastvideo_h3_swiglu_gemm')
    import flashinfer
    from flashinfer import SfLayout, nvfp4_quantize
    from fastvideo.layers.quantization.nvfp4_config import _mm_fp4_row_chunks
    from fastvideo.models.dits.minimax_h3_fusions.swiglu_gemm import pair_weight_columns
    from fastvideo.models.dits.minimax_h3_fusions.swiglu_quant import swiglu_quantize_nvfp4

    torch.manual_seed(831)
    rows, n, k = 29678, 512, 256
    unit = torch.ones((), device='cuda', dtype=torch.float32)
    a = torch.randn(rows, k, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    qa, sa = nvfp4_quantize(a, unit, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    qw, sw = nvfp4_quantize(w, unit, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    bf16 = flashinfer.mm_fp4(qa, qw.T, sa, sw.T, unit, torch.bfloat16, backend='cudnn')
    expected_q, expected_s, _ = swiglu_quantize_nvfp4(bf16, unit)
    paired_w, paired_s = pair_weight_columns(qw, sw)
    qn, sn = expected_q.numel(), expected_s.numel()
    qstorage = torch.full((qn + 1048576,), 165, device='cuda', dtype=torch.uint8)
    sstorage = torch.full((sn + 1048576,), 165, device='cuda', dtype=torch.uint8)
    q, scales = qstorage[:qn].view_as(expected_q), sstorage[:sn].view_as(expected_s)
    start = 0
    for chunk, chunk_scale in _mm_fp4_row_chunks(qa, sa, 8):
        end = start + chunk.shape[0]
        native.run(chunk, paired_w, chunk_scale, paired_s, unit, unit,
                   q[start:end], scales[start:(end + 127) // 128 * 128])
        start = end
    assert start == rows
    assert torch.equal(q, expected_q) and torch.equal(scales, expected_s)
    assert bool((qstorage[qn:] == 165).all()) and bool((sstorage[sn:] == 165).all())
