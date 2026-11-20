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
    native = pytest.importorskip('fastvideo_kernel.h3_swiglu_gemm_binding',
                                 reason='build fastvideo-kernel with TORCH_CUDA_ARCH_LIST=12.1a')
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
    native = pytest.importorskip('fastvideo_kernel.h3_swiglu_gemm_binding',
                                 reason='build fastvideo-kernel with TORCH_CUDA_ARCH_LIST=12.1a')
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


def _native_inputs(rows, n=512, k=256):
    from flashinfer import SfLayout, nvfp4_quantize
    from fastvideo.models.dits.minimax_h3_fusions.swiglu_gemm import pair_weight_columns

    unit = torch.ones((), device='cuda', dtype=torch.float32)
    a = torch.randn(rows, k, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    qa, sa = nvfp4_quantize(a, unit, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    qw, sw = nvfp4_quantize(w, unit, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    paired_w, paired_s = pair_weight_columns(qw, sw)
    return qa, paired_w, sa, paired_s, unit


@pytest.mark.parametrize("output,shrink", [("q", 1), ("scales", 1), ("amax", 1), ("values", 1),
                                           ("scales", 0), ("amax", 0)])
@torch.inference_mode()
def test_undersized_outputs_raise_before_launch(output, shrink):
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("native SM121 epilogue")
    native = pytest.importorskip('fastvideo_kernel.h3_swiglu_gemm_binding',
                                 reason='build fastvideo-kernel with TORCH_CUDA_ARCH_LIST=12.1a')
    rows, n = 129, 512
    qa, paired_w, sa, paired_s, unit = _native_inputs(rows, n)
    padded = (rows + 127) // 128 * 128
    outputs = {
        'q': torch.empty(rows, n // 4, device='cuda', dtype=torch.uint8),
        'scales': torch.empty(padded, n // 32, device='cuda', dtype=torch.uint8),
        'amax': torch.empty(padded, n // 32, device='cuda', dtype=torch.float32),
        'values': torch.empty(rows, n // 2, device='cuda', dtype=torch.bfloat16),
    }
    outputs[output] = (outputs[output].flatten()[:-1] if shrink else outputs[output][:rows].clone())
    with pytest.raises(ValueError, match=f'GEMM {output} '):
        native.run(qa, paired_w, sa, paired_s, unit, unit, outputs['q'], outputs['scales'],
                   outputs['amax'], outputs['values'])


@torch.inference_mode()
def test_misplaced_or_mistyped_outputs_raise_before_launch():
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("native SM121 epilogue")
    native = pytest.importorskip('fastvideo_kernel.h3_swiglu_gemm_binding',
                                 reason='build fastvideo-kernel with TORCH_CUDA_ARCH_LIST=12.1a')
    rows, n = 128, 512
    qa, paired_w, sa, paired_s, unit = _native_inputs(rows, n)
    q = torch.empty(rows, n // 4, device='cuda', dtype=torch.uint8)
    scales = torch.empty(rows, n // 32, device='cuda', dtype=torch.uint8)
    with pytest.raises(ValueError, match='GEMM q '):
        native.run(qa, paired_w, sa, paired_s, unit, unit, q.cpu(), scales)
    with pytest.raises(ValueError, match='GEMM q '):
        native.run(qa, paired_w, sa, paired_s, unit, unit, q.view(torch.int8), scales)
    with pytest.raises(ValueError, match='GEMM q '):
        native.run(qa, paired_w, sa, paired_s, unit, unit, q.T.contiguous().T, scales)
    with pytest.raises(ValueError, match='GEMM alpha '):
        native.run(qa, paired_w, sa, paired_s, unit.double(), unit, q, scales)
    with pytest.raises(ValueError, match='GEMM b_scale '):
        native.run(qa, paired_w, sa, paired_s.contiguous(), unit, unit, q, scales)
    with pytest.raises(ValueError, match='contiguous b.T'):
        native.run(qa, paired_w.contiguous(), sa, paired_s, unit, unit, q, scales)


@torch.inference_mode()
def test_try_swiglu_gemm_matches_unfused_feed_forward_and_reuses_paired_weights(monkeypatch):
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("native SM121 epilogue")
    pytest.importorskip('fastvideo_kernel.h3_swiglu_gemm_binding',
                        reason='build fastvideo-kernel with TORCH_CUDA_ARCH_LIST=12.1a')
    from flashinfer import SfLayout, nvfp4_quantize
    import fastvideo.layers.quantization.nvfp4_config as nvfp4
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3FeedForward
    from fastvideo.models.dits.minimax_h3_fusions import swiglu_gemm

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        ff = MiniMaxH3FeedForward(5376, 14336, quant_config=nvfp4.NVFP4Config(),
                                  prefix="minimax_h3.transformer_blocks.0.ff", fuse_swiglu=True)
    finally:
        torch.set_default_dtype(previous_dtype)
    generator = torch.Generator().manual_seed(4219)
    for parameter in ff.parameters():
        parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * 0.02)
    ff = ff.cuda().eval()
    nvfp4.convert_model_to_nvfp4(ff)
    ff.fc_out._nvfp4_input_global_sf = torch.tensor(1792.0, device='cuda', dtype=torch.float32)

    torch.manual_seed(5376)
    hidden = torch.randn(7430, 5376, device='cuda', dtype=torch.bfloat16)
    input_scale = ((448.0 * 6.0) / hidden.float().abs().amax()).reshape(())
    assert float(input_scale) != 1.0
    qa, sa = nvfp4_quantize(hidden, input_scale, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    pre_quantized = (qa, sa, input_scale)

    with monkeypatch.context() as patch:
        patch.setattr(swiglu_gemm, "_backend", lambda: None)
        expected = ff._forward(hidden, pre_quantized)

    pairings = []
    pair = swiglu_gemm.pair_weight_columns
    monkeypatch.setattr(swiglu_gemm, "pair_weight_columns", lambda *args: pairings.append(1) or pair(*args))
    first = swiglu_gemm.try_swiglu_gemm(ff, hidden, pre_quantized)
    paired_ptr = ff.fc_in._h3_swiglu_paired_weight.data_ptr()
    second = swiglu_gemm.try_swiglu_gemm(ff, hidden, pre_quantized)

    assert first is not None and second is not None
    assert len(pairings) == 1
    assert ff.fc_in._h3_swiglu_paired_weight.data_ptr() == paired_ptr
    assert torch.equal(first.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(second.view(torch.int16), expected.view(torch.int16))
