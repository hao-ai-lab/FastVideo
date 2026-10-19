# SPDX-License-Identifier: Apache-2.0
"""The DGX Spark (GB10) NVFP4 quantization fence covers inference and the QAT linear.

GB10 is emulated by patching ``_is_dgx_spark``; any Blackwell GPU runs the kernels.
"""
from __future__ import annotations

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")

from fastvideo.layers import fp4linear  # noqa: E402
from fastvideo.layers.quantization import nvfp4_config as nv  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
                                reason="NVFP4 kernels need a Blackwell GPU")

LAYOUT_128X4 = flashinfer.SfLayout.layout_128x4


def _activation():
    torch.manual_seed(0)
    x = torch.randn(200, 256, device="cuda", dtype=torch.bfloat16)
    return x, (448.0 * 6.0) / x.float().abs().amax()


def _pad_rows(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.pad(x, (0, 0, 0, (-x.shape[0]) % 128))


def _spy_quantize(monkeypatch):
    real = flashinfer.nvfp4_quantize
    pdl = []

    def spy(*args, **kwargs):
        pdl.append(kwargs.get("enable_pdl"))
        return real(*args, **kwargs)

    monkeypatch.setattr(flashinfer, "nvfp4_quantize", spy)
    return real, pdl


def _spy_sync(monkeypatch):
    real = torch.cuda.Stream.synchronize
    syncs = []
    monkeypatch.setattr(torch.cuda.Stream, "synchronize", lambda self: syncs.append(self) or real(self))
    return syncs


@pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1),
                    reason="needs a non-GB10 Blackwell GPU")
def test_fenced_quantize_matches_flashinfer_without_a_fence_off_spark(monkeypatch) -> None:
    real, pdl = _spy_quantize(monkeypatch)
    syncs = _spy_sync(monkeypatch)
    x, global_sf = _activation()
    quantized, scales = nv.nvfp4_quantize_fenced(x, global_sf, LAYOUT_128X4.value)
    expected = real(x, global_sf, sfLayout=LAYOUT_128X4, do_shuffle=False)
    assert pdl == [None] and syncs == []
    assert torch.equal(quantized, expected[0]) and torch.equal(scales, expected[1])


def test_spark_inference_op_disables_pdl_and_fences(monkeypatch) -> None:
    real, pdl = _spy_quantize(monkeypatch)
    syncs = _spy_sync(monkeypatch)
    monkeypatch.setattr(nv, "_is_dgx_spark", lambda _index: True)
    x, global_sf = _activation()
    quantized, scales = nv._nvfp4_quantize(x, global_sf, sfLayout=LAYOUT_128X4)
    expected = real(_pad_rows(x), global_sf, sfLayout=LAYOUT_128X4, do_shuffle=False)
    assert pdl == [False] and len(syncs) == 1
    assert torch.equal(quantized, expected[0][:x.shape[0]]) and torch.equal(scales, expected[1])


def test_spark_rejects_cuda_graph_capture(monkeypatch) -> None:
    monkeypatch.setattr(nv, "_is_dgx_spark", lambda _index: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    x, global_sf = _activation()
    with pytest.raises(RuntimeError, match="completion fence"):
        nv.nvfp4_quantize_fenced(x, global_sf, LAYOUT_128X4.value)


def test_qat_linear_quantizes_through_the_fence_with_unchanged_numerics(monkeypatch) -> None:
    torch.manual_seed(1)
    x = torch.randn(4, 64, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16) / 16
    bias = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    off_spark = fp4linear._LinearFWD4BWD16Fn.apply(x, weight, bias, "cutlass", 16, True)

    _, pdl = _spy_quantize(monkeypatch)
    syncs = _spy_sync(monkeypatch)
    monkeypatch.setattr(nv, "_is_dgx_spark", lambda _index: True)
    on_spark = fp4linear._LinearFWD4BWD16Fn.apply(x, weight, bias, "cutlass", 16, True)
    assert pdl == [False, False] and len(syncs) == 2
    assert torch.equal(on_spark, off_spark)

    reference = torch.nn.functional.linear(x.float(), weight.float(), bias.float())
    relative = (on_spark.float() - reference).norm() / reference.norm()
    assert relative < 0.15


@pytest.mark.parametrize("rows", [200, 256])
def test_swizzled_quantize_skips_the_pad_copy_bit_exactly(monkeypatch, rows: int) -> None:
    torch.manual_seed(2)
    x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
    global_sf = (448.0 * 6.0) / x.float().abs().amax()
    expected = flashinfer.nvfp4_quantize(_pad_rows(x), global_sf, sfLayout=LAYOUT_128X4, do_shuffle=False)
    # nv.F is torch.nn.functional, so the spy sees every pad; install it after the reference.
    pads = []
    real_pad = torch.nn.functional.pad
    monkeypatch.setattr(nv.F, "pad", lambda *args, **kwargs: pads.append(args) or real_pad(*args, **kwargs))
    quantized, scales = nv._nvfp4_quantize(x, global_sf, sfLayout=LAYOUT_128X4)
    assert pads == []
    assert quantized.shape[0] == rows
    assert torch.equal(quantized, expected[0][:rows]) and torch.equal(scales, expected[1])


@pytest.mark.parametrize("rows", [200, 256])
@pytest.mark.parametrize("layout", ["layout_128x4", "layout_8x4", "layout_linear"])
@pytest.mark.parametrize("do_shuffle", [False, True])
def test_fake_quantize_reports_flashinfer_shapes(rows: int, layout: str, do_shuffle: bool) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    if do_shuffle and (layout != "layout_128x4" or rows % 128):
        pytest.skip("FlashInfer only shuffles 128x4, and _nvfp4_quantize row-pads shuffled inputs")
    sf_layout = getattr(flashinfer.SfLayout, layout).value
    nv._register_ops_once()
    torch.manual_seed(3)
    x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
    global_sf = (448.0 * 6.0) / x.float().abs().amax()
    real = torch.ops.fastvideo_fp4.nvfp4_quantize(x, global_sf, sf_layout, do_shuffle)
    with FakeTensorMode() as mode:
        fake = torch.ops.fastvideo_fp4.nvfp4_quantize(mode.from_tensor(x), mode.from_tensor(global_sf), sf_layout,
                                                      do_shuffle)
    assert [tuple(t.shape) for t in fake] == [tuple(t.shape) for t in real]


def test_compiled_swizzled_quantize_quantizes_once_without_row_guards(monkeypatch) -> None:
    from torch._dynamo.testing import CompileCounterWithBackend

    calls = []
    real_fenced = nv.nvfp4_quantize_fenced
    monkeypatch.setattr(nv, "nvfp4_quantize_fenced", lambda x, *args: calls.append(x.shape[0]) or real_fenced(x, *args))
    counter = CompileCounterWithBackend("inductor")
    torch._dynamo.reset()
    compiled = torch.compile(lambda x, sf: nv._nvfp4_quantize(x, sf, sfLayout=LAYOUT_128X4),
                             backend=counter,
                             dynamic=True,
                             fullgraph=True)
    for rows in (200, 256, 300):
        torch.manual_seed(rows)
        x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
        global_sf = (448.0 * 6.0) / x.float().abs().amax()
        calls.clear()
        quantized, scales = compiled(x, global_sf)
        expected = flashinfer.nvfp4_quantize(_pad_rows(x), global_sf, sfLayout=LAYOUT_128X4, do_shuffle=False)
        assert calls == [rows]
        assert torch.equal(quantized, expected[0][:rows]) and torch.equal(scales, expected[1])
    assert counter.frame_count == 1
    torch._dynamo.reset()


def test_unpadded_flashinfer_scales_fall_back_to_the_pad_copy_once(monkeypatch) -> None:
    monkeypatch.setattr(nv, "_FLASHINFER_PADS_128X4_SCALES", True)
    real_fenced = nv.nvfp4_quantize_fenced
    calls = []

    def unpadded(x, global_sf, sf_layout, do_shuffle=False):
        calls.append(x.shape[0])
        quantized, scales = real_fenced(x, global_sf, sf_layout, do_shuffle)
        return quantized, scales[:x.shape[0]]

    monkeypatch.setattr(nv, "nvfp4_quantize_fenced", unpadded)
    x, global_sf = _activation()
    nv._nvfp4_quantize(x, global_sf, sfLayout=LAYOUT_128X4)
    assert calls == [200, 256] and nv._FLASHINFER_PADS_128X4_SCALES is False
    calls.clear()
    nv._nvfp4_quantize(x, global_sf, sfLayout=LAYOUT_128X4)
    assert calls == [256]
