# SPDX-License-Identifier: Apache-2.0
"""Row chunking for the NVFP4 mm_fp4 wrapper: chunks must tile the rows exactly, on 128-row boundaries."""
import math

import pytest
import torch

from fastvideo.layers.quantization import nvfp4_config
from fastvideo.layers.quantization.nvfp4_config import _mm_fp4_row_chunks


@pytest.mark.parametrize("rows", [35328, 35000, 16384, 20481, 14859])
@pytest.mark.parametrize("parts", [1, 2, 4, 8])
def test_row_chunks_cover_rows_in_order(rows: int, parts: int) -> None:
    a = torch.arange(rows * 2, dtype=torch.float32).view(rows, 2)
    scale = torch.arange(rows + (-rows) % 128, dtype=torch.float32).view(-1, 1)
    chunks = _mm_fp4_row_chunks(a, scale, parts)
    assert torch.equal(torch.cat([c for c, _ in chunks]), a)
    # every chunk start is a 128-row scale tile boundary
    offset = 0
    for chunk, chunk_scale in chunks:
        assert offset % 128 == 0
        assert chunk_scale[0, 0].item() == offset
        scale_hi = min(math.ceil((offset + chunk.shape[0]) / 128) * 128, scale.shape[0])
        assert chunk_scale.shape[0] == scale_hi - offset
        offset += chunk.shape[0]
    assert offset == rows


@pytest.mark.parametrize(("use_8x4", "a_scale_rows"), [(True, 20000), (False, None)])
def test_row_split_stays_whole_without_128x4_scale_rows(monkeypatch, use_8x4, a_scale_rows):
    calls = []

    def fake_chunked(*args):
        calls.append(args[-1])
        return None

    monkeypatch.setattr(nvfp4_config, "_mm_fp4_chunked", fake_chunked)
    a = torch.zeros(256, 64)
    a_scale = torch.zeros(256, 1) if a_scale_rows else torch.zeros(256)
    assert nvfp4_config._mm_fp4_row_split(a, torch.zeros(32, 64), a_scale, torch.zeros(32, 1), None,
                                          torch.bfloat16, 16, use_8x4, "cudnn", True) is None
    assert calls == [1]


@pytest.mark.parametrize("policy,expected", [("off", "whole"), ("table", "split")])
def test_large_cudnn_call_splits_only_when_opted_in(monkeypatch, policy, expected):
    from fastvideo import envs

    calls = []
    monkeypatch.setattr(nvfp4_config, "_register_ops_once", lambda: None)
    monkeypatch.setattr(torch.ops.fastvideo_fp4, "mm_fp4", lambda *args: calls.append("whole"), raising=False)
    monkeypatch.setattr(nvfp4_config, "_mm_fp4_row_split", lambda *args: calls.append("split"))
    with envs.FASTVIDEO_NVFP4_MM_ROW_SPLIT.override(policy):
        nvfp4_config._mm_fp4(torch.empty(16384, 32), torch.empty(16, 64), torch.empty(16384, 1),
                              torch.empty(16, 1), None, torch.bfloat16, None, backend="cudnn")
    assert calls == [expected]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 direct-row parity requires CUDA")
def test_compact_h3_direct_rows_match_whole_cudnn(monkeypatch):
    """Real GB10 GEMMs, including non-aligned final rows and the public dispatch."""
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("This measured dispatch is specific to GB10")
    import flashinfer
    from fastvideo import envs

    torch.manual_seed(7331)
    with torch.inference_mode():
        a = torch.randn((14859, 5376), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((28672, 5376), device="cuda", dtype=torch.bfloat16)
        sf = torch.ones((), device="cuda", dtype=torch.float32)
        qa, sa = flashinfer.nvfp4_quantize(a, sf)
        qw, sw = flashinfer.nvfp4_quantize(weight, sf)
        qa, qw = qa.view(torch.uint8), qw.view(torch.uint8)
        expected = flashinfer.mm_fp4(qa, qw.T, sa, sw.T, sf, torch.bfloat16, backend="cudnn")
        calls = []
        direct = nvfp4_config._mm_fp4_direct_rows

        def recorded(*args):
            calls.append(args[-1])
            return direct(*args)

        monkeypatch.setattr(nvfp4_config, "_mm_fp4_direct_rows", recorded)
        with envs.FASTVIDEO_NVFP4_MM_ROW_SPLIT.override("table"):
            actual = nvfp4_config._mm_fp4(qa, qw.T, sa, sw.T, sf, torch.bfloat16, None, backend="cudnn")
        assert calls == [8]
        assert torch.equal(actual, expected)
        calls.clear()
        with envs.FASTVIDEO_NVFP4_MM_ROW_SPLIT.override("off"):
            whole = nvfp4_config._mm_fp4(qa, qw.T, sa, sw.T, sf, torch.bfloat16, None, backend="cudnn")
        assert not calls
        assert torch.equal(whole, expected)
