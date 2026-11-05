# SPDX-License-Identifier: Apache-2.0
"""Row chunking for the NVFP4 mm_fp4 wrapper: chunks must tile the rows exactly, on 128-row boundaries."""
import math

import pytest
import torch

from fastvideo.layers.quantization import nvfp4_config
from fastvideo.layers.quantization.nvfp4_config import _mm_fp4_row_chunks


@pytest.mark.parametrize("rows", [35328, 35000, 16384, 20481])
@pytest.mark.parametrize("parts", [1, 2, 4])
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
