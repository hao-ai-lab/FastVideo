# SPDX-License-Identifier: Apache-2.0
"""Row chunking for the NVFP4 mm_fp4 wrapper: chunks must tile the rows exactly, on 128-row boundaries."""
import pytest
import torch

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
        offset += chunk.shape[0]
    assert offset == rows
