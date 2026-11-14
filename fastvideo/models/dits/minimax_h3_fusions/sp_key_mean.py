# SPDX-License-Identifier: Apache-2.0
"""Recover global key means from the SP receive's complete tile statistics.

Every tile contributes its FP32 mean times its valid count, and the total is
normalized by the padded sequence length, matching the zero-padded key tensor.
This changes floating-point reduction order and may alter the generated sample.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _key_mean(Pool, Counts, Mean, HEAD_STRIDE: tl.constexpr, TILE_STRIDE: tl.constexpr,
              TILES: tl.constexpr, SEQ: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    head = tl.program_id(0)
    rows = tl.arange(0, ROWS)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    pooled = tl.load(Pool + head * HEAD_STRIDE + rows[:, None] * TILE_STRIDE + cols[None, :],
                     rows[:, None] < TILES, other=0.0)
    count = tl.load(Counts + rows, rows < TILES, other=0).to(tl.float32)
    pooled = tl.where(count[:, None] > 0, pooled, 0.0)
    total = tl.sum(pooled * count[:, None], 0)
    mean = total * (1.0 / SEQ)
    tl.store(Mean + head * 128 + cols, mean)


def key_mean_from_tiles(pool: torch.Tensor, counts: torch.Tensor, sequence: int) -> torch.Tensor:
    """Return BF16 ``[1, 1, H, 128]`` means from FP32 ``[1, H, tiles, 128]``.

    Counts must be the same divisors used to produce ``pool``. Padding is
    included in ``sequence`` and contributes zero, just as in ``key.mean``.
    """
    batch, heads, tiles, dim = pool.shape
    if (batch != 1 or dim != 128 or sequence != tiles * 64 or pool.stride(-1) != 1
            or pool.dtype != torch.float32 or counts.shape != (tiles,)
            or counts.dtype not in (torch.int32, torch.int64) or not pool.is_cuda
            or counts.device != pool.device):
        raise ValueError("Pooled SP key means require CUDA tile-64 FP32 statistics with 128 channels")
    result = torch.empty((1, 1, heads, dim), device=pool.device, dtype=torch.bfloat16)
    _key_mean[(heads, 4)](
        pool, counts, result, pool.stride(1), pool.stride(2), tiles, sequence,
        triton.next_power_of_2(tiles), 32, num_warps=4,
    )
    return result
