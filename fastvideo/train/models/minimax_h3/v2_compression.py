# SPDX-License-Identifier: Apache-2.0
"""Training-side alias of the shared FastH3 token compression (the implementation serves inference too)."""
from fastvideo.models.dits.minimax_h3_token_compression import (MODES, CompressionSpec, TokenCompression,
                                                                block_flops_share, compressed_grid, grid_ops,
                                                                token_reduction)

__all__ = [
    "MODES", "CompressionSpec", "TokenCompression", "block_flops_share", "compressed_grid", "grid_ops",
    "token_reduction"
]
