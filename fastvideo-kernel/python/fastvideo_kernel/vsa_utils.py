"""VSA metadata utilities — standalone, no fastvideo framework dependency.

Provides the tile-partition index helpers and variable-block-size
computations that are needed to call `video_sparse_attn` or
`block_sparse_attn_from_indices` without depending on the full
fastvideo package.
"""

from __future__ import annotations

import functools
import math

import torch

VSA_TILE_SIZE = (4, 4, 4)
_SUPPORTED_VSA_BLOCK_VOLUMES = (64, 256)


def _canonicalize_device(device: torch.device | str) -> torch.device:
    """Resolve an indexless CUDA device before it is used as a cache key."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


@functools.lru_cache(maxsize=10)
def get_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    """Map raster-order token indices to tile-contiguous order.

    Groups spatially adjacent tokens into (ts_t x ts_h x ts_w) tiles
    so that each tile's tokens are contiguous in the output.
    """
    T, H, W = dit_seq_shape
    ts, hs, ws = tile_size
    indices = torch.arange(T * H * W, device=device, dtype=torch.long).reshape(T, H, W)
    ls = []
    for t in range(math.ceil(T / ts)):
        for h in range(math.ceil(H / hs)):
            for w in range(math.ceil(W / ws)):
                ls.append(indices[
                    t * ts:min(t * ts + ts, T),
                    h * hs:min(h * hs + hs, H),
                    w * ws:min(w * ws + ws, W),
                ].flatten())
    return torch.cat(ls, dim=0)


@functools.lru_cache(maxsize=10)
def get_reverse_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    """Inverse of get_tile_partition_indices: tile order back to raster."""
    return torch.argsort(get_tile_partition_indices(dit_seq_shape, tile_size, device))


@functools.lru_cache(maxsize=10)
def construct_variable_block_sizes(
    dit_seq_shape: tuple[int, int, int],
    num_tiles: tuple[int, int, int],
    device: torch.device,
    tile_size: tuple[int, int, int] = VSA_TILE_SIZE,
) -> torch.LongTensor:
    """Compute the number of valid tokens in each tile.

    Tiles at the boundary of each dimension may contain fewer tokens
    when the video shape is not evenly divisible by tile_size.
    """
    t, h, w = dit_seq_shape
    ts_t, ts_h, ts_w = tile_size
    n_t, n_h, n_w = num_tiles

    def _sizes(dim_len: int, tile: int, n: int) -> torch.LongTensor:
        sizes = torch.full((n,), tile, dtype=torch.int, device=device)
        remainder = dim_len - (n - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(t, ts_t, n_t)
    h_sizes = _sizes(h, ts_h, n_h)
    w_sizes = _sizes(w, ts_w, n_w)

    return (
        t_sizes[:, None, None]
        * h_sizes[None, :, None]
        * w_sizes[None, None, :]
    ).reshape(-1)


def get_non_pad_index(
    variable_block_sizes: torch.LongTensor,
    max_block_size: int,
) -> torch.LongTensor:
    """Find positions of real tokens within a block-padded layout.

    Each block occupies max_block_size slots. This returns the flat
    indices of the valid (non-padding) positions.
    """
    n_win = variable_block_sizes.shape[0]
    device = variable_block_sizes.device
    starts_pad = torch.arange(n_win, device=device) * max_block_size
    index_pad = starts_pad[:, None] + torch.arange(max_block_size, device=device)[None, :]
    index_mask = torch.arange(max_block_size, device=device)[None, :] < variable_block_sizes[:, None]
    return index_pad[index_mask]


def build_vsa_metadata(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int] = VSA_TILE_SIZE,
    device: torch.device | str = "cuda",
) -> dict:
    """Build all VSA metadata from a video latent shape in one call.

    Args:
        dit_seq_shape: (T, H, W) — temporal frames, spatial height, width.
        tile_size: (ts_t, ts_h, ts_w) — tokens per tile in each dimension.
            The resulting tile volume must be supported by the VSA kernels.
        device: Target device for index tensors.

    Returns:
        Dict with keys: tile_partition_indices, reverse_tile_partition_indices,
        variable_block_sizes, non_pad_index, num_tiles, max_block_size.
    """
    device = _canonicalize_device(device)

    T, H, W = dit_seq_shape
    ts_t, ts_h, ts_w = tile_size
    max_block_size = math.prod(tile_size)
    if max_block_size not in _SUPPORTED_VSA_BLOCK_VOLUMES:
        raise ValueError(
            f"Unsupported VSA tile volume {max_block_size} for tile_size={tile_size}; "
            f"supported volumes are {_SUPPORTED_VSA_BLOCK_VOLUMES}."
        )

    num_tiles = (
        math.ceil(T / ts_t),
        math.ceil(H / ts_h),
        math.ceil(W / ts_w),
    )

    tile_indices = get_tile_partition_indices(dit_seq_shape, tile_size, device)
    reverse_tile_indices = get_reverse_tile_partition_indices(dit_seq_shape, tile_size, device)
    vbs = construct_variable_block_sizes(dit_seq_shape, num_tiles, device, tile_size)
    npi = get_non_pad_index(vbs, max_block_size)

    return {
        "tile_partition_indices": tile_indices,
        "reverse_tile_partition_indices": reverse_tile_indices,
        "variable_block_sizes": vbs,
        "non_pad_index": npi,
        "num_tiles": num_tiles,
        "max_block_size": max_block_size,
    }
