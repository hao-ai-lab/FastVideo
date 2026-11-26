# SPDX-License-Identifier: Apache-2.0
"""VSA for MiniMax H3's packed mixed-modality self-attention.

H3 runs one joint bidirectional attention over
``[text | condition keyframes | audio | generated video]``, so this
backend differs from the Wan-tuned ``video_sparse_attn``:

- Tiles are ``[segment-pure prefix chunks] + [3D video tiles]``; prefix
  tiles never straddle segment boundaries. The tile size is selectable at
  metadata build time: 256 tokens ``(4,8,8)`` (default), 128 tokens
  ``(4,4,8)``, or 64 tokens ``(4,4,4)`` (see ``VSA_H3_TILE_SHAPES``).
- The builder takes the packed sequence as ordered segments: dense row
  counts (text, audio, image references) and one 3-D sparse region per
  video, wherever it sits. The generated video is the last region; a Ref2VA
  PDD run adds one earlier region per reference video. Rows are permuted to
  ``[dense chunks][region 0 tiles][region 1 tiles]...`` and
  ``untile_combined_index`` inverts the permutation, so the kernels see one
  dense prefix followed by video tiles. Every video query keeps its own
  top-k of EACH region (``video_tile_spans`` / ``span_sparsities``); the
  reference-video regions may use a different keep rate than the generated
  video.
- Selection is pure Python on pooled tile scores; the block-sparse kernel
  consumes an explicit bool mask, so no kernel changes are needed.
- The compression branch is gated by ``to_gate_compress``, which the base
  H3 checkpoint does not carry: the loader zero-initializes it, so
  untrained inference is exactly pure sparse and finetuning can learn the
  gate. VSA-distilled students (e.g. FastVideo-Minimax-H3-Preview) ship
  trained gates, which load and activate the branch.
- Non-video *queries* are always dense. Non-video *keys* are either
  always-selected for every query ("exempt", default) or compete in
  top-k under a FLOP-matched budget ("compete") — the ablation axis,
  switched per request via ``generate_video(..., vsa_mode=...)``
  (default: exempt). Per-request scheduling knobs
  (``vsa_dense_first_n_steps``, ``vsa_dense_layers``) let mixed schedules
  run the diffuse steps/layers dense while pushing the rest harder.

At tile256, FA4 CuTe or the Triton 256-to-64 expansion implements attention.
Tile64 uses Triton by default; ``FASTVIDEO_VSA_PTX_BLACKWELL=1`` opts into
native Blackwell forward and its registered autograd operator for training.
Tile128 requires the native Blackwell forward and, when gradients are tracked,
the native128 backward. Missing support fails explicitly; no64-token fallback
can implement128-token masks. ``FASTVIDEO_VSA_TRITON=1`` keeps tile64 training
on Triton and is incompatible with tile128 training.

An odd native64/128 tile count gets one internal, zero-valid partner tile.
Selection, compression, and returned rows retain the logical geometry. Training
uses fresh scatter buffers so later layers cannot overwrite saved activations.
Trailing transformer-capacity rows never enter tiling, pooling or top-k;
postprocessing restores their shape with zero output rows.

Inference at tile 128 prefers the native forward; when the extension or device
cannot run it, each 128-token tile splits into two contiguous 64-token halves
on the Triton kernels, and the call fails closed if those are missing too.

The ``vsa2`` variant (``MiniMaxH3VSAMetadataBuilder.build(variant="vsa2")``) runs VSA2 instead, on the fused
kernels of ``fastvideo-kernel/vsa2_train`` (router, coarse and fine branches and the gated merge in one call). It
serves PAN2's joint ``[video; text]`` attention for the FastPAN2 student, needs sm_100a / sm_103a (B200 / GB200,
B300) and raises on any other GPU; the extension builds on first use.
"""

import functools
import importlib.util
import itertools
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import torch

try:
    from fastvideo_kernel.block_sparse_attn import block_sparse_attn as block_sparse_attn_64_bhsd
    from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_256_bshd
    from fastvideo_kernel.triton_kernels.index import map_to_index
except ImportError:
    block_sparse_attn_64_bhsd = None
    block_sparse_attn_256_bshd = None
    map_to_index = None

try:
    # Optional: only present in fastvideo_kernel builds that carry the
    # ptx_blackwell block-sparse forward. The module itself imports
    # fine without the compiled symbols (`_HAS_VSA_PTX_BLACKWELL` is then False and
    # `is_supported` says no), so this only guards *module* availability.
    from fastvideo_kernel import block_sparse_attn_ptx_blackwell as _ptx_blackwell
except ImportError:
    _ptx_blackwell = None

try:
    from fastvideo_kernel.block_sparse_attn import (_ptx_blackwell_backward_is_supported,
                                                    block_sparse_attn_ptx_blackwell_op)
except ImportError:
    _ptx_blackwell_backward_is_supported = None
    block_sparse_attn_ptx_blackwell_op = None

import fastvideo.envs as envs
from fastvideo.attention.backends.abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                                                   AttentionMetadataBuilder, layer_idx_from_prefix)
from fastvideo.attention.backends.video_sparse_attn import (compute_topk, construct_variable_block_sizes,
                                                            get_non_pad_index, get_tile_partition_indices,
                                                            scatter_into_tile_buf)
from fastvideo.attention.backends.video_sparse_attn_h3_probe import probe_enabled, record_probe
from fastvideo.attention.backends.video_sparse_attn_h3_scatter import (scatter_rows_heads_first,
                                                                       supports_heads_first_scatter)
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Opt-in switch for the data-center Blackwell CUDA forward on the tile-64 no-grad path.
VSA_PTX_BLACKWELL_ENV = "FASTVIDEO_VSA_PTX_BLACKWELL"

VSA_H3_TILE_SIZE = (4, 8, 8)  # 256 elements -> FA4 CuTe fastpath on sm10.x (default)
_TILE_ELEMS = math.prod(VSA_H3_TILE_SIZE)
# Selectable tile geometries, keyed by element count (= the build-time
# ``tile_size``). 64 runs the native 64-token Triton block-sparse kernels for
# forward AND backward — the block map is already at kernel granularity, so no
# 256->64 mask expansion is involved and FASTVIDEO_VSA_CUTEDSL does not apply.
VSA_H3_TILE_SHAPES: dict[int, tuple[int, int, int]] = {
    _TILE_ELEMS: VSA_H3_TILE_SIZE,
    64: (4, 4, 4),
    # ptx_blackwell forward only (128-token block instantiation).
    128: (4, 4, 8),
}
# Tile sizes the ptx_blackwell kernel runs natively; its CTAs own
# adjacent pairs of query tiles, so an odd count gets one zero-valid partner.
_PTX_BLACKWELL_TILE_ELEMS = (64, 128)


@torch.library.custom_op(
    "fastvideo::h3_vsa_ptx_blackwell_from_mask_compat",
    mutates_args=(),
    device_types="cuda",
)
def _h3_vsa_ptx_blackwell_from_mask_compat(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> torch.Tensor:
    """Compile-safe mask adapter for kernel wheels predating the mask API."""
    if _ptx_blackwell is None or map_to_index is None:
        raise RuntimeError("The ptx_blackwell compatibility route requires the raw kernel and map_to_index")
    q2k_idx, q2k_num = map_to_index(block_map)
    out, _ = _ptx_blackwell.block_sparse_attn_ptx_blackwell(
        q,
        k,
        v,
        q2k_idx.to(torch.int32).contiguous(),
        q2k_num.to(torch.int32).contiguous(),
        variable_block_sizes.to(torch.int32).contiguous(),
        need_lse=False,
    )
    return out


@torch.library.register_fake("fastvideo::h3_vsa_ptx_blackwell_from_mask_compat")
def _h3_vsa_ptx_blackwell_from_mask_compat_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> torch.Tensor:
    del k, v, block_map, variable_block_sizes
    return torch.empty_like(q)


@torch.library.custom_op("fastvideo::h3_vsa_block_mask", mutates_args=())
def _h3_vsa_block_mask(scores: torch.Tensor) -> torch.Tensor:
    """``_build_block_mask`` reading the per-request geometry at run time, so compiled graphs don't guard on it."""
    from fastvideo.forward_context import get_forward_context

    metadata = get_forward_context().attn_metadata
    return _build_block_mask(scores, metadata.num_prefix_tiles, metadata.VSA_sparsity, metadata.exempt,
                             metadata.video_tile_spans, metadata.span_sparsities)


@torch.library.register_fake("fastvideo::h3_vsa_block_mask")
def _h3_vsa_block_mask_fake(scores: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(scores, dtype=torch.bool)


def _ptx_blackwell_has_compile_safe_mask_route(ptx_blackwell_mod: Any) -> bool:
    return (callable(getattr(ptx_blackwell_mod, "block_sparse_attn_ptx_blackwell_from_mask", None))
            or (callable(getattr(ptx_blackwell_mod, "block_sparse_attn_ptx_blackwell", None))
                and map_to_index is not None))


def _ptx_blackwell_from_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> tuple[torch.Tensor, None]:
    """Use the native mask entry when installed, otherwise the local adapter."""
    native = getattr(_ptx_blackwell, "block_sparse_attn_ptx_blackwell_from_mask", None)
    if callable(native):
        return native(q, k, v, block_map, variable_block_sizes)
    return _h3_vsa_ptx_blackwell_from_mask_compat(q, k, v, block_map, variable_block_sizes), None


def token_tile_and_valid(variable_block_sizes: torch.Tensor,
                         tile_elems: int = _TILE_ELEMS) -> tuple[torch.Tensor, torch.Tensor]:
    """Per padded-token tile id and pad-validity mask.

    The single encoding of the padding contract, shared by the probe and the
    test oracle so they cannot drift from the backend's tile geometry.
    ``tile_elems`` must match the metadata the sizes came from
    (``MiniMaxH3VSAMetadata.tile_elems``).
    """
    device = variable_block_sizes.device
    token_tile = torch.arange(variable_block_sizes.numel(), device=device).repeat_interleave(tile_elems)
    token_valid = (torch.arange(tile_elems, device=device)[None, :] < variable_block_sizes[:, None]).reshape(-1)
    return token_tile, token_valid


def _validate_h3_segment_geometry(
    segments: tuple[int | tuple[int, int, int], ...],
    variable_block_sizes: torch.Tensor,
    untile_combined_index: torch.Tensor,
    tile_elems: int = _TILE_ELEMS,
) -> None:
    """Fail synchronously on out-of-bounds tile geometry.

    ``segments`` is the packed-order interleave: an ``int`` is a dense
    (prefix) segment's row count, a triple is one video region's token grid.

    Invariants the block-sparse kernel trusts without checking:
    every tile's valid size is in (0, tile_elems]; the sizes sum to the
    packed sequence length; and ``untile_combined_index`` maps each packed
    row to exactly one non-pad slot of the padded tile buffer. A violation
    would surface only as an async device fault at some later kernel or
    collective (e.g. an FSDP all-gather), which is unattributable — so raise
    here, once per cached geometry, with the numbers in hand.
    """
    total = sum(math.prod(segment) if isinstance(segment, tuple) else segment for segment in segments)
    n_pad = variable_block_sizes.numel() * tile_elems
    sizes_min = int(variable_block_sizes.min())
    sizes_max = int(variable_block_sizes.max())
    sizes_sum = int(variable_block_sizes.sum())
    if sizes_min < 1 or sizes_max > tile_elems or sizes_sum != total:
        raise ValueError(f"VSA-H3 tile sizes out of bounds for segments={segments}, "
                         f"tile_elems={tile_elems}: min={sizes_min}, max={sizes_max}, sum={sizes_sum}, "
                         f"expected sum={total}.")
    if untile_combined_index.numel() != total:
        raise ValueError(f"VSA-H3 untile index has {untile_combined_index.numel()} entries for a packed "
                         f"sequence of {total} rows (segments={segments}).")
    idx_min = int(untile_combined_index.min())
    idx_max = int(untile_combined_index.max())
    if idx_min < 0 or idx_max >= n_pad:
        # Range first: the pad-slot gather below would itself index out of
        # bounds (the very async fault this guard exists to preempt).
        raise ValueError(f"VSA-H3 untile index is not an injective map into non-pad slots: range "
                         f"[{idx_min}, {idx_max}] vs padded length {n_pad} (segments={segments}).")
    in_tile_offset = untile_combined_index % tile_elems
    maps_into_pad = bool((in_tile_offset >= variable_block_sizes[untile_combined_index // tile_elems]).any())
    if maps_into_pad or int(torch.unique(untile_combined_index).numel()) != total:
        raise ValueError(f"VSA-H3 untile index is not an injective map into non-pad slots: "
                         f"pad-slot hit={maps_into_pad} (segments={segments}).")


# Where a grid axis that is not a multiple of its tile puts the shortfall: ``last`` (one short tile at the end, the
# upstream VSA layout and the default), ``first`` (one short tile at the start, e.g. a lone image-conditioned latent
# frame) or ``center`` (the deficit spread one row at a time over the middle tiles, so no tile is much shorter than
# the rest). The partition is metadata only: every tile is still padded to the tile volume for the kernels.
VSA_H3_AXIS_POLICIES = ("last", "first", "center")
_DEFAULT_AXIS_POLICY = ("last", "last", "last")


def _axis_tile_sizes(length: int, tile: int, policy: str) -> tuple[int, ...]:
    """Token counts of the ``ceil(length / tile)`` tiles along one axis under ``policy``."""
    num_tiles = math.ceil(length / tile)
    deficit = num_tiles * tile - length
    sizes = [tile] * num_tiles
    if deficit == 0:
        return tuple(sizes)
    if policy == "last":
        sizes[-1] = tile - deficit
    elif policy == "first":
        sizes[0] = tile - deficit
    elif policy == "center":
        base, extra = divmod(deficit, num_tiles)
        start = (num_tiles - extra) // 2
        sizes = [tile - base - int(start <= index < start + extra) for index in range(num_tiles)]
    else:
        raise ValueError(f"VSA-H3 axis policy must be one of {VSA_H3_AXIS_POLICIES}, got {policy!r}")
    return tuple(sizes)


def _policy_tile_partition(
    grid: tuple[int, int, int],
    tile_shape: tuple[int, int, int],
    axis_policy: tuple[str, str, str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raster-to-tile permutation and per-tile token counts of one video region under per-axis policies.

    Tiles are enumerated t-major with w fastest and each tile's rows stay in raster order, as
    ``get_tile_partition_indices`` and ``construct_variable_block_sizes`` lay out the ``last`` policy.
    """
    sizes = [
        _axis_tile_sizes(length, tile, policy)
        for length, tile, policy in zip(grid, tile_shape, axis_policy, strict=True)
    ]
    bounds = [[0, *itertools.accumulate(axis_sizes)] for axis_sizes in sizes]
    raster = torch.arange(math.prod(grid), device=device, dtype=torch.long).reshape(grid)
    parts = [
        raster[t0:t1, h0:h1, w0:w1].flatten() for t0, t1 in itertools.pairwise(bounds[0])
        for h0, h1 in itertools.pairwise(bounds[1]) for w0, w1 in itertools.pairwise(bounds[2])
    ]
    t_sizes, h_sizes, w_sizes = (torch.tensor(axis_sizes, dtype=torch.long, device=device) for axis_sizes in sizes)
    block_sizes = (t_sizes[:, None, None] * h_sizes[None, :, None] * w_sizes[None, None, :]).reshape(-1)
    return torch.cat(parts), block_sizes


@functools.lru_cache(maxsize=10)
def _h3_segment_tile_geometry(
    segments: tuple[int | tuple[int, int, int], ...],
    device: torch.device,
    tile_shape: tuple[int, int, int] = VSA_H3_TILE_SIZE,
    axis_policy: tuple[str, str, str] = _DEFAULT_AXIS_POLICY,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, tuple[tuple[int, int], ...]]:
    """Tile a packed sequence of interleaved dense segments and video regions.

    ``segments`` lists, in packed order, dense segment row counts (``int``)
    and video-region token grids (``(t, h, w)`` triples). Rows are permuted so
    every dense chunk lands in the leading prefix tiles and each video region
    contributes one contiguous run of 3-D tiles, wherever the dense segments
    sit in the packed sequence. Dense chunks never straddle a segment
    boundary. ``axis_policy`` places each video axis's partial tile
    (``VSA_H3_AXIS_POLICIES``); the default is the upstream layout.

    Returns (tile_partition_indices, variable_block_sizes,
    untile_combined_index, num_prefix_tiles, num_video_tiles,
    video_tile_spans), where ``video_tile_spans`` holds one ``[start, end)``
    tile range per video region (prefix tiles included in the numbering).
    """
    tile_elems = math.prod(tile_shape)

    prefix_sizes: list[int] = []
    prefix_rows: list[torch.Tensor] = []
    video_entries: list[tuple[int, tuple[int, int, int]]] = []
    cursor = 0
    for segment in segments:
        if isinstance(segment, tuple):
            video_entries.append((cursor, segment))
            cursor += math.prod(segment)
            continue
        full, rem = divmod(segment, tile_elems)
        prefix_sizes.extend([tile_elems] * full)
        if rem:
            prefix_sizes.append(rem)
        prefix_rows.append(torch.arange(cursor, cursor + segment, device=device, dtype=torch.long))
        cursor += segment
    num_prefix_tiles = len(prefix_sizes)

    ts_t, ts_h, ts_w = tile_shape
    video_index_parts: list[torch.Tensor] = []
    video_size_parts: list[torch.Tensor] = []
    video_tile_spans: list[tuple[int, int]] = []
    tile_cursor = num_prefix_tiles
    for start, grid in video_entries:
        t, h, w = grid
        if axis_policy == _DEFAULT_AXIS_POLICY:
            num_tiles = (math.ceil(t / ts_t), math.ceil(h / ts_h), math.ceil(w / ts_w))
            region_sizes = construct_variable_block_sizes(grid, num_tiles, device, tile_shape)
            region_index = get_tile_partition_indices(grid, tile_shape, device)
        else:
            region_index, region_sizes = _policy_tile_partition(grid, tile_shape, axis_policy, device)
        video_index_parts.append(region_index + start)
        video_size_parts.append(region_sizes)
        video_tile_spans.append((tile_cursor, tile_cursor + int(region_sizes.numel())))
        tile_cursor += int(region_sizes.numel())
    num_video_tiles = tile_cursor - num_prefix_tiles

    tile_partition_indices = torch.cat(prefix_rows + video_index_parts)
    # cat promotes the int32 helper output to int64 alongside the prefix sizes
    variable_block_sizes = torch.cat([torch.tensor(prefix_sizes, dtype=torch.long, device=device)] + video_size_parts)

    # get_non_pad_index is lru-cached on tensor identity; variable_block_sizes
    # is itself cached by this function, so the identity stays stable.
    non_pad_index = get_non_pad_index(variable_block_sizes, tile_elems)

    untile_combined_index = non_pad_index[torch.argsort(tile_partition_indices)]
    # One-time (lru-cached) synchronous bounds check; see _validate_h3_segment_geometry.
    _validate_h3_segment_geometry(segments, variable_block_sizes, untile_combined_index, tile_elems)
    return (tile_partition_indices, variable_block_sizes, untile_combined_index, num_prefix_tiles, num_video_tiles,
            tuple(video_tile_spans))


# The sparse-attention algorithms this backend runs. ``vsa`` is VSA-H3 as described above. ``vsa2`` is VSA2 (Zhang et
# al., "Improving Video Sparse Attention with Fine-grained Router and Sparse Rebasing", ICML 2026) on the fused
# training kernels of fastvideo-kernel/vsa2_train (sm_100a / sm_103a): a gradient-free router over 2 x 2 x 2 windows
# that keeps, per head, the N_v * topk video tile pairs it scores highest plus every diagonal pair; a coarse branch
# over every block's pooled rows, the dense prefix included; the fine branch; and the gated merge, in one call. Its
# kernels fix their own layout (``_vsa2_segment_geometry``): uniform 4 x 4 x 8 tiles padded at the end of each axis.
VSA_H3_VARIANTS = ("vsa", "vsa2")
_VSA2_TILE_SHAPE = VSA_H3_TILE_SHAPES[128]


def _vsa2_row_valid(num_prefix_tiles: int, dense_rows: int, extent: tuple[int, int, int],
                    device: torch.device) -> torch.Tensor:
    """[(P + N_v) * 128] bool: the rows of VSA2's layout that hold a token, as its kernels derive them from the
    geometry alone (vsa2_train/reference.py, ``row_valid``)."""
    rows = math.prod(_VSA2_TILE_SHAPE)
    bt, bh, bw = _VSA2_TILE_SHAPE
    nt, nh, nw = (math.ceil(n / b) for n, b in zip(extent, _VSA2_TILE_SHAPE, strict=True))
    tile = torch.arange(nt * nh * nw, device=device)
    slot = torch.arange(rows, device=device)
    origin = torch.stack([tile // (nh * nw) * bt, tile // nw % nh * bh, tile % nw * bw], -1)
    local = torch.stack([slot // (bh * bw), slot // bw % bh, slot % bw], -1)
    video = ((origin[:, None, :] + local[None, :, :]) < torch.tensor(extent, device=device)).all(-1).reshape(-1)
    return torch.cat([torch.arange(num_prefix_tiles * rows, device=device) < dense_rows, video])


@functools.lru_cache(maxsize=10)
def _vsa2_segment_geometry(
    segments: tuple[int | tuple[int, int, int], ...],
    device: torch.device,
    reverse_time: bool,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """VSA2's layout of a packed sequence of dense segments and one video region (vsa2_train/reference.py).

    The dense segments' rows come first, in packed order, as one run cut into 128-row blocks, the last one padded at
    its end; then the video's 4 x 4 x 8 tiles in raster order of the tile grid, every token at its own slot of its
    tile (raster order of the tile), and the latent padded up to whole tiles at the end of each axis. Unlike VSA-H3's
    layouts, a partial tile keeps its padding where the latent's edge cuts it instead of packing its tokens first.
    ``reverse_time`` enumerates the frames last to first, so that the short tile along time holds the first frames
    (VSA-H3's ``first`` policy, which keeps an image-conditioned first latent frame in a tile of its own) while the
    kernels pad the end of the axis: attention does not depend on the order of the tiles or of a tile's rows, only on
    the partition into tiles and router windows.

    Returns (variable_block_sizes, untile_combined_index, num_prefix_tiles, num_video_tiles): each block's token
    count, and the row of the tiled sequence of each packed row.
    """
    rows = math.prod(_VSA2_TILE_SHAPE)
    videos = [segment for segment in segments if isinstance(segment, tuple)]
    if len(videos) != 1:
        raise ValueError(f"VSA2 tiles exactly one video region, got {len(videos)}.")
    extent = videos[0]
    t, h, w = extent
    bt, bh, bw = _VSA2_TILE_SHAPE
    nt, nh, nw = (math.ceil(n / b) for n, b in zip(extent, _VSA2_TILE_SHAPE, strict=True))
    dense_rows = sum(segment for segment in segments if not isinstance(segment, tuple))
    num_prefix_tiles = math.ceil(dense_rows / rows)
    num_video_tiles = nt * nh * nw

    ti = torch.arange(t, device=device, dtype=torch.long)
    if reverse_time:
        ti = t - 1 - ti
    hi = torch.arange(h, device=device, dtype=torch.long)
    wi = torch.arange(w, device=device, dtype=torch.long)
    tile = ((ti // bt)[:, None, None] * nh + (hi // bh)[None, :, None]) * nw + (wi // bw)[None, None, :]
    slot = ((ti % bt)[:, None, None] * bh + (hi % bh)[None, :, None]) * bw + (wi % bw)[None, None, :]
    video_rows = ((num_prefix_tiles + tile) * rows + slot).reshape(-1)
    parts: list[torch.Tensor] = []
    dense_cursor = 0
    for segment in segments:
        if isinstance(segment, tuple):
            parts.append(video_rows)
        else:
            parts.append(torch.arange(dense_cursor, dense_cursor + segment, device=device, dtype=torch.long))
            dense_cursor += segment
    untile_combined_index = torch.cat(parts)

    full, rem = divmod(dense_rows, rows)
    prefix_sizes = torch.tensor([rows] * full + ([rem] if rem else []), dtype=torch.long, device=device)
    ct, ch, cw = (torch.tensor([min(b, n - b * i) for i in range(math.ceil(n / b))], dtype=torch.long, device=device)
                  for n, b in zip(extent, _VSA2_TILE_SHAPE, strict=True))
    video_sizes = (ct[:, None, None] * ch[None, :, None] * cw[None, None, :]).reshape(-1)
    variable_block_sizes = torch.cat([prefix_sizes, video_sizes])

    # One-time (lru-cached) synchronous check: the packed rows land one to one on the rows the kernels take as tokens.
    valid = _vsa2_row_valid(num_prefix_tiles, dense_rows, extent, device)
    hit = torch.zeros_like(valid)
    hit[untile_combined_index] = True
    if (untile_combined_index.numel() != dense_rows + t * h * w or not torch.equal(hit, valid)
            or int(valid.sum()) != untile_combined_index.numel()):
        raise ValueError(f"VSA2 geometry for {segments} does not map the packed rows one to one onto its token rows.")
    return variable_block_sizes, untile_combined_index, num_prefix_tiles, num_video_tiles


# The GPUs the fused VSA2 kernels are built for: data-center Blackwell (B200 / GB200 at sm_100a, B300 at sm_103a). They
# use tcgen05 tensor memory, TMA and cluster launch control, so a build for any other architecture cannot run them.
_VSA2_CAPABILITIES = ((10, 0), (10, 3))


def require_vsa2_device(device_index: int | None = None) -> None:
    """Raise a clear error unless the CUDA device (default: the current one) can run the VSA2 kernels."""
    if not torch.cuda.is_available():
        raise RuntimeError("VSA2 sparse attention (vsa_variant='vsa2', e.g. FastPAN2) needs a CUDA sm_100a / sm_103a "
                           "GPU (B200, GB200, B300); no CUDA device is available.")
    _require_vsa2_device(torch.cuda.current_device() if device_index is None else device_index)


@functools.cache
def _require_vsa2_device(device_index: int) -> None:
    """Raise a clear error when the CUDA device cannot run the VSA2 kernels, instead of letting them build and trap."""
    capability = torch.cuda.get_device_capability(device_index)
    if capability not in _VSA2_CAPABILITIES:
        name = torch.cuda.get_device_name(device_index)
        raise RuntimeError(f"VSA2 sparse attention (vsa_variant='vsa2', e.g. FastPAN2) runs only on sm_100a / sm_103a "
                           f"GPUs (B200, GB200, B300); {name} is sm_{capability[0]}{capability[1]}.")


@functools.cache
def _vsa2_attention() -> Any:
    """``vsa2_attention`` of fastvideo-kernel/vsa2_train (``FASTVIDEO_VSA2_TRAIN_DIR``, by default this checkout's),
    whose extension builds on first use (see its ``_module``). The module registers its custom ops on import, so it is
    loaded once per process, as ``vsa2_train``, which is also how its own tests import it."""
    module = sys.modules.get("vsa2_train")
    if module is None:
        root = envs.FASTVIDEO_VSA2_TRAIN_DIR.get() or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "fastvideo-kernel", "vsa2_train")
        path = os.path.join(root, "vsa2_train.py")
        if not os.path.isfile(path):
            raise RuntimeError(
                f"VSA2 needs fastvideo-kernel/vsa2_train; {path} does not exist (FASTVIDEO_VSA2_TRAIN_DIR).")
        spec = importlib.util.spec_from_file_location("vsa2_train", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["vsa2_train"] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules["vsa2_train"]
            raise
    return module.vsa2_attention


class MiniMaxH3VSABackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "VIDEO_SPARSE_ATTN_H3"

    @staticmethod
    def get_impl_cls() -> type["MiniMaxH3VSAImpl"]:
        return MiniMaxH3VSAImpl

    @staticmethod
    def get_metadata_cls() -> type["MiniMaxH3VSAMetadata"]:
        return MiniMaxH3VSAMetadata

    @staticmethod
    def get_builder_cls() -> type["MiniMaxH3VSAMetadataBuilder"]:
        return MiniMaxH3VSAMetadataBuilder


class _MiniMaxH3VSATileBufferHolder:
    """Builder-owned no-grad tile scratch and its active geometry."""

    def __init__(self) -> None:
        self.buffer: torch.Tensor | None = None
        self.untile_geometry: torch.Tensor | None = None
        # FASTVIDEO_H3_VSA_HEADS_FIRST_TILE: a [B, H, S_pad, D] buffer whose
        # transpose is the tile buffer, and the geometry it was written for.
        self.heads_first_buffer: torch.Tensor | None = None
        self.heads_first_geometry: torch.Tensor | None = None


@dataclass
class MiniMaxH3VSAMetadata(AttentionMetadata):
    total_seq_length: int
    num_prefix_tiles: int
    num_video_tiles: int
    exempt: bool
    variable_block_sizes: torch.Tensor
    untile_combined_index: torch.Tensor
    # Device-side copy of ``dense_layers``. Regional fullgraph capture uses
    # this tensor with each implementation's tensor-valued layer index so the
    # shared block code does not specialize once per Python ``layer_idx``.
    dense_layers_tensor: torch.Tensor
    # One [start, end) tile range per sparse video region, in packed order
    # with the generated video last, and the sparsity of that region's top-k.
    video_tile_spans: tuple[tuple[int, int], ...]
    span_sparsities: tuple[float, ...]
    # tokens per tile (256, 128 or 64); selects the tile geometry AND the
    # kernel route in forward() (256 -> VSA-256 CuTe/Triton, 128 ->
    # ptx_blackwell, 64 -> native Triton or opt-in ptx_blackwell)
    block_mask_plan: torch.Tensor = field(init=False, repr=False)
    pad_tokens: int = 0
    ref2va_policy: str | None = None
    reference_video_regions: int = 0
    tile_elems: int = _TILE_ELEMS
    # layers forced dense regardless of sparsity (probe-guided opt-outs)
    dense_layers: tuple[int, ...] = ()
    # Builder-owned padded tile buffer. It records the geometry that last
    # populated the allocation so a same-shaped geometry change can clear
    # stale pad rows once while steady-state denoising reuses the buffer.
    tile_buf_holder: _MiniMaxH3VSATileBufferHolder | None = None
    # Whether the pooled compression branch also attends to the dense prefix
    # tiles. False keeps the prefix (PAN2's text, which the fine branch
    # already attends exactly) out of the compression keys; the prefix rows'
    # own compression output is unaffected because their gate is zero.
    compress_prefix_keys: bool = True
    # The sparse-attention algorithm (``VSA_H3_VARIANTS``). For ``vsa2`` the
    # rows are VSA2's layout and ``vsa2_thw`` / ``vsa2_text`` are the video's
    # extent in tokens and the dense rows its kernels take.
    variant: str = "vsa"
    vsa2_thw: tuple[int, int, int] = (0, 0, 0)
    vsa2_text: int = 0
    vsa2_topk: int = 0
    # Scatter rows straight into the heads-first layout the 64/128-token kernels read, as
    # ``FASTVIDEO_H3_VSA_HEADS_FIRST_TILE`` does process-wide: bit-identical, no-grad uncompiled CUDA only. A pipeline
    # turns it on for its own sampling (PAN2: ``PAN2PipelineConfig.vsa_heads_first_tile``).
    heads_first_tile: bool = False

    def __post_init__(self) -> None:
        if type(self.pad_tokens) is not int or self.pad_tokens < 0:
            raise ValueError("VSA-H3 pad_tokens must be a non-negative integer")
        # Host control data is consumed inside an opaque, non-differentiable
        # top-k op during training compilation, avoiding per-span value guards.
        self.block_mask_plan = torch.tensor(
            [(self.num_prefix_tiles, lo, hi, sparsity)
             for (lo, hi), sparsity in zip(self.video_tile_spans, self.span_sparsities, strict=True)],
            dtype=torch.float64,
            device="cpu").reshape(-1, 4)


class MiniMaxH3VSAMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self) -> None:
        self._tile_buf_holder = _MiniMaxH3VSATileBufferHolder()

    def prepare(self) -> None:
        pass

    def build(  # type: ignore
        self,
        current_timestep: int,
        patch_size: tuple[int, int, int],
        VSA_sparsity: float,
        packed_segments: tuple[int | tuple[int, int, int], ...],
        device: torch.device,
        exempt: bool = True,
        dense_layers: tuple[int, ...] = (),
        tile_size: int = _TILE_ELEMS,
        ref_keep_rate: float | None = None,
        pad_tokens: int = 0,
        video_tile_policy: tuple[str, str, str] | None = None,
        compress_prefix_keys: bool = True,
        variant: str = "vsa",
        heads_first_tile: bool = False,
        **kwargs: dict[str, Any],
    ) -> MiniMaxH3VSAMetadata:
        """Build per-step metadata for one packed H3 sequence.

        ``packed_segments`` lists the sequence in packed order: an ``int`` is
        a dense segment's row count, and a ``(t, h, w)`` triple is the raw
        latent shape of one sparse video region. The last region is the
        generated video and follows ``VSA_sparsity``; every earlier region is
        a reference video and keeps ``ref_keep_rate`` of its tiles, which a
        build with reference-video regions requires. ``exempt=False``
        (compete mode) supports only the generated-video region.
        ``video_tile_policy`` places each video axis's partial tile (one of
        ``VSA_H3_AXIS_POLICIES`` per ``(t, h, w)`` axis; default ``last``).
        ``compress_prefix_keys`` is described on :class:`MiniMaxH3VSAMetadata`. ``variant`` picks the algorithm
        (``VSA_H3_VARIANTS``): ``vsa2`` needs 128-token tiles, one video
        region, exempt prefix keys that also enter the compression branch, and
        a ``video_tile_policy`` of ``last`` or ``first`` along time and
        ``last`` along height and width, which its kernels' layout can hold.
        """
        if type(pad_tokens) is not int or pad_tokens < 0:
            raise ValueError("VSA-H3 pad_tokens must be a non-negative integer")
        axis_policy = _DEFAULT_AXIS_POLICY if video_tile_policy is None else tuple(str(p) for p in video_tile_policy)
        if len(axis_policy) != 3 or any(policy not in VSA_H3_AXIS_POLICIES for policy in axis_policy):
            raise ValueError(f"VSA-H3 video_tile_policy must give one of {VSA_H3_AXIS_POLICIES} per (t, h, w) axis, "
                             f"got {video_tile_policy!r}")
        if variant not in VSA_H3_VARIANTS:
            raise ValueError(f"VSA-H3 variant must be one of {VSA_H3_VARIANTS}, got {variant!r}")
        if variant == "vsa2":
            if int(tile_size) != 128:
                raise ValueError(f"VSA2 runs 128-token (4, 4, 8) tiles; got tile_size={tile_size!r}")
            if axis_policy[0] not in ("last", "first") or axis_policy[1:] != ("last", "last"):
                raise ValueError("VSA2's kernels pad the end of each axis: video_tile_policy must be last or first "
                                 f"along time and last along height and width, got {axis_policy}")
            if ref_keep_rate is not None or not exempt:
                raise ValueError("VSA2 has one video region and exempt prefix keys: ref_keep_rate and compete mode "
                                 "are VSA-H3 only")
            if not compress_prefix_keys:
                raise ValueError("VSA2's compression branch pools every block, the dense prefix included; "
                                 "compress_prefix_keys=False is VSA-H3 only")
        if ref_keep_rate is not None and not 0.0 < float(ref_keep_rate) <= 1.0:
            raise ValueError("VSA-H3 ref_keep_rate must be in (0, 1]")
        tile_shape = VSA_H3_TILE_SHAPES.get(int(tile_size))
        if tile_shape is None:
            raise ValueError(f"VSA-H3 tile_size must be one of {sorted(VSA_H3_TILE_SHAPES)}, got {tile_size!r}")
        # Video regions become token grids under the patch size; empty dense segments are dropped.
        token_segments: list[int | tuple[int, int, int]] = []
        for segment in packed_segments:
            if isinstance(segment, tuple):
                if any(int(v) <= 0 or int(v) % p for v, p in zip(segment, patch_size, strict=True)):
                    raise ValueError(f"VSA-H3 video region latent shape {tuple(segment)} is not a positive multiple "
                                     f"of patch {tuple(patch_size)}.")
                t, h, w = (int(v) // p for v, p in zip(segment, patch_size, strict=True))
                token_segments.append((t, h, w))
            elif segment > 0:
                token_segments.append(int(segment))
        num_regions = sum(isinstance(segment, tuple) for segment in token_segments)
        if num_regions == 0:
            raise ValueError("VSA-H3 needs at least one video region.")
        reference_sparsities: tuple[float, ...] = ()
        if num_regions > 1:
            if ref_keep_rate is None:
                raise ValueError(f"A VSA-H3 build with {num_regions - 1} reference-video region(s) needs "
                                 "ref_keep_rate.")
            if not exempt:
                raise ValueError(f"vsa_mode='compete' supports only the generated-video region; this build has "
                                 f"{num_regions - 1} reference-video region(s). Use vsa_mode='exempt'.")
            reference_sparsities = (1.0 - float(ref_keep_rate), ) * (num_regions - 1)
        total_seq_length = sum(math.prod(s) if isinstance(s, tuple) else s for s in token_segments)

        vsa2_thw, vsa2_text, vsa2_topk = (0, 0, 0), 0, 0
        video_tile_spans: tuple[tuple[int, int], ...]
        if variant == "vsa2":
            variable_block_sizes, untile_combined_index, num_prefix_tiles, num_video_tiles = _vsa2_segment_geometry(
                tuple(token_segments), device, axis_policy[0] == "first")
            video_tile_spans = ((num_prefix_tiles, num_prefix_tiles + num_video_tiles), )
            vsa2_thw = next(segment for segment in token_segments if isinstance(segment, tuple))
            vsa2_text = sum(segment for segment in token_segments if not isinstance(segment, tuple))
            # The video tiles each query tile keeps on average; a dense build keeps them all.
            vsa2_topk = compute_topk(max(float(VSA_sparsity), 0.0), num_video_tiles)
            if device.type == "cuda":
                _require_vsa2_device(device.index if device.index is not None else torch.cuda.current_device())
            # Register the kernels' custom ops before anything traces a forward (the extension builds on first call).
            _vsa2_attention()
        else:
            (_tile_partition_indices, variable_block_sizes, untile_combined_index, num_prefix_tiles, num_video_tiles,
             video_tile_spans) = _h3_segment_tile_geometry(tuple(token_segments), device, tile_shape, axis_policy)

        # A dense build (sparsity 0: dense steps) keeps every region dense,
        # whatever the reference keep rate.
        VSA_sparsity = float(VSA_sparsity)
        span_sparsities = (0.0, ) * num_regions if VSA_sparsity <= 0.0 else (*reference_sparsities, VSA_sparsity)

        dense_layers = tuple(int(layer) for layer in dense_layers)
        return MiniMaxH3VSAMetadata(
            current_timestep=current_timestep,
            VSA_sparsity=VSA_sparsity,
            total_seq_length=total_seq_length,
            pad_tokens=pad_tokens,
            ref2va_policy="p2_multi_region" if ref_keep_rate is not None else None,
            reference_video_regions=num_regions - 1,
            num_prefix_tiles=num_prefix_tiles,
            num_video_tiles=num_video_tiles,
            exempt=exempt,
            variable_block_sizes=variable_block_sizes,
            untile_combined_index=untile_combined_index,
            dense_layers_tensor=torch.tensor(dense_layers, device=device, dtype=torch.int64),
            video_tile_spans=video_tile_spans,
            span_sparsities=span_sparsities,
            tile_elems=int(tile_size),
            dense_layers=dense_layers,
            tile_buf_holder=self._tile_buf_holder,
            compress_prefix_keys=bool(compress_prefix_keys),
            variant=variant,
            vsa2_thw=vsa2_thw,
            vsa2_text=vsa2_text,
            vsa2_topk=vsa2_topk,
            heads_first_tile=bool(heads_first_tile),
        )


def assert_ref2va_vsa_metadata(
    metadata: MiniMaxH3VSAMetadata,
    *,
    expected_reference_video_regions: int,
    target_sparsity: float,
    ref_keep_rate: float | None,
) -> None:
    """Fail closed unless metadata represents the requested Ref2VA P2 policy.

    The last video span is the generated target and every preceding span is a
    reference video. Text, reference audio/images, and target audio are dense
    prefix tiles. Both modular training and inference validation call this
    after building metadata so a stale single-region/P1 layout cannot silently
    exempt reference-video conditioning from VSA.
    """
    expected_reference_video_regions = int(expected_reference_video_regions)
    if expected_reference_video_regions < 0:
        raise ValueError("expected_reference_video_regions must be non-negative")
    expected_span_count = expected_reference_video_regions + 1
    if len(metadata.video_tile_spans) != expected_span_count:
        raise ValueError("Ref2VA P2 metadata must contain one sparse span per reference video plus the target; "
                         f"expected {expected_span_count} spans ({expected_reference_video_regions} references), got "
                         f"{metadata.video_tile_spans}. A stale single-region layout would leave reference-video "
                         "conditioning dense.")
    if len(metadata.span_sparsities) != expected_span_count:
        raise ValueError("Ref2VA P2 metadata sparsity policy is not aligned with its video spans: "
                         f"{len(metadata.span_sparsities)} values for {expected_span_count} spans.")
    if metadata.ref2va_policy != "p2_multi_region" or metadata.reference_video_regions != expected_reference_video_regions:
        raise ValueError(
            "Ref2VA P2 metadata has no valid multi-region policy receipt: "
            f"policy={metadata.ref2va_policy!r}, reference_video_regions={metadata.reference_video_regions}, "
            f"expected {expected_reference_video_regions}.")

    cursor = metadata.num_prefix_tiles
    for index, (start, end) in enumerate(metadata.video_tile_spans):
        if start != cursor or end <= start:
            raise ValueError("Ref2VA P2 video tile spans must be positive and contiguous after the dense prefix; "
                             f"span {index} is {(start, end)} after cursor {cursor}.")
        cursor = end
    if cursor != metadata.num_prefix_tiles + metadata.num_video_tiles:
        raise ValueError("Ref2VA P2 video tile spans do not cover the complete sparse-video suffix: "
                         f"ended at {cursor}, expected {metadata.num_prefix_tiles + metadata.num_video_tiles}.")

    target_sparsity = float(target_sparsity)
    if target_sparsity <= 0.0:
        expected_sparsities = (0.0, ) * expected_span_count
    else:
        reference_sparsity = target_sparsity if ref_keep_rate is None else 1.0 - float(ref_keep_rate)
        if expected_reference_video_regions and reference_sparsity <= 0.0:
            raise ValueError("Ref2VA P2 requested sparse target attention but would leave reference-video conditioning "
                             f"dense (ref_keep_rate={ref_keep_rate!r}). Use a keep rate below 1.")
        expected_sparsities = ((reference_sparsity, ) * expected_reference_video_regions + (target_sparsity, ))
    if any(not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
           for actual, expected in zip(metadata.span_sparsities, expected_sparsities, strict=True)):
        raise ValueError("Ref2VA P2 metadata does not match the requested per-region sparsity policy: "
                         f"got {metadata.span_sparsities}, expected {expected_sparsities}.")


def _pool_tiles(x: torch.Tensor, variable_block_sizes: torch.Tensor, tile_elems: int = _TILE_ELEMS) -> torch.Tensor:
    """fp32 mean over each tile_elems-token tile. x: [B, S_pad, H, D] -> [B, H, n_tiles, D].

    Pad positions in the tile buffer are guaranteed zero (zeros-init, never
    written), so a plain sum with fp32 accumulation needs no validity mask
    and no materialized fp32 temp; dividing by the true tile size makes it
    the masked mean exactly.
    """
    batch, seq_len, heads, dim = x.shape
    n_tiles = seq_len // tile_elems
    pooled = x.view(batch, n_tiles, tile_elems, heads, dim).sum(dim=2, dtype=torch.float32)
    pooled = pooled / variable_block_sizes.view(1, -1, 1, 1)
    return pooled.permute(0, 2, 1, 3)


def _compression_scores(scores: torch.Tensor, attn_metadata: MiniMaxH3VSAMetadata) -> torch.Tensor:
    """The pooled scores the compression branch softmaxes: without the prefix columns unless the metadata keeps them.

    Excluding them keeps rows that the fine branch already attends exactly (the dense prefix) from being counted a
    second time through their pooled means. Selection always uses the unmodified ``scores``.
    """
    num_prefix_tiles = attn_metadata.num_prefix_tiles
    if attn_metadata.compress_prefix_keys or num_prefix_tiles == 0:
        return scores
    prefix_columns = torch.arange(scores.shape[-1], device=scores.device) < num_prefix_tiles
    return scores.masked_fill(prefix_columns, float("-inf"))


def _build_block_mask(
    scores: torch.Tensor,
    num_prefix_tiles: int,
    VSA_sparsity: float,
    exempt: bool,
    video_tile_spans: tuple[tuple[int, int], ...],
    span_sparsities: tuple[float, ...],
) -> torch.Tensor:
    """scores: [B, H, n_tiles, n_tiles] -> bool mask, same shape.

    Prefix rows (non-video queries) stay dense. With ``exempt``, every query
    keeps every prefix column, and every video query keeps its own top-k of
    EACH video region, with the budget from that region's ``span_sparsities``
    entry. Without ``exempt`` (compete mode, generated-video region only),
    prefix and video columns compete for one FLOP-matched top-k.
    ``VSA_sparsity <= 0`` (dense steps and layers) keeps every tile.
    ``video_tile_spans`` come from the cached geometry, which lays them out
    contiguously after the prefix tiles, with one ``span_sparsities`` entry
    each.
    """
    n_tiles = scores.shape[-1]
    if VSA_sparsity <= 0.0:
        span_sparsities = tuple(VSA_sparsity for _ in video_tile_spans)
    span_topk = [
        compute_topk(sparsity, end - start)
        for (start, end), sparsity in zip(video_tile_spans, span_sparsities, strict=True)
    ]
    if all(k == end - start for k, (start, end) in zip(span_topk, video_tile_spans, strict=True)):
        return torch.ones_like(scores, dtype=torch.bool)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if exempt or num_prefix_tiles == 0:
        for (start, end), k_span in zip(video_tile_spans, span_topk, strict=True):
            if k_span == end - start:
                mask[..., start:end] = True
                continue
            idx = scores[..., start:end].topk(k_span, dim=-1).indices + start
            mask.scatter_(-1, idx, True)
        mask[..., :num_prefix_tiles] = True
    else:
        # compete: prefix keys enter the top-k under a FLOP-matched budget,
        # which is defined for the generated-video region alone (the builder
        # rejects compete mode with reference-video regions)
        k_total = min(span_topk[0] + num_prefix_tiles, n_tiles)
        idx = scores.topk(k_total, dim=-1).indices
        mask.scatter_(-1, idx, True)
    mask[:, :, :num_prefix_tiles, :] = True
    return mask


@torch.library.custom_op("fastvideo::h3_region_topk_mask", mutates_args=())
def h3_region_topk_mask(scores: torch.Tensor, plan: torch.Tensor, sparsity: float, exempt: bool) -> torch.Tensor:
    """Apply the unchanged per-region top-k without specializing on host lengths."""
    rows = plan.tolist()
    if not rows or any(row[0] != rows[0][0] for row in rows):
        raise ValueError("H3 sparse selection requires one common dense prefix")
    spans = tuple((int(row[1]), int(row[2])) for row in rows)
    return _build_block_mask(scores, int(rows[0][0]), sparsity, exempt, spans, tuple(row[3] for row in rows))


@h3_region_topk_mask.register_fake
def _h3_region_topk_mask_fake(scores: torch.Tensor, plan: torch.Tensor, sparsity: float, exempt: bool) -> torch.Tensor:
    return torch.empty_like(scores, dtype=torch.bool)


def _ptx_blackwell_unavailable_reason(ptx_blackwell_mod: Any, query_bhsd: torch.Tensor,
                                      variable_block_sizes: torch.Tensor, grad_mode: bool) -> str | None:
    """Why the opt-in data-center Blackwell route cannot run here, or None if it can.

    Pure decision logic, split out so the routing is unit-testable without a
    GPU or the compiled extension (tests substitute ``ptx_blackwell_mod``). Order
    matters only for the message: the cheapest, most actionable reason first.
    """
    if ptx_blackwell_mod is None:
        return "fastvideo_kernel.block_sparse_attn_ptx_blackwell is not installed"
    if grad_mode:
        return "inputs require grad; use the registered training operator"
    if not ptx_blackwell_mod.is_supported(query_bhsd, variable_block_sizes):
        return (
            "block_sparse_attn_ptx_blackwell.is_supported returned False (needs an sm_100 or sm_103 device, a built "
            "extension, bf16, head_dim 128, an even tile count, and integer tile sizes)")
    return None


def _is_datacenter_blackwell(device: torch.device) -> bool:
    """sm_100/sm_103 (B200, GB200, B300), the devices the ptx_blackwell kernel targets."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] == 10


@functools.cache
def _fine_attention_from_name(name: str) -> Any:
    kind, _, mode = name.partition("-")
    modes = {"emulator": ("mix8", "mixmx", "qk", "bf16")}
    if kind not in modes or mode not in modes[kind]:
        accepted = ", ".join(f"{prefix}-{value}" for prefix, values in modes.items() for value in values)
        raise ValueError(f"Invalid FASTVIDEO_H3_VSA_FINE_ATTENTION={name!r}; accepted names: {accepted}")
    from fastvideo.attention.backends.fp4_vsa_qat import FP4AttentionNumerics, FP4VSAFineAttention
    if mode == "bf16":
        return FP4VSAFineAttention(FP4AttentionNumerics(quantize=False))
    prefix = {"mix8": "fp8", "mixmx": "mxfp8", "qk": None}[mode]
    return FP4VSAFineAttention(
        FP4AttentionNumerics(quantize=True, quantize_pv=False, smooth_k=True, prefix_precision=prefix))


def fine_attention_from_env() -> Any:
    """The ``FASTVIDEO_H3_VSA_FINE_ATTENTION`` replacement of the tile-128 fine kernel (shared), or None."""
    name = envs.FASTVIDEO_H3_VSA_FINE_ATTENTION.get()
    return _fine_attention_from_name(name) if name else None


class MiniMaxH3VSAImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.prefix = prefix
        self.layer_idx = layer_idx_from_prefix(prefix, default=-1)
        self.head_size = head_size
        self._sm89_kernel = envs.FASTVIDEO_H3_VSA_SM89_KERNEL.get()
        if self._sm89_kernel not in {"original", "bf16", "int8"}:
            raise ValueError("FASTVIDEO_H3_VSA_SM89_KERNEL must be original, bf16, or int8")
        # Generic torch.compile must not specialize the shared VSA forward on
        # the Python ``layer_idx`` value of each of H3's 50 blocks. This
        # tensor is prepared after weights load and drives only the compiled
        # dense-layer decision; it does not opt the module into ptx_blackwell.
        self._compile_layer_idx: torch.Tensor | None = None
        # None means the regional-compile preparation hook has not run.  The
        # eager path deliberately ignores this cache and preserves its
        # request-time env/probe/fallback behavior; only Dynamo capture reads
        # the prepared, static route.
        self._regional_compile_ptx_blackwell_enabled: bool | None = None
        # Tile-128 route per (device, dtype, head size): None when the CUDA
        # kernel can run it, else why not. The kernel's predicate otherwise
        # depends only on the tile-128 buffer contract (contiguous BHSD,
        # 128-token blocks, an even tile count with the partner tile, integer
        # tile sizes), which every call meets, so it is evaluated once.
        self._tile128_route: dict[tuple[torch.device, torch.dtype, int], str | None] = {}
        # Optional replacement of the fine block-sparse kernel only (selection,
        # mask and the gated compression branch are unchanged). Called as
        # ``fn(query, key, value, mask, metadata) -> [B, S, H, D]`` on the
        # logical tile-ordered rows. Quantization-aware training installs its
        # FP4 emulator here; it is also the only grad-capable tile-128 route.
        self.fine_attention_override: Any = fine_attention_from_env()

    def prepare_for_compile(self, device: torch.device) -> None:
        """Tensorize per-layer state shared by every torch.compile route."""
        self._compile_layer_idx = torch.tensor(self.layer_idx, device=device, dtype=torch.int64)
        if self.fine_attention_override is not None:
            # forward() cannot run the emulator inside a traced graph, and logging there
            # would break fullgraph capture, so say it here.
            logger.warning_once("FASTVIDEO_H3_VSA_FINE_ATTENTION is ignored inside torch.compile; compiled "
                                "VSA-H3 layers run the native fine attention. Keep the DiT eager to use it.")

    def prepare_for_regional_compile(self, device: torch.device) -> str | None:
        """Resolve the inference-only ptx_blackwell route before fullgraph capture.

        The ordinary eager route probes the environment, extension, device,
        and tensor contract at every call so it can warn and fall back.  Those
        Python/device-capability checks are not safe inside a regional
        ``fullgraph=True`` block.  Probe one representative tile-64 input on
        the loaded model's device now, then let ``forward`` specialize on the
        resulting plain bool while Dynamo is compiling.
        """
        if self._compile_layer_idx is None:
            self.prepare_for_compile(device)
        requested = envs.FASTVIDEO_VSA_PTX_BLACKWELL.get()
        enabled = False
        reason = None if requested else f"{VSA_PTX_BLACKWELL_ENV}=1 is required for compile-safe VSA-H3 attention"
        if requested:
            if _ptx_blackwell is None:
                reason = "fastvideo_kernel.block_sparse_attn_ptx_blackwell is not installed"
            elif not _ptx_blackwell_has_compile_safe_mask_route(_ptx_blackwell):
                reason = ("neither a native block_sparse_attn_ptx_blackwell_from_mask entry nor the raw ptx_blackwell "
                          "kernel plus map_to_index compatibility route is installed")
            else:
                # Two 64-token blocks exercise the exact ptx_blackwell inference
                # specialization while keeping the one-time probe tiny.  The
                # kernel predicate checks extension presence, CUDA capability,
                # dtype/layout, head size, block size, and even block count
                # without reading metadata tensor contents.
                probe_query = torch.empty((1, 1, 128, self.head_size), device=device, dtype=torch.bfloat16)
                probe_block_sizes = torch.full((2, ), 64, device=device, dtype=torch.int32)
                reason = _ptx_blackwell_unavailable_reason(
                    _ptx_blackwell,
                    probe_query,
                    probe_block_sizes,
                    grad_mode=False,
                )
                enabled = reason is None

        self._regional_compile_ptx_blackwell_enabled = enabled
        if enabled:
            route = ("native fastvideo-kernel mask entry" if callable(
                getattr(_ptx_blackwell, "block_sparse_attn_ptx_blackwell_from_mask", None)) else
                     "FastVideo compatibility mask adapter")
            logger.info_once(f"VSA-H3 regional compile mask route: {route}")
        if requested and reason is not None:
            logger.warning_once(f"VSA-H3 regional compile is unavailable and will stay eager: {reason}")
        return reason

    def tile(self, x: torch.Tensor, attn_metadata: MiniMaxH3VSAMetadata) -> torch.Tensor:
        """Scatter rows into the padded tile buffer (pad positions stay zero).

        Without grad tracking the returned tensor aliases the builder-owned
        buffer; callers must consume it before the next ``tile()`` (both call
        sites in ``forward()`` read it immediately). A grad-tracking forward
        instead receives a fresh buffer and leaves the holder untouched, so
        the builder never retains autograd state across steps. Odd no-grad
        ptx_blackwell requests (tile 128 always, tile 64 when opted in) carry one
        additional all-zero tile internally, including the native training
        path; metadata and observable outputs retain the logical geometry.
        """
        compiling = torch.compiler.is_compiling()
        regional_compiling = compiling and self._regional_compile_ptx_blackwell_enabled is True
        if not regional_compiling and x.shape[1] != attn_metadata.total_seq_length + attn_metadata.pad_tokens:
            raise ValueError(f"VSA-H3 metadata was built for sequence length {attn_metadata.total_seq_length}, "
                             f"got {x.shape[1]}. A non-packed sequence (e.g. the token refiner) is "
                             "routed to the VSA-H3 backend; exclude it from the supported backends.")
        if not regional_compiling:
            # Capacity rows never enter tile geometry, pooling or sparse budgets.
            x = x[:, :attn_metadata.total_seq_length]
        n_tiles = attn_metadata.variable_block_sizes.numel()
        grad_mode = torch.is_grad_enabled() and x.requires_grad
        if regional_compiling:
            ptx_blackwell_requested = bool(self._regional_compile_ptx_blackwell_enabled)
        elif compiling:
            ptx_blackwell_requested = grad_mode and envs.FASTVIDEO_VSA_PTX_BLACKWELL.get()
        else:
            ptx_blackwell_requested = envs.FASTVIDEO_VSA_PTX_BLACKWELL.get()
        # Tile 128 has no route other than the ptx_blackwell CUDA kernel. VSA2's kernels take any tile count, so
        # they need no partner tile.
        ptx_blackwell_route = attn_metadata.variant == "vsa" and (attn_metadata.tile_elems == 128 or
                                                                  (attn_metadata.tile_elems == 64
                                                                   and ptx_blackwell_requested))
        if regional_compiling:
            # Pad arithmetically so odd and even tile counts share one graph.
            needs_ptx_blackwell_pair, kernel_tiles = False, n_tiles + n_tiles % 2
        else:
            needs_ptx_blackwell_pair = (n_tiles % 2 != 0 and ptx_blackwell_route
                                        and (not grad_mode or block_sparse_attn_ptx_blackwell_op is not None))
            kernel_tiles = n_tiles + int(needs_ptx_blackwell_pair)
        target_shape = (x.shape[0], kernel_tiles * attn_metadata.tile_elems, x.shape[-2], x.shape[-1])

        # A grad-tracking forward must not reuse the builder-owned buffer. The
        # holder outlives the training step -- one builder serves the whole run,
        # and every step's metadata references the same holder -- while every
        # VSA layer writes this one buffer in place. A graph-tracked tiled
        # tensor left on the holder therefore anchors the step's in-place
        # autograd edges, and through them the activations they saved, for the
        # rest of training. This is the VSA-H3 counterpart of the Wan tile-cache
        # OOM (#1423), which training fixed with ``vsa_cache_tile_buf=False``.
        if grad_mode or regional_compiling:
            return scatter_into_tile_buf(x, target_shape, attn_metadata.untile_combined_index, None)

        # ``untile_combined_index`` maps each packed row to a logical tile
        # slot. Different geometries can share one transport shape; clear a
        # reused allocation once when the mapping identity changes so no old
        # valid row can survive as padding.
        holder = attn_metadata.tile_buf_holder
        if holder is None:
            raise RuntimeError("VSA-H3 metadata has no builder-owned tile buffer holder")
        if (not compiling and attn_metadata.tile_elems in _PTX_BLACKWELL_TILE_ELEMS
                and (getattr(attn_metadata, "heads_first_tile", False) or envs.FASTVIDEO_H3_VSA_HEADS_FIRST_TILE.get())
                and supports_heads_first_scatter(x, attn_metadata.untile_combined_index)):
            return self._tile_heads_first(x, attn_metadata, holder, kernel_tiles, needs_ptx_blackwell_pair)
        buffer_matches = (holder.buffer is not None and holder.buffer.shape == target_shape
                          and holder.buffer.dtype == x.dtype and holder.buffer.device == x.device)
        if buffer_matches and holder.untile_geometry is not attn_metadata.untile_combined_index:
            holder.buffer.zero_()
        holder.buffer = scatter_into_tile_buf(x, target_shape, attn_metadata.untile_combined_index, holder.buffer)
        holder.untile_geometry = attn_metadata.untile_combined_index
        if needs_ptx_blackwell_pair:
            # A prior even geometry can reuse this allocation and may have
            # written the last tile as logical data.
            holder.buffer[:, n_tiles * attn_metadata.tile_elems:].zero_()
        return holder.buffer

    @staticmethod
    def _tile_heads_first(x: torch.Tensor, attn_metadata: MiniMaxH3VSAMetadata, holder: _MiniMaxH3VSATileBufferHolder,
                          kernel_tiles: int, needs_ptx_blackwell_pair: bool) -> torch.Tensor:
        """``tile()`` into a heads-first buffer; returns its ``[B, S_pad, H, D]`` view.

        The 64/128-token kernels transpose the tiled tensor to BHSD and make it
        contiguous before every call; written heads-first, that copy is free.
        Same aliasing and zero-padding contract as the BSHD buffer.
        """
        n_tiles = attn_metadata.variable_block_sizes.numel()
        shape = (x.shape[0], x.shape[2], kernel_tiles * attn_metadata.tile_elems, x.shape[3])
        buffer = holder.heads_first_buffer
        if buffer is None or buffer.shape != shape or buffer.dtype != x.dtype or buffer.device != x.device:
            buffer = torch.zeros(shape, device=x.device, dtype=x.dtype)
        elif holder.heads_first_geometry is not attn_metadata.untile_combined_index:
            buffer.zero_()
        scatter_rows_heads_first(x, attn_metadata.untile_combined_index, buffer)
        holder.heads_first_buffer = buffer
        holder.heads_first_geometry = attn_metadata.untile_combined_index
        if needs_ptx_blackwell_pair:
            buffer[:, :, n_tiles * attn_metadata.tile_elems:].zero_()
        return buffer.transpose(1, 2)

    def preprocess_qkv(self, qkv: torch.Tensor, attn_metadata: MiniMaxH3VSAMetadata) -> torch.Tensor:
        return self.tile(qkv, attn_metadata)

    def postprocess_output(self, output: torch.Tensor, attn_metadata: MiniMaxH3VSAMetadata) -> torch.Tensor:
        real_output = output[:, attn_metadata.untile_combined_index]
        return torch.nn.functional.pad(real_output, (0, 0, 0, 0, 0, attn_metadata.pad_tokens))

    def _forward_vsa2(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor | None,
        attn_metadata: MiniMaxH3VSAMetadata,
    ) -> torch.Tensor:
        """VSA2 on the fused vsa2_train kernels: the router, both branches and the gated merge in one call.

        The rows are VSA2's layout (``_vsa2_segment_geometry``), whose padding rows ``tile()`` leaves zero. The
        kernels read heads first, so query, key, value and gate go to ``[B, H, S, D]`` bf16 (no copy when ``tile()``
        wrote them heads first). A layer in ``dense_layers``, or a dense build, keeps every video tile pair; without
        a gate the compression branch is computed and multiplied by zero. The kernels are the custom op
        ``torch.ops.vsa2_train.forward`` (registered by the metadata build), so torch.compile, fullgraph included,
        holds them as one opaque call with symbolic geometry.
        """
        seq_len = attn_metadata.variable_block_sizes.numel() * attn_metadata.tile_elems
        for name, tensor in (("query", query), ("key", key), ("value", value), ("gate", gate_compress)):
            if tensor is not None and tensor.shape[1] != seq_len:
                raise ValueError(f"VSA2 tiled {name} has length {tensor.shape[1]}, expected {seq_len}.")
        if self.fine_attention_override is not None:
            raise ValueError("FASTVIDEO_H3_VSA_FINE_ATTENTION replaces VSA-H3's fine kernel; VSA2 fuses its own.")
        if not query.is_cuda:
            raise RuntimeError("VSA2 sparse attention runs only on CUDA sm_100a / sm_103a GPUs.")
        if not torch.compiler.is_compiling():
            _require_vsa2_device(query.device.index if query.device.index is not None else torch.cuda.current_device())
        topk = attn_metadata.vsa2_topk
        if attn_metadata.dense_layers and self.layer_idx in attn_metadata.dense_layers:
            topk = attn_metadata.num_video_tiles
        q, k, v = (x.transpose(1, 2).to(torch.bfloat16).contiguous() for x in (query, key, value))
        gate = (torch.zeros_like(q)
                if gate_compress is None else gate_compress.transpose(1, 2).to(torch.bfloat16).contiguous())
        if not torch.compiler.is_compiling():
            logger.info_once("MiniMax-H3 VSA2 forward: fused vsa2_train kernels")
        t, h, w = attn_metadata.vsa2_thw
        out, _, _ = torch.ops.vsa2_train.forward(q, k, v, gate, t, h, w, attn_metadata.vsa2_text, topk)
        return out.transpose(1, 2).to(query.dtype).contiguous()

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor | None,
        attn_metadata: MiniMaxH3VSAMetadata,
    ) -> torch.Tensor:
        compiling = torch.compiler.is_compiling()
        regional_compiling = compiling and self._regional_compile_ptx_blackwell_enabled is True
        if attn_metadata.variant == "vsa2":
            return self._forward_vsa2(query, key, value, gate_compress, attn_metadata)

        tile_elems = attn_metadata.tile_elems
        if regional_compiling and tile_elems != 64:
            raise RuntimeError("VSA-H3 regional fullgraph compile requires 64-token tiles; disable "
                               "inference_torch_compile for tile-128/256 runs.")
        if tile_elems == 64:
            if block_sparse_attn_64_bhsd is None:
                raise NotImplementedError("fastvideo_kernel.block_sparse_attn is not installed")
        elif tile_elems == 256 and block_sparse_attn_256_bshd is None:
            raise NotImplementedError("fastvideo_kernel.block_sparse_attn_256 is not installed")

        # Probe recording performs filesystem writes and host synchronizations,
        # so the loader keeps probe-enabled runs eager. Avoid even reading that
        # environment switch while Dynamo captures a regional full graph.
        # The metadata always describes the trained logical geometry.
        # ``tile()`` may append exactly one transport-only partner for an odd
        # tile-64/128 ptx_blackwell call. Keep score selection and the gate branch on
        # the logical prefix, and reject every other shape before a kernel sees it.
        n_tiles = attn_metadata.variable_block_sizes.numel()
        logical_seq_len = n_tiles * tile_elems
        if regional_compiling:
            pair_tiles = query.shape[1] // tile_elems - n_tiles
            has_ptx_blackwell_pair = True
        else:
            pair_pad_seq_len = logical_seq_len + tile_elems
            pair_pad_is_valid = tile_elems in _PTX_BLACKWELL_TILE_ELEMS and n_tiles % 2 != 0
            allowed_seq_lengths = (logical_seq_len, pair_pad_seq_len) if pair_pad_is_valid else (logical_seq_len, )
            if query.shape[1] not in allowed_seq_lengths:
                expected = (f"the logical length {logical_seq_len} or one ptx_blackwell partner tile "
                            f"({pair_pad_seq_len})" if pair_pad_is_valid else f"the logical length {logical_seq_len}")
                raise ValueError(f"VSA-H3 tiled query has length {query.shape[1]}, expected {expected}.")
            has_ptx_blackwell_pair = query.shape[1] == pair_pad_seq_len
            pair_tiles = int(has_ptx_blackwell_pair)
        for name, tensor in (("key", key), ("value", value)):
            if tensor.shape[1] != query.shape[1]:
                raise ValueError(f"VSA-H3 tiled {name} length {tensor.shape[1]} does not match query "
                                 f"length {query.shape[1]}.")
        if gate_compress is not None and gate_compress.shape[1] != query.shape[1]:
            raise ValueError(f"VSA-H3 tiled gate length {gate_compress.shape[1]} does not match query "
                             f"length {query.shape[1]}.")

        logical_query = query[:, :logical_seq_len]
        logical_key = key[:, :logical_seq_len]
        logical_value = value[:, :logical_seq_len]
        logical_gate = gate_compress[:, :logical_seq_len] if gate_compress is not None else None

        # Probe-guided per-layer opt-out: diffuse layers run dense (all-True
        # mask) while the rest keep the configured sparsity. During any
        # prepared capture, keep the layer decision tensor-valued so the 50
        # block instances reuse one graph instead of specializing on the
        # Python layer_idx attribute.
        force_dense = None
        compile_layer_idx = self._compile_layer_idx if compiling else None
        if compile_layer_idx is not None:
            force_dense = (attn_metadata.dense_layers_tensor == compile_layer_idx).any()
            layer_sparsity = attn_metadata.VSA_sparsity
        else:
            layer_sparsity = 0.0 if self.layer_idx in attn_metadata.dense_layers else attn_metadata.VSA_sparsity
        probe_dir = None if compiling else probe_enabled()

        scores = None
        if layer_sparsity > 0.0 or gate_compress is not None or probe_dir is not None:
            q_pooled = _pool_tiles(logical_query, attn_metadata.variable_block_sizes, tile_elems)
            k_pooled = _pool_tiles(logical_key, attn_metadata.variable_block_sizes, tile_elems)
            scores = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) / (query.shape[-1]**0.5)
            if probe_dir is not None:
                record_probe(probe_dir, self.layer_idx, logical_query, logical_key, scores, attn_metadata)

        if scores is None:
            mask = torch.ones(query.shape[0], query.shape[2], n_tiles, n_tiles, dtype=torch.bool, device=query.device)
        elif regional_compiling:
            mask = _h3_vsa_block_mask(scores)
        elif compiling and torch.is_grad_enabled():
            mask = h3_region_topk_mask(scores, attn_metadata.block_mask_plan, layer_sparsity, attn_metadata.exempt)
        else:
            mask = _build_block_mask(
                scores,
                attn_metadata.num_prefix_tiles,
                layer_sparsity,
                attn_metadata.exempt,
                attn_metadata.video_tile_spans,
                attn_metadata.span_sparsities,
            )
        if force_dense is not None:
            # A scalar bool tensor broadcasts over the block map. This exactly
            # preserves the eager dense-layer contract without a Python branch.
            mask = mask | force_dense

        if self.fine_attention_override is not None and not compiling:
            out = self.fine_attention_override(logical_query, logical_key, logical_value, mask, attn_metadata)
        elif tile_elems in _PTX_BLACKWELL_TILE_ELEMS:
            # Native 64/128-token path: the block map is already at the
            # kernels' granularity. These entries take BHSD ([B, H, S_pad, D]);
            # mirror block_sparse_attn_256_bshd's Triton branch and transpose
            # around the call.
            # One gate for the sm89 QK/PV route below: the kernels are
            # inference-only, BF16, head-128, and validated for Ada (8, 9).
            # ``sm89_strided`` marks the INT8-QK entry, the only one that reads
            # the BSHD strided views directly.
            sm89_route = (self._sm89_kernel != "original" and not torch.is_grad_enabled() and not compiling
                          and query.dtype == torch.bfloat16 and query.shape[-1] == 128 and query.is_cuda
                          and torch.cuda.get_device_capability(query.device) == (8, 9))
            sm89_strided = sm89_route and self._sm89_kernel == "int8"
            q_bhsd = query.transpose(1, 2)
            k_bhsd = key.transpose(1, 2)
            v_bhsd = value.transpose(1, 2)
            if not sm89_strided:
                q_bhsd, k_bhsd, v_bhsd = (t.contiguous() for t in (q_bhsd, k_bhsd, v_bhsd))

            ptx_blackwell_mask = mask
            ptx_blackwell_variable_block_sizes = attn_metadata.variable_block_sizes
            if has_ptx_blackwell_pair:
                # The synthetic tile is neither a logical query nor key. Its
                # all-False row yields q2k_num=0, the all-False column keeps it
                # out of real rows, and vbs=0 masks all of its key slots.
                ptx_blackwell_mask = torch.nn.functional.pad(mask, (0, pair_tiles, 0, pair_tiles), value=False)
                ptx_blackwell_variable_block_sizes = torch.nn.functional.pad(
                    attn_metadata.variable_block_sizes,
                    (0, pair_tiles),
                    value=0,
                )

            # ptx_blackwell forward, opt-in at tile 64 and the only route at
            # tile128. Training uses the registered autograd entry; inference
            # retains the established raw forward and regional-compile routes.
            grad_mode = torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad)
            use_ptx_blackwell = False
            split_tile128 = False
            force_triton = envs.FASTVIDEO_VSA_TRITON.get() if grad_mode else False
            if grad_mode and tile_elems == 128 and force_triton:
                raise RuntimeError("VSA-H3 tile128 training requires native forward and backward; Triton was forced")
            if grad_mode and (tile_elems == 128 or (envs.FASTVIDEO_VSA_PTX_BLACKWELL.get() and not force_triton)):
                # The public kernel supplies a registered autograd operator;
                # never call its raw forward-only entry while tracking grads.
                supported = (_ptx_blackwell is not None and block_sparse_attn_ptx_blackwell_op is not None
                             and _ptx_blackwell.is_supported(q_bhsd, ptx_blackwell_variable_block_sizes))
                if tile_elems == 128:
                    backward_supported = (_ptx_blackwell_backward_is_supported is not None
                                          and _ptx_blackwell_backward_is_supported(q_bhsd,
                                                                                   ptx_blackwell_variable_block_sizes))
                    if not supported or not backward_supported:
                        raise RuntimeError("VSA-H3 tile128 training requires the ptx_blackwell forward and backward")
                use_ptx_blackwell = supported
            elif tile_elems == 128:
                # The CUDA kernel's 128-token instantiation needs no opt-in; without it,
                # the Triton tile-64 kernels run each tile as two halves.
                if grad_mode:
                    raise NotImplementedError("VSA-H3 tile 128 supports no-grad inference forwards only.")
                route_key = (q_bhsd.device, q_bhsd.dtype, int(q_bhsd.shape[-1]))
                if route_key not in self._tile128_route:
                    reason = _ptx_blackwell_unavailable_reason(_ptx_blackwell, q_bhsd,
                                                               ptx_blackwell_variable_block_sizes, grad_mode)
                    if reason is None and map_to_index is None:
                        reason = "fastvideo_kernel.triton_kernels.index (map_to_index) is not importable"
                    if reason is not None and not compiling and _is_datacenter_blackwell(q_bhsd.device):
                        # These GPUs should have the native kernel; a broken build
                        # would otherwise only show up as a slow run.
                        logger.warning_once(f"VSA-H3 tile 128 on a data-center Blackwell GPU is falling back to "
                                            f"the slower Triton tile-64 kernels: {reason}.")
                    self._tile128_route[route_key] = reason
                reason = self._tile128_route[route_key]
                if reason is None:
                    use_ptx_blackwell = True
                elif block_sparse_attn_64_bhsd is None:
                    raise RuntimeError(f"VSA-H3 tile 128 requires the ptx_blackwell block-sparse kernel or "
                                       f"the Triton tile-64 kernels: {reason}.")
                else:
                    split_tile128 = True
            elif regional_compiling:
                if self._regional_compile_ptx_blackwell_enabled is None:
                    raise RuntimeError(
                        "VSA-H3 ptx_blackwell routing was not resolved before torch.compile; "
                        "call prepare_for_regional_compile(device) on every MiniMaxH3VSAImpl after loading weights.")
                # The preparation probe established module/device/kernel
                # support.  Keep only static tensor/geometry facts here; no
                # env access, device-capability query, or is_supported call may
                # enter the Dynamo graph.
                if not (self._regional_compile_ptx_blackwell_enabled and not grad_mode
                        and q_bhsd.dtype == torch.bfloat16 and q_bhsd.shape[-1] == 128
                        and ptx_blackwell_variable_block_sizes.numel() % 2 == 0):
                    raise RuntimeError(
                        "VSA-H3 regional fullgraph compile requires the prepared ptx_blackwell BF16/head-128 route "
                        "on a supported device; disable inference_torch_compile for this request.")
                use_ptx_blackwell = True
            elif not compiling and envs.FASTVIDEO_VSA_PTX_BLACKWELL.get():
                reason = _ptx_blackwell_unavailable_reason(_ptx_blackwell, q_bhsd, ptx_blackwell_variable_block_sizes,
                                                           grad_mode)
                if reason is None and map_to_index is None:
                    reason = "fastvideo_kernel.triton_kernels.index (map_to_index) is not importable"
                if reason is None:
                    use_ptx_blackwell = True
                else:
                    logger.warning_once(
                        f"{VSA_PTX_BLACKWELL_ENV}=1 but falling back to the Triton-64 kernels: {reason}")

            if use_ptx_blackwell:
                # prepare_for_regional_compile logs the selected mask route before
                # capture. Logging from this branch would itself break a
                # ``fullgraph=True`` forward.
                if not compiling:
                    logger.info_once(
                        f"MiniMax-H3 VSA tile-{tile_elems} forward: using the ptx_blackwell block-sparse kernel")
                if grad_mode:
                    q2k_idx, q2k_num = map_to_index(ptx_blackwell_mask)
                    out_bhsd, _ = block_sparse_attn_ptx_blackwell_op(q_bhsd, k_bhsd, v_bhsd, q2k_idx, q2k_num,
                                                                     ptx_blackwell_variable_block_sizes.to(torch.int32))
                elif regional_compiling:
                    # The compile-safe wrapper keeps both Triton mask
                    # compaction and the raw pybind launch behind one
                    # fake-backed custom-op boundary.
                    out_bhsd, _ = _ptx_blackwell_from_mask(
                        q_bhsd,
                        k_bhsd,
                        v_bhsd,
                        ptx_blackwell_mask,
                        ptx_blackwell_variable_block_sizes,
                    )
                else:
                    # Preserve the established eager/index-native route and
                    # compatibility with older kernel wheels. Per-row counts
                    # are non-uniform (prefix queries are dense; video queries
                    # run prefix+top-k); the kernel reads each query tile's own count.
                    q2k_idx, q2k_num = map_to_index(ptx_blackwell_mask)
                    out_bhsd, _ = _ptx_blackwell.block_sparse_attn_ptx_blackwell(
                        q_bhsd,
                        k_bhsd,
                        v_bhsd,
                        q2k_idx,
                        q2k_num,
                        ptx_blackwell_variable_block_sizes.to(torch.int32),
                        need_lse=False,
                    )
            else:
                if has_ptx_blackwell_pair:
                    q_bhsd = q_bhsd[:, :, :logical_seq_len]
                    k_bhsd = k_bhsd[:, :, :logical_seq_len]
                    v_bhsd = v_bhsd[:, :, :logical_seq_len]
                    if not sm89_strided:
                        q_bhsd, k_bhsd, v_bhsd = (t.contiguous() for t in (q_bhsd, k_bhsd, v_bhsd))
                if split_tile128:
                    # No sm_100a kernel on this device: run each 128-token tile as two
                    # contiguous 64-token halves on the Triton kernels. Every half inherits
                    # its tile's mask entries, and a partial tile's valid tokens fill the
                    # first half before the second, so the selection is unchanged.
                    logger.info_once("MiniMax-H3 VSA tile-128 forward: Triton tile-64 kernels on split tiles")
                    mask64 = mask.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
                    vbs = attn_metadata.variable_block_sizes
                    vbs64 = torch.stack((vbs.clamp(max=64), (vbs - 64).clamp(min=0)), dim=-1).flatten()
                    out_bhsd, _ = block_sparse_attn_64_bhsd(q_bhsd, k_bhsd, v_bhsd, mask64, vbs64)
                elif sm89_route:
                    from fastvideo.attention.backends.minimax_h3_sparse_int8 import sparse_sm89_attention
                    logger.info_once(f"MiniMax-H3 VSA tile-64 forward: sm89 {self._sm89_kernel} QK / BF16 PV")
                    out_bhsd = sparse_sm89_attention(q_bhsd,
                                                     k_bhsd,
                                                     v_bhsd,
                                                     mask,
                                                     attn_metadata.variable_block_sizes,
                                                     int8_qk=self._sm89_kernel == "int8")
                else:
                    out_bhsd, _ = block_sparse_attn_64_bhsd(
                        q_bhsd,
                        k_bhsd,
                        v_bhsd,
                        mask,
                        attn_metadata.variable_block_sizes,
                    )
            if has_ptx_blackwell_pair and use_ptx_blackwell:
                out_bhsd = out_bhsd[:, :, :logical_seq_len]
            out = out_bhsd.transpose(1, 2).contiguous()
            # Fine attention is complete. Release its layout copies before the
            # gated compression merge creates full-sequence temporaries.
            del q_bhsd, k_bhsd, v_bhsd, out_bhsd
        else:
            out, _ = block_sparse_attn_256_bshd(
                logical_query,
                logical_key,
                logical_value,
                mask,
                attn_metadata.variable_block_sizes,
            )

        if logical_gate is not None:
            # Wan-style compression branch: dense attention over pooled tiles,
            # broadcast to each tile's rows, scaled by the learned gate
            # (zero-initialized for H3 => branch contributes nothing until
            # finetuned; the model layer skips it entirely for all-zero gates).
            v_pooled = _pool_tiles(logical_value, attn_metadata.variable_block_sizes, tile_elems)
            out_c = torch.matmul(torch.softmax(_compression_scores(scores, attn_metadata), dim=-1),
                                 v_pooled)  # [B, H, n_tiles, D]
            out_c = out_c.permute(0, 2, 1, 3).to(out.dtype)  # [B, n_tiles, H, D]
            batch, seq_len, heads, dim = out.shape
            out_tiled = out.view(batch, n_tiles, tile_elems, heads, dim)
            gate_tiled = logical_gate.view(batch, n_tiles, tile_elems, heads, dim)
            out_c = out_c.unsqueeze(2)
            if out.requires_grad or gate_tiled.dtype != out.dtype:
                # Out-of-place: ``out`` is the tensor the attention kernel's
                # autograd node saved for its backward, so an in-place add here
                # bumps its version counter and backward dies with "one of the
                # variables needed for gradient computation has been modified".
                # Fused addcmul keeps one full-sequence temporary instead of two.
                out = torch.addcmul(out_tiled, out_c, gate_tiled).view(batch, seq_len, heads, dim)
            else:
                # Inference: ``out`` is a fresh kernel output nobody else holds,
                # so accumulate into it and allocate nothing.
                out_tiled.addcmul_(out_c, gate_tiled)
        return out
