# SPDX-License-Identifier: Apache-2.0
"""VSA for MiniMax H3's packed mixed-modality self-attention.

H3 runs one joint bidirectional attention over
``[text | condition keyframes | audio | generated video]``, so this
backend differs from the Wan-tuned ``video_sparse_attn``:

- Tiles are ``[segment-pure prefix chunks] + [3D video tiles]``; prefix
  tiles never straddle segment boundaries. The tile size is selectable at
  metadata build time: 256 tokens ``(4,8,8)`` (default) or 64 tokens
  ``(4,4,4)`` (see ``VSA_H3_TILE_SHAPES``).
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

At tile 256 this targets sm10.x through the FA4 CuTe 256-tile path
(``FASTVIDEO_VSA_CUTEDSL=1``); the Triton 256→64 expansion is the
fallback and keeps identical mask semantics. At tile 64 the block map is
already at the kernels' native 64-token granularity, so both forward and
backward run the Triton block-sparse kernels directly (no expansion,
``FASTVIDEO_VSA_CUTEDSL`` does not apply). A third, opt-in route exists
for the tile-64 FORWARD only: ``FASTVIDEO_VSA_SM100A=1`` sends no-grad
forwards through the sm_100a CUDA block-sparse kernel
(``fastvideo_kernel.block_sparse_attn_sm100a``, upstream PR #1719 plus
our per-q-tile ``q2k_num`` fix) when the extension is built, the device
is sm_100, and the geometry qualifies. The CUDA kernel assigns adjacent
pairs of query tiles to CTAs, so an odd logical tile count receives one
internal, zero-valid partner tile for the no-grad call only. Score search,
the trained mask, gate-compress, and the returned packed sequence remain on
the original logical tiles. Grad-tracking forwards and every backward stay
on Triton unchanged. If the env is set but a precondition fails, the route
logs one warning and falls back.
"""

import functools
import math
import os
from dataclasses import dataclass
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
    # Optional: only present in fastvideo_kernel builds that carry the sm_100a
    # CUDA block-sparse forward (upstream PR #1719). The module itself imports
    # fine without the compiled symbols (`_HAS_VSA_SM100A` is then False and
    # `is_supported` says no), so this only guards *module* availability.
    from fastvideo_kernel import block_sparse_attn_sm100a as _sm100a
except ImportError:
    _sm100a = None

from fastvideo.attention.backends.abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                                                   AttentionMetadataBuilder, layer_idx_from_prefix)
from fastvideo.attention.backends.video_sparse_attn import (compute_topk, construct_variable_block_sizes,
                                                            get_non_pad_index, get_tile_partition_indices,
                                                            scatter_into_tile_buf)
from fastvideo.attention.backends.video_sparse_attn_h3_probe import probe_enabled, record_probe
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Opt-in switch for the sm_100a CUDA forward on the tile-64 no-grad path.
VSA_SM100A_ENV = "FASTVIDEO_VSA_SM100A"

VSA_H3_TILE_SIZE = (4, 8, 8)  # 256 elements -> FA4 CuTe fastpath on sm10.x (default)
_TILE_ELEMS = math.prod(VSA_H3_TILE_SIZE)
# Selectable tile geometries, keyed by element count (= the build-time
# ``tile_size``). 64 runs the native 64-token Triton block-sparse kernels for
# forward AND backward — the block map is already at kernel granularity, so no
# 256->64 mask expansion is involved and FASTVIDEO_VSA_CUTEDSL does not apply.
VSA_H3_TILE_SHAPES: dict[int, tuple[int, int, int]] = {
    _TILE_ELEMS: VSA_H3_TILE_SIZE,
    64: (4, 4, 4),
}


@torch.library.custom_op(
    "fastvideo::h3_vsa_sm100a_from_mask_compat",
    mutates_args=(),
    device_types="cuda",
)
def _h3_vsa_sm100a_from_mask_compat(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> torch.Tensor:
    """Compile-safe mask adapter for kernel wheels predating the mask API."""
    if _sm100a is None or map_to_index is None:
        raise RuntimeError("The sm100a compatibility route requires the raw kernel and map_to_index")
    q2k_idx, q2k_num = map_to_index(block_map)
    out, _ = _sm100a.block_sparse_attn_sm100a(
        q,
        k,
        v,
        q2k_idx.to(torch.int32).contiguous(),
        q2k_num.to(torch.int32).contiguous(),
        variable_block_sizes.to(torch.int32).contiguous(),
        need_lse=False,
    )
    return out


@torch.library.register_fake("fastvideo::h3_vsa_sm100a_from_mask_compat")
def _h3_vsa_sm100a_from_mask_compat_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> torch.Tensor:
    del k, v, block_map, variable_block_sizes
    return torch.empty_like(q)


def _sm100a_has_compile_safe_mask_route(sm100a_mod: Any) -> bool:
    return (callable(getattr(sm100a_mod, "block_sparse_attn_sm100a_from_mask", None))
            or (callable(getattr(sm100a_mod, "block_sparse_attn_sm100a", None)) and map_to_index is not None))


def _sm100a_from_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> tuple[torch.Tensor, None]:
    """Use the native mask entry when installed, otherwise the local adapter."""
    native = getattr(_sm100a, "block_sparse_attn_sm100a_from_mask", None)
    if callable(native):
        return native(q, k, v, block_map, variable_block_sizes)
    return _h3_vsa_sm100a_from_mask_compat(q, k, v, block_map, variable_block_sizes), None


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


def _validate_h3_tile_geometry(
    prefix_segments: tuple[int, ...],
    dit_seq_shape: tuple[int, int, int],
    variable_block_sizes: torch.Tensor,
    untile_combined_index: torch.Tensor,
    tile_elems: int = _TILE_ELEMS,
) -> None:
    """Fail synchronously on out-of-bounds tile geometry.

    Invariants the block-sparse kernel trusts without checking:
    every tile's valid size is in (0, tile_elems]; the sizes sum to the
    packed sequence length; and ``untile_combined_index`` maps each packed
    row to exactly one non-pad slot of the padded tile buffer. A violation
    would surface only as an async device fault at some later kernel or
    collective (e.g. an FSDP all-gather), which is unattributable — so raise
    here, once per cached geometry, with the numbers in hand.
    """
    total = sum(prefix_segments) + math.prod(dit_seq_shape)
    n_pad = variable_block_sizes.numel() * tile_elems
    sizes_min = int(variable_block_sizes.min())
    sizes_max = int(variable_block_sizes.max())
    sizes_sum = int(variable_block_sizes.sum())
    if sizes_min < 1 or sizes_max > tile_elems or sizes_sum != total:
        raise ValueError(f"VSA-H3 tile sizes out of bounds for prefix={prefix_segments}, video={dit_seq_shape}, "
                         f"tile_elems={tile_elems}: min={sizes_min}, max={sizes_max}, sum={sizes_sum}, "
                         f"expected sum={total}.")
    if untile_combined_index.numel() != total:
        raise ValueError(f"VSA-H3 untile index has {untile_combined_index.numel()} entries for a packed "
                         f"sequence of {total} rows (prefix={prefix_segments}, video={dit_seq_shape}).")
    idx_min = int(untile_combined_index.min())
    idx_max = int(untile_combined_index.max())
    if idx_min < 0 or idx_max >= n_pad:
        # Range first: the pad-slot gather below would itself index out of
        # bounds (the very async fault this guard exists to preempt).
        raise ValueError(f"VSA-H3 untile index is not an injective map into non-pad slots: range "
                         f"[{idx_min}, {idx_max}] vs padded length {n_pad} "
                         f"(prefix={prefix_segments}, video={dit_seq_shape}).")
    in_tile_offset = untile_combined_index % tile_elems
    maps_into_pad = bool((in_tile_offset >= variable_block_sizes[untile_combined_index // tile_elems]).any())
    if maps_into_pad or int(torch.unique(untile_combined_index).numel()) != total:
        raise ValueError(f"VSA-H3 untile index is not an injective map into non-pad slots: "
                         f"pad-slot hit={maps_into_pad} "
                         f"(prefix={prefix_segments}, video={dit_seq_shape}).")


@functools.lru_cache(maxsize=10)
def _h3_tile_geometry(
    prefix_segments: tuple[int, ...],
    dit_seq_shape: tuple[int, int, int],
    device: torch.device,
    tile_shape: tuple[int, int, int] = VSA_H3_TILE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Tile the packed sequence: segment-pure prefix chunks, then video tiles.

    Returns (tile_partition_indices, variable_block_sizes,
    untile_combined_index, num_prefix_tiles, num_video_tiles).
    """
    tile_elems = math.prod(tile_shape)
    prefix_len = sum(prefix_segments)

    prefix_sizes: list[int] = []
    for segment in prefix_segments:
        full, rem = divmod(segment, tile_elems)
        prefix_sizes.extend([tile_elems] * full)
        if rem:
            prefix_sizes.append(rem)
    num_prefix_tiles = len(prefix_sizes)

    ts_t, ts_h, ts_w = tile_shape
    t, h, w = dit_seq_shape
    num_tiles = (math.ceil(t / ts_t), math.ceil(h / ts_h), math.ceil(w / ts_w))
    video_sizes = construct_variable_block_sizes(dit_seq_shape, num_tiles, device, tile_shape)
    num_video_tiles = int(video_sizes.numel())

    video_indices = get_tile_partition_indices(dit_seq_shape, tile_shape, device) + prefix_len
    tile_partition_indices = torch.cat([
        torch.arange(prefix_len, device=device, dtype=torch.long),
        video_indices,
    ])
    # cat promotes the int32 helper output to int64 alongside the prefix sizes
    variable_block_sizes = torch.cat([
        torch.tensor(prefix_sizes, dtype=torch.long, device=device),
        video_sizes,
    ])

    # get_non_pad_index is lru-cached on tensor identity; variable_block_sizes
    # is itself cached by this function, so the identity stays stable.
    non_pad_index = get_non_pad_index(variable_block_sizes, tile_elems)

    untile_combined_index = non_pad_index[torch.argsort(tile_partition_indices)]
    # One-time (lru-cached) synchronous bounds check; see _validate_h3_tile_geometry.
    _validate_h3_tile_geometry(prefix_segments, dit_seq_shape, variable_block_sizes, untile_combined_index, tile_elems)
    return (tile_partition_indices, variable_block_sizes, untile_combined_index, num_prefix_tiles, num_video_tiles)


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
    # tokens per tile (256 or 64); selects the tile geometry AND the kernel
    # route in forward() (256 -> VSA-256 CuTe/Triton, 64 -> native Triton)
    tile_elems: int = _TILE_ELEMS
    # layers forced dense regardless of sparsity (probe-guided opt-outs)
    dense_layers: tuple[int, ...] = ()
    # Builder-owned padded tile buffer. It records the geometry that last
    # populated the allocation so a same-shaped geometry change can clear
    # stale pad rows once while steady-state denoising reuses the buffer.
    tile_buf_holder: _MiniMaxH3VSATileBufferHolder | None = None


class MiniMaxH3VSAMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self) -> None:
        self._tile_buf_holder = _MiniMaxH3VSATileBufferHolder()

    def prepare(self) -> None:
        pass

    def build(  # type: ignore
        self,
        current_timestep: int,
        raw_latent_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        VSA_sparsity: float,
        prefix_segments: tuple[int, ...],
        device: torch.device,
        exempt: bool = True,
        dense_layers: tuple[int, ...] = (),
        tile_size: int = _TILE_ELEMS,
        **kwargs: dict[str, Any],
    ) -> MiniMaxH3VSAMetadata:
        tile_shape = VSA_H3_TILE_SHAPES.get(int(tile_size))
        if tile_shape is None:
            raise ValueError(f"VSA-H3 tile_size must be one of {sorted(VSA_H3_TILE_SHAPES)}, got {tile_size!r}")
        dit_seq_shape = (raw_latent_shape[0] // patch_size[0], raw_latent_shape[1] // patch_size[1],
                         raw_latent_shape[2] // patch_size[2])
        prefix_segments = tuple(int(s) for s in prefix_segments if s > 0)
        total_seq_length = sum(prefix_segments) + math.prod(dit_seq_shape)

        (_tile_partition_indices, variable_block_sizes, untile_combined_index, num_prefix_tiles,
         num_video_tiles) = _h3_tile_geometry(prefix_segments, dit_seq_shape, device, tile_shape)

        dense_layers = tuple(int(layer) for layer in dense_layers)
        return MiniMaxH3VSAMetadata(
            current_timestep=current_timestep,
            VSA_sparsity=VSA_sparsity,
            total_seq_length=total_seq_length,
            num_prefix_tiles=num_prefix_tiles,
            num_video_tiles=num_video_tiles,
            exempt=exempt,
            variable_block_sizes=variable_block_sizes,
            untile_combined_index=untile_combined_index,
            tile_elems=int(tile_size),
            dense_layers=dense_layers,
            dense_layers_tensor=torch.tensor(dense_layers, device=device, dtype=torch.int64),
            tile_buf_holder=self._tile_buf_holder,
        )


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


def _build_block_mask(
    scores: torch.Tensor,
    num_prefix_tiles: int,
    num_video_tiles: int,
    VSA_sparsity: float,
    exempt: bool,
) -> torch.Tensor:
    """scores: [B, H, n_tiles, n_tiles] -> bool mask, same shape."""
    n_tiles = scores.shape[-1]
    k_vid = compute_topk(VSA_sparsity, num_video_tiles)
    if k_vid == num_video_tiles:
        return torch.ones_like(scores, dtype=torch.bool)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if exempt or num_prefix_tiles == 0:
        video_cols = scores[..., num_prefix_tiles:]
        idx = video_cols.topk(k_vid, dim=-1).indices + num_prefix_tiles
        mask.scatter_(-1, idx, True)
        mask[..., :num_prefix_tiles] = True
    else:
        k_total = min(k_vid + num_prefix_tiles, n_tiles)
        idx = scores.topk(k_total, dim=-1).indices
        mask.scatter_(-1, idx, True)
    mask[:, :, :num_prefix_tiles, :] = True
    return mask


def _sm100a_unavailable_reason(sm100a_mod: Any, query_bhsd: torch.Tensor, variable_block_sizes: torch.Tensor,
                               grad_mode: bool) -> str | None:
    """Why the opt-in sm_100a forward route cannot run here, or None if it can.

    Pure decision logic, split out so the routing is unit-testable without a
    GPU or the compiled extension (tests substitute ``sm100a_mod``). Order
    matters only for the message: the cheapest, most actionable reason first.
    """
    if sm100a_mod is None:
        return "fastvideo_kernel.block_sparse_attn_sm100a is not installed"
    if grad_mode:
        return "inputs require grad and the sm_100a kernel is forward-only; grad paths keep Triton"
    if not sm100a_mod.is_supported(query_bhsd, variable_block_sizes):
        return ("block_sparse_attn_sm100a.is_supported returned False (needs an sm_100 device, a built "
                "extension, bf16, head_dim 128, an even tile count, and integer tile sizes)")
    return None


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
        # None means the regional-compile preparation hook has not run.  The
        # eager path deliberately ignores this cache and preserves its
        # request-time env/probe/fallback behavior; only Dynamo capture reads
        # the prepared, static route.
        self._regional_compile_sm100a_enabled: bool | None = None
        self._regional_compile_layer_idx: torch.Tensor | None = None

    def prepare_for_regional_compile(self, device: torch.device) -> str | None:
        """Resolve the inference-only sm_100a route before fullgraph capture.

        The ordinary eager route probes the environment, extension, device,
        and tensor contract at every call so it can warn and fall back.  Those
        Python/device-capability checks are not safe inside a regional
        ``fullgraph=True`` block.  Probe one representative tile-64 input on
        the loaded model's device now, then let ``forward`` specialize on the
        resulting plain bool while Dynamo is compiling.
        """
        requested = os.environ.get(VSA_SM100A_ENV, "0") == "1"
        enabled = False
        reason = None if requested else f"{VSA_SM100A_ENV}=1 is required for compile-safe VSA-H3 attention"
        if requested:
            if _sm100a is None:
                reason = "fastvideo_kernel.block_sparse_attn_sm100a is not installed"
            elif not _sm100a_has_compile_safe_mask_route(_sm100a):
                reason = ("neither a native block_sparse_attn_sm100a_from_mask entry nor the raw sm100a "
                          "kernel plus map_to_index compatibility route is installed")
            else:
                # Two 64-token blocks exercise the exact sm_100a inference
                # specialization while keeping the one-time probe tiny.  The
                # kernel predicate checks extension presence, CUDA capability,
                # dtype/layout, head size, block size, and even block count
                # without reading metadata tensor contents.
                probe_query = torch.empty((1, 1, 128, self.head_size), device=device, dtype=torch.bfloat16)
                probe_block_sizes = torch.full((2, ), 64, device=device, dtype=torch.int32)
                reason = _sm100a_unavailable_reason(
                    _sm100a,
                    probe_query,
                    probe_block_sizes,
                    grad_mode=False,
                )
                enabled = reason is None

        self._regional_compile_sm100a_enabled = enabled
        # Keep this marker unset when preparation fails. Generic/training
        # torch.compile must retain the established Triton attention route.
        self._regional_compile_layer_idx = (torch.tensor(self.layer_idx, device=device, dtype=torch.int64)
                                            if enabled else None)
        if enabled:
            route = ("native fastvideo-kernel mask entry" if callable(
                getattr(_sm100a, "block_sparse_attn_sm100a_from_mask", None)) else
                     "FastVideo compatibility mask adapter")
            logger.info_once(f"VSA-H3 regional compile mask route: {route}")
        if requested and reason is not None:
            logger.warning_once(f"VSA-H3 regional compile is unavailable and will stay eager: {reason}")
        return reason

    def tile(self, x: torch.Tensor, attn_metadata: MiniMaxH3VSAMetadata) -> torch.Tensor:
        """Scatter rows into the padded tile buffer (pad positions stay zero).

        The returned tensor aliases the builder-owned buffer; callers must
        consume it before the next ``tile()`` (both call sites in
        ``forward()`` read it immediately). Odd tile-64 no-grad sm100a
        requests carry one additional all-zero tile internally; metadata and
        all observable outputs retain the logical geometry.
        """
        if x.shape[1] != attn_metadata.total_seq_length:
            raise ValueError(f"VSA-H3 metadata was built for sequence length {attn_metadata.total_seq_length}, "
                             f"got {x.shape[1]}. A non-packed sequence (e.g. the token refiner) is "
                             "routed to the VSA-H3 backend; exclude it from the supported backends.")
        n_tiles = attn_metadata.variable_block_sizes.numel()
        grad_mode = torch.is_grad_enabled() and x.requires_grad
        compiling = torch.compiler.is_compiling()
        regional_compiling = compiling and self._regional_compile_layer_idx is not None
        if regional_compiling:
            sm100a_requested = bool(self._regional_compile_sm100a_enabled)
        elif compiling:
            # Training/generic compile keeps the long-standing Triton route.
            sm100a_requested = False
        else:
            sm100a_requested = os.environ.get(VSA_SM100A_ENV, "0") == "1"
        needs_sm100a_pair = (attn_metadata.tile_elems == 64 and n_tiles % 2 != 0 and not grad_mode and sm100a_requested)
        kernel_tiles = n_tiles + int(needs_sm100a_pair)
        target_shape = (x.shape[0], kernel_tiles * attn_metadata.tile_elems, x.shape[-2], x.shape[-1])

        # ``untile_combined_index`` maps each packed row to a logical tile
        # slot. Different geometries can share one transport shape; clear a
        # reused allocation once when the mapping identity changes so no old
        # valid row can survive as padding.
        holder = attn_metadata.tile_buf_holder
        if holder is None:
            raise RuntimeError("VSA-H3 metadata has no builder-owned tile buffer holder")
        buffer_matches = (holder.buffer is not None and holder.buffer.shape == target_shape
                          and holder.buffer.dtype == x.dtype and holder.buffer.device == x.device)
        if buffer_matches and holder.untile_geometry is not attn_metadata.untile_combined_index:
            holder.buffer.zero_()
        holder.buffer = scatter_into_tile_buf(x, target_shape, attn_metadata.untile_combined_index, holder.buffer)
        holder.untile_geometry = attn_metadata.untile_combined_index
        if needs_sm100a_pair:
            # A prior even geometry can reuse this allocation and may have
            # written the last tile as logical data.
            holder.buffer[:, n_tiles * attn_metadata.tile_elems:].zero_()
        return holder.buffer

    def preprocess_qkv(self, qkv: torch.Tensor, attn_metadata: MiniMaxH3VSAMetadata) -> torch.Tensor:
        return self.tile(qkv, attn_metadata)

    def postprocess_output(self, output: torch.Tensor, attn_metadata: MiniMaxH3VSAMetadata) -> torch.Tensor:
        return output[:, attn_metadata.untile_combined_index]

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor | None,
        attn_metadata: MiniMaxH3VSAMetadata,
    ) -> torch.Tensor:
        compiling = torch.compiler.is_compiling()
        regional_compiling = compiling and self._regional_compile_layer_idx is not None

        tile_elems = attn_metadata.tile_elems
        if regional_compiling and tile_elems != 64:
            raise RuntimeError("VSA-H3 regional fullgraph compile requires 64-token tiles; disable "
                               "inference_torch_compile for tile-256/CuTe runs.")
        if tile_elems == 64:
            if block_sparse_attn_64_bhsd is None:
                raise NotImplementedError("fastvideo_kernel.block_sparse_attn is not installed")
        elif block_sparse_attn_256_bshd is None:
            raise NotImplementedError("fastvideo_kernel.block_sparse_attn_256 is not installed")

        # Probe recording performs filesystem writes and host synchronizations,
        # so the loader keeps probe-enabled runs eager. Avoid even reading that
        # environment switch while Dynamo captures a regional full graph.
        # The metadata always describes the trained logical geometry.
        # ``tile()`` may append exactly one transport-only partner for an odd
        # tile-64 sm100a call. Keep score selection and the gate branch on the
        # logical prefix, and reject every other shape before a kernel sees it.
        n_tiles = attn_metadata.variable_block_sizes.numel()
        logical_seq_len = n_tiles * tile_elems
        pair_pad_seq_len = logical_seq_len + tile_elems
        pair_pad_is_valid = tile_elems == 64 and n_tiles % 2 != 0
        allowed_seq_lengths = (logical_seq_len, pair_pad_seq_len) if pair_pad_is_valid else (logical_seq_len, )
        if query.shape[1] not in allowed_seq_lengths:
            expected = (f"the logical length {logical_seq_len} or one sm100a partner tile "
                        f"({pair_pad_seq_len})" if pair_pad_is_valid else f"the logical length {logical_seq_len}")
            raise ValueError(f"VSA-H3 tiled query has length {query.shape[1]}, expected {expected}.")
        has_sm100a_pair = query.shape[1] == pair_pad_seq_len
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
        # mask) while the rest keep the configured sparsity. During regional
        # capture, keep the layer decision tensor-valued so the 50 block
        # instances reuse one graph instead of specializing on layer_idx.
        force_dense = None
        if regional_compiling:
            assert self._regional_compile_layer_idx is not None
            force_dense = (attn_metadata.dense_layers_tensor == self._regional_compile_layer_idx).any()
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
        else:
            mask = _build_block_mask(
                scores,
                attn_metadata.num_prefix_tiles,
                attn_metadata.num_video_tiles,
                layer_sparsity,
                attn_metadata.exempt,
            )
        if force_dense is not None:
            # A scalar bool tensor broadcasts over the block map. This exactly
            # preserves the eager dense-layer contract without a Python branch.
            mask = mask | force_dense

        if tile_elems == 64:
            # Native 64-token path: the block map is already at the kernels'
            # granularity. Both 64-token entries take BHSD ([B, H, S_pad, D]);
            # mirror block_sparse_attn_256_bshd's Triton branch and transpose
            # around the call.
            q_bhsd = query.transpose(1, 2).contiguous()
            k_bhsd = key.transpose(1, 2).contiguous()
            v_bhsd = value.transpose(1, 2).contiguous()

            sm100a_mask = mask
            sm100a_variable_block_sizes = attn_metadata.variable_block_sizes
            if has_sm100a_pair:
                # The synthetic tile is neither a logical query nor key. Its
                # all-False row yields q2k_num=0, the all-False column keeps it
                # out of real rows, and vbs=0 masks all of its key slots.
                sm100a_mask = torch.nn.functional.pad(mask, (0, 1, 0, 1), value=False)
                sm100a_variable_block_sizes = torch.nn.functional.pad(
                    attn_metadata.variable_block_sizes,
                    (0, 1),
                    value=0,
                )

            # Opt-in sm_100a CUDA forward (upstream PR #1719 + per-q-tile
            # q2k_num fix). Forward-only: grad-tracking calls stay on Triton
            # so autograd keeps the Triton fwd+bwd pairing untouched. The
            # kernel does return an LSE in Triton's M format, so a future
            # fwd/bwd pairing is possible, but it is not built here.
            grad_mode = torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad)
            use_sm100a = False
            if regional_compiling:
                if self._regional_compile_sm100a_enabled is None:
                    raise RuntimeError(
                        "VSA-H3 sm_100a routing was not resolved before torch.compile; "
                        "call prepare_for_regional_compile(device) on every MiniMaxH3VSAImpl after loading weights.")
                # The preparation probe established module/device/kernel
                # support.  Keep only static tensor/geometry facts here; no
                # env access, device-capability query, or is_supported call may
                # enter the Dynamo graph.
                if not (self._regional_compile_sm100a_enabled and not grad_mode and q_bhsd.dtype == torch.bfloat16
                        and q_bhsd.shape[-1] == 128 and sm100a_variable_block_sizes.numel() % 2 == 0):
                    raise RuntimeError(
                        "VSA-H3 regional fullgraph compile requires the prepared sm_100a BF16/head-128 route "
                        "on a supported device; disable inference_torch_compile for this request.")
                use_sm100a = True
            elif not compiling and os.environ.get(VSA_SM100A_ENV, "0") == "1":
                reason = _sm100a_unavailable_reason(_sm100a, q_bhsd, sm100a_variable_block_sizes, grad_mode)
                if reason is None and map_to_index is None:
                    reason = "fastvideo_kernel.triton_kernels.index (map_to_index) is not importable"
                if reason is None:
                    use_sm100a = True
                else:
                    logger.warning_once(f"{VSA_SM100A_ENV}=1 but falling back to the Triton-64 kernels: {reason}")

            if use_sm100a:
                # Regional preparation emits the compile-route receipt before
                # capture. Logging from this branch would itself break a
                # ``fullgraph=True`` forward.
                if not compiling:
                    logger.info_once("MiniMax-H3 VSA tile-64 forward: using the sm100a CUDA block-sparse kernel")
                if regional_compiling:
                    # The compile-safe wrapper keeps both Triton mask
                    # compaction and the raw pybind launch behind one
                    # fake-backed custom-op boundary.
                    out_bhsd, _ = _sm100a_from_mask(
                        q_bhsd,
                        k_bhsd,
                        v_bhsd,
                        sm100a_mask,
                        sm100a_variable_block_sizes,
                    )
                else:
                    # Preserve the established eager/index-native route and
                    # compatibility with older kernel wheels. Per-row counts
                    # are non-uniform (prefix queries are dense; video queries
                    # run prefix+top-k), which the fixed kernel supports.
                    q2k_idx, q2k_num = map_to_index(sm100a_mask)
                    out_bhsd, _ = _sm100a.block_sparse_attn_sm100a(
                        q_bhsd,
                        k_bhsd,
                        v_bhsd,
                        q2k_idx,
                        q2k_num,
                        sm100a_variable_block_sizes.to(torch.int32),
                        need_lse=False,
                    )
            else:
                if has_sm100a_pair:
                    q_bhsd = q_bhsd[:, :, :logical_seq_len].contiguous()
                    k_bhsd = k_bhsd[:, :, :logical_seq_len].contiguous()
                    v_bhsd = v_bhsd[:, :, :logical_seq_len].contiguous()
                out_bhsd, _ = block_sparse_attn_64_bhsd(
                    q_bhsd,
                    k_bhsd,
                    v_bhsd,
                    mask,
                    attn_metadata.variable_block_sizes,
                )
            if has_sm100a_pair and use_sm100a:
                out_bhsd = out_bhsd[:, :, :logical_seq_len]
            out = out_bhsd.transpose(1, 2).contiguous()
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
            out_c = torch.matmul(torch.softmax(scores, dim=-1), v_pooled)  # [B, H, n_tiles, D]
            out_c = out_c.permute(0, 2, 1, 3).to(out.dtype)  # [B, n_tiles, H, D]
            batch, seq_len, heads, dim = out.shape
            # Out-of-place: on the CuTe backend ``out`` is the tensor FA4's
            # autograd node saved for its backward, so an in-place add here
            # bumps its version counter and backward dies with "one of the
            # variables needed for gradient computation has been modified".
            out_tiled = out.view(batch, n_tiles, tile_elems, heads, dim)
            gate_tiled = logical_gate.view(batch, n_tiles, tile_elems, heads, dim)
            out = (out_tiled + out_c.unsqueeze(2) * gate_tiled).view(batch, seq_len, heads, dim)
        return out
