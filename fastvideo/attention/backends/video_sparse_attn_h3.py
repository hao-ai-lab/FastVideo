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
is sm_100, and the geometry qualifies; grad-tracking forwards and every
backward stay on Triton unchanged. If the env is set but a precondition
fails, the route logs one warning and falls back.
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


@dataclass
class MiniMaxH3VSAMetadata(AttentionMetadata):
    total_seq_length: int
    num_prefix_tiles: int
    num_video_tiles: int
    exempt: bool
    variable_block_sizes: torch.Tensor
    untile_combined_index: torch.Tensor
    # tokens per tile (256 or 64); selects the tile geometry AND the kernel
    # route in forward() (256 -> VSA-256 CuTe/Triton, 64 -> native Triton)
    tile_elems: int = _TILE_ELEMS
    # layers forced dense regardless of sparsity (probe-guided opt-outs)
    dense_layers: tuple[int, ...] = ()
    # Single-slot holder for the padded tile buffer, owned by the BUILDER so
    # one buffer serves the whole denoising loop (pad slots stay zero and
    # every non-pad slot is fully overwritten per tile(), so cross-step reuse
    # is valid; saves a ~1.4 GB alloc+memset per step at 720p). VSA-H3 runs
    # eager today — revisit the reuse if it ever goes under cudagraphs.
    tile_buf_holder: list = None  # type: ignore[assignment]


class MiniMaxH3VSAMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self) -> None:
        self._tile_buf_holder: list = [None]

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
            dense_layers=tuple(int(layer) for layer in dense_layers),
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

    def tile(self, x: torch.Tensor, attn_metadata: MiniMaxH3VSAMetadata) -> torch.Tensor:
        """Scatter rows into the padded tile buffer (pad positions stay zero).

        The returned tensor aliases the builder-owned buffer; callers must
        consume it before the next ``tile()`` (both call sites in
        ``forward()`` read it immediately).
        """
        if x.shape[1] != attn_metadata.total_seq_length:
            raise ValueError(f"VSA-H3 metadata was built for sequence length {attn_metadata.total_seq_length}, "
                             f"got {x.shape[1]}. A non-packed sequence (e.g. the token refiner) is "
                             "routed to the VSA-H3 backend; exclude it from the supported backends.")
        n_tiles = attn_metadata.variable_block_sizes.numel()
        target_shape = (x.shape[0], n_tiles * attn_metadata.tile_elems, x.shape[-2], x.shape[-1])

        # single scatter: untile_combined_index maps original row i to its
        # padded slot, so this is exactly the inverse of postprocess_output
        holder = attn_metadata.tile_buf_holder
        holder[0] = scatter_into_tile_buf(x, target_shape, attn_metadata.untile_combined_index, holder[0])
        return holder[0]

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
        tile_elems = attn_metadata.tile_elems
        if tile_elems == 64:
            if block_sparse_attn_64_bhsd is None:
                raise NotImplementedError("fastvideo_kernel.block_sparse_attn is not installed")
        elif block_sparse_attn_256_bshd is None:
            raise NotImplementedError("fastvideo_kernel.block_sparse_attn_256 is not installed")

        # probe-guided per-layer opt-out: diffuse layers run dense (all-True
        # mask) while the rest keep the configured sparsity
        layer_sparsity = 0.0 if self.layer_idx in attn_metadata.dense_layers else attn_metadata.VSA_sparsity
        probe_dir = probe_enabled()

        scores = None
        if layer_sparsity > 0.0 or gate_compress is not None or probe_dir is not None:
            q_pooled = _pool_tiles(query, attn_metadata.variable_block_sizes, tile_elems)
            k_pooled = _pool_tiles(key, attn_metadata.variable_block_sizes, tile_elems)
            scores = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) / (query.shape[-1]**0.5)
            if probe_dir is not None:
                record_probe(probe_dir, self.layer_idx, query, key, scores, attn_metadata)

        if scores is None:
            n_tiles = attn_metadata.variable_block_sizes.numel()
            mask = torch.ones(query.shape[0], query.shape[2], n_tiles, n_tiles, dtype=torch.bool, device=query.device)
        else:
            mask = _build_block_mask(
                scores,
                attn_metadata.num_prefix_tiles,
                attn_metadata.num_video_tiles,
                layer_sparsity,
                attn_metadata.exempt,
            )

        if tile_elems == 64:
            # Native 64-token path: the block map is already at the kernels'
            # granularity. Both 64-token entries take BHSD ([B, H, S_pad, D]);
            # mirror block_sparse_attn_256_bshd's Triton branch and transpose
            # around the call.
            q_bhsd = query.transpose(1, 2).contiguous()
            k_bhsd = key.transpose(1, 2).contiguous()
            v_bhsd = value.transpose(1, 2).contiguous()

            # Opt-in sm_100a CUDA forward (upstream PR #1719 + per-q-tile
            # q2k_num fix). Forward-only: grad-tracking calls stay on Triton
            # so autograd keeps the Triton fwd+bwd pairing untouched. The
            # kernel does return an LSE in Triton's M format, so a future
            # fwd/bwd pairing is possible, but it is not built here.
            use_sm100a = False
            if os.environ.get(VSA_SM100A_ENV, "0") == "1":
                grad_mode = torch.is_grad_enabled() and (query.requires_grad or key.requires_grad
                                                         or value.requires_grad)
                reason = _sm100a_unavailable_reason(_sm100a, q_bhsd, attn_metadata.variable_block_sizes, grad_mode)
                if reason is None and map_to_index is None:
                    reason = "fastvideo_kernel.triton_kernels.index (map_to_index) is not importable"
                if reason is None:
                    use_sm100a = True
                elif not torch.compiler.is_compiling():
                    logger.warning_once(f"{VSA_SM100A_ENV}=1 but falling back to the Triton-64 kernels: {reason}")

            if use_sm100a:
                # The sm_100a entry is index-native; compact the bool map the
                # same way the Triton bool entry does internally. Per-row
                # counts are NON-uniform here (prefix query tiles are dense,
                # video tiles run prefix+top-k) -- legal for the fixed kernel,
                # silently wrong on the pre-fix upstream one.
                q2k_idx, q2k_num = map_to_index(mask)
                out_bhsd, _ = _sm100a.block_sparse_attn_sm100a(
                    q_bhsd,
                    k_bhsd,
                    v_bhsd,
                    q2k_idx,
                    q2k_num,
                    attn_metadata.variable_block_sizes.to(torch.int32),
                    need_lse=False,
                )
            else:
                out_bhsd, _ = block_sparse_attn_64_bhsd(
                    q_bhsd,
                    k_bhsd,
                    v_bhsd,
                    mask,
                    attn_metadata.variable_block_sizes,
                )
            out = out_bhsd.transpose(1, 2).contiguous()
        else:
            out, _ = block_sparse_attn_256_bshd(query, key, value, mask, attn_metadata.variable_block_sizes)

        if gate_compress is not None:
            # Wan-style compression branch: dense attention over pooled tiles,
            # broadcast to each tile's rows, scaled by the learned gate
            # (zero-initialized for H3 => branch contributes nothing until
            # finetuned; the model layer skips it entirely for all-zero gates).
            v_pooled = _pool_tiles(value, attn_metadata.variable_block_sizes, tile_elems)
            out_c = torch.matmul(torch.softmax(scores, dim=-1), v_pooled)  # [B, H, n_tiles, D]
            out_c = out_c.permute(0, 2, 1, 3).to(out.dtype)  # [B, n_tiles, H, D]
            batch, seq_len, heads, dim = out.shape
            n_tiles = attn_metadata.variable_block_sizes.numel()
            # Out-of-place: on the CuTe backend ``out`` is the tensor FA4's
            # autograd node saved for its backward, so an in-place add here
            # bumps its version counter and backward dies with "one of the
            # variables needed for gradient computation has been modified".
            out_tiled = out.view(batch, n_tiles, tile_elems, heads, dim)
            gate_tiled = gate_compress.view(batch, n_tiles, tile_elems, heads, dim)
            out = (out_tiled + out_c.unsqueeze(2) * gate_tiled).view(batch, seq_len, heads, dim)
        return out
