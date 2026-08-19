# SPDX-License-Identifier: Apache-2.0
"""VSA for MiniMax H3's packed mixed-modality self-attention.

H3 runs one joint bidirectional attention over
``[text | condition keyframes | audio | generated video]``, so this
backend differs from the Wan-tuned ``video_sparse_attn``:

- Tiles are ``[segment-pure prefix chunks] + [3D (4,8,8) video tiles]``;
  prefix tiles never straddle segment boundaries.
- Selection is pure Python on pooled tile scores; the block-sparse kernel
  consumes an explicit bool mask, so no kernel changes are needed.
- The compression branch is gated by ``to_gate_compress``, which the H3
  checkpoint does not carry: the loader zero-initializes it, so untrained
  inference is exactly pure sparse and finetuning can learn the gate.
- Non-video *queries* are always dense. Non-video *keys* are either
  always-selected for every query ("exempt", default) or compete in
  top-k under a FLOP-matched budget ("compete") — the ablation axis,
  switched per request via ``generate_video(..., vsa_mode=...)``
  (default: exempt). Per-request scheduling knobs
  (``vsa_dense_first_n_steps``, ``vsa_dense_layers``) let mixed schedules
  run the diffuse steps/layers dense while pushing the rest harder.

Targets sm10.x through the FA4 CuTe 256-tile path
(``FASTVIDEO_VSA_CUTEDSL=1``); the Triton 256→64 expansion is the
fallback and keeps identical mask semantics.
"""

import functools
import math
from dataclasses import dataclass
from typing import Any

import torch

try:
    from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_256_bshd
except ImportError:
    block_sparse_attn_256_bshd = None

from fastvideo.attention.backends.abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                                                   AttentionMetadataBuilder, layer_idx_from_prefix)
from fastvideo.attention.backends.video_sparse_attn import (compute_topk, construct_variable_block_sizes,
                                                            get_non_pad_index, get_tile_partition_indices,
                                                            scatter_into_tile_buf)
from fastvideo.attention.backends.video_sparse_attn_h3_probe import probe_enabled, record_probe

VSA_H3_TILE_SIZE = (4, 8, 8)  # 256 elements -> FA4 CuTe fastpath on sm10.x
_TILE_ELEMS = math.prod(VSA_H3_TILE_SIZE)


def token_tile_and_valid(variable_block_sizes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per padded-token tile id and pad-validity mask.

    The single encoding of the padding contract, shared by the probe and the
    test oracle so they cannot drift from the backend's tile geometry.
    """
    device = variable_block_sizes.device
    token_tile = torch.arange(variable_block_sizes.numel(), device=device).repeat_interleave(_TILE_ELEMS)
    token_valid = (torch.arange(_TILE_ELEMS, device=device)[None, :] < variable_block_sizes[:, None]).reshape(-1)
    return token_tile, token_valid


@functools.lru_cache(maxsize=10)
def _h3_tile_geometry(
    prefix_segments: tuple[int, ...],
    dit_seq_shape: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Tile the packed sequence: segment-pure prefix chunks, then video tiles.

    Returns (tile_partition_indices, variable_block_sizes,
    untile_combined_index, num_prefix_tiles, num_video_tiles).
    """
    prefix_len = sum(prefix_segments)

    prefix_sizes: list[int] = []
    for segment in prefix_segments:
        full, rem = divmod(segment, _TILE_ELEMS)
        prefix_sizes.extend([_TILE_ELEMS] * full)
        if rem:
            prefix_sizes.append(rem)
    num_prefix_tiles = len(prefix_sizes)

    ts_t, ts_h, ts_w = VSA_H3_TILE_SIZE
    t, h, w = dit_seq_shape
    num_tiles = (math.ceil(t / ts_t), math.ceil(h / ts_h), math.ceil(w / ts_w))
    video_sizes = construct_variable_block_sizes(dit_seq_shape, num_tiles, device, VSA_H3_TILE_SIZE)
    num_video_tiles = int(video_sizes.numel())

    video_indices = get_tile_partition_indices(dit_seq_shape, VSA_H3_TILE_SIZE, device) + prefix_len
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
    non_pad_index = get_non_pad_index(variable_block_sizes, _TILE_ELEMS)

    untile_combined_index = non_pad_index[torch.argsort(tile_partition_indices)]
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
            **kwargs: dict[str, Any],
    ) -> MiniMaxH3VSAMetadata:
        dit_seq_shape = (raw_latent_shape[0] // patch_size[0], raw_latent_shape[1] // patch_size[1],
                         raw_latent_shape[2] // patch_size[2])
        prefix_segments = tuple(int(s) for s in prefix_segments if s > 0)
        total_seq_length = sum(prefix_segments) + math.prod(dit_seq_shape)

        (_tile_partition_indices, variable_block_sizes, untile_combined_index, num_prefix_tiles,
         num_video_tiles) = _h3_tile_geometry(prefix_segments, dit_seq_shape, device)

        return MiniMaxH3VSAMetadata(
            current_timestep=current_timestep,
            VSA_sparsity=VSA_sparsity,
            total_seq_length=total_seq_length,
            num_prefix_tiles=num_prefix_tiles,
            num_video_tiles=num_video_tiles,
            exempt=exempt,
            variable_block_sizes=variable_block_sizes,
            untile_combined_index=untile_combined_index,
            dense_layers=tuple(int(layer) for layer in dense_layers),
            tile_buf_holder=self._tile_buf_holder,
        )


def _pool_tiles(x: torch.Tensor, variable_block_sizes: torch.Tensor) -> torch.Tensor:
    """fp32 mean over each 256-token tile. x: [B, S_pad, H, D] -> [B, H, n_tiles, D].

    Pad positions in the tile buffer are guaranteed zero (zeros-init, never
    written), so a plain sum with fp32 accumulation needs no validity mask
    and no materialized fp32 temp; dividing by the true tile size makes it
    the masked mean exactly.
    """
    batch, seq_len, heads, dim = x.shape
    n_tiles = seq_len // _TILE_ELEMS
    pooled = x.view(batch, n_tiles, _TILE_ELEMS, heads, dim).sum(dim=2, dtype=torch.float32)
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
        target_shape = (x.shape[0], n_tiles * _TILE_ELEMS, x.shape[-2], x.shape[-1])

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
        if block_sparse_attn_256_bshd is None:
            raise NotImplementedError("fastvideo_kernel.block_sparse_attn_256 is not installed")

        # probe-guided per-layer opt-out: diffuse layers run dense (all-True
        # mask) while the rest keep the configured sparsity
        layer_sparsity = 0.0 if self.layer_idx in attn_metadata.dense_layers else attn_metadata.VSA_sparsity
        probe_dir = probe_enabled()

        scores = None
        if layer_sparsity > 0.0 or gate_compress is not None or probe_dir is not None:
            q_pooled = _pool_tiles(query, attn_metadata.variable_block_sizes)
            k_pooled = _pool_tiles(key, attn_metadata.variable_block_sizes)
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

        out, _ = block_sparse_attn_256_bshd(query, key, value, mask, attn_metadata.variable_block_sizes)

        if gate_compress is not None:
            # Wan-style compression branch: dense attention over pooled tiles,
            # broadcast to each tile's rows, scaled by the learned gate
            # (zero-initialized for H3 => branch contributes nothing until
            # finetuned; the model layer skips it entirely for all-zero gates).
            v_pooled = _pool_tiles(value, attn_metadata.variable_block_sizes)
            out_c = torch.matmul(torch.softmax(scores, dim=-1), v_pooled)  # [B, H, n_tiles, D]
            out_c = out_c.permute(0, 2, 1, 3).to(out.dtype)  # [B, n_tiles, H, D]
            batch, seq_len, heads, dim = out.shape
            n_tiles = attn_metadata.variable_block_sizes.numel()
            # Out-of-place: on the CuTe backend ``out`` is the tensor FA4's
            # autograd node saved for its backward, so an in-place add here
            # bumps its version counter and backward dies with "one of the
            # variables needed for gradient computation has been modified".
            out_tiled = out.view(batch, n_tiles, _TILE_ELEMS, heads, dim)
            gate_tiled = gate_compress.view(batch, n_tiles, _TILE_ELEMS, heads, dim)
            out = (out_tiled + out_c.unsqueeze(2) * gate_tiled).view(batch, seq_len, heads, dim)
        return out
