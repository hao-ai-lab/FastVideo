# SPDX-License-Identifier: Apache-2.0
import functools
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

try:
    from fastvideo_kernel import video_sparse_attn
except ImportError:
    video_sparse_attn = None
try:
    from fastvideo_kernel import video_sparse_attn_bshd
except ImportError:
    video_sparse_attn_bshd = None

from typing import Any

import fastvideo.envs as envs
from fastvideo.attention.backends.abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                                                   AttentionMetadataBuilder)
from fastvideo.distributed import get_sp_group
from fastvideo.logger import init_logger

logger = init_logger(__name__)
# VSA tile shape. The tile volume picks the kernel path automatically in
# forward(): (4,4,4)=64 -> existing TK/Triton path (default, unchanged);
# (4,8,8)=256 -> FA4 CuTe block-sparse attention fastpath (Blackwell).
VSA_TILE_SIZE = (4, 4, 4)


@functools.lru_cache(maxsize=10)
def get_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    T, H, W = dit_seq_shape
    ts, hs, ws = tile_size
    indices = torch.arange(T * H * W, device=device, dtype=torch.long).reshape(T, H, W)
    ls = []
    for t in range(math.ceil(T / ts)):
        for h in range(math.ceil(H / hs)):
            for w in range(math.ceil(W / ws)):
                ls.append(indices[t * ts:min(t * ts + ts, T), h * hs:min(h * hs + hs, H),
                                  w * ws:min(w * ws + ws, W)].flatten())
    index = torch.cat(ls, dim=0)
    return index


@functools.lru_cache(maxsize=10)
def get_reverse_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    return torch.argsort(get_tile_partition_indices(dit_seq_shape, tile_size, device))


@functools.lru_cache(maxsize=10)
def construct_variable_block_sizes(
    dit_seq_shape: tuple[int, int, int],
    num_tiles: tuple[int, int, int],
    device: torch.device,
    tile_size: tuple[int, int, int] = VSA_TILE_SIZE,
) -> torch.LongTensor:
    """
    Compute the number of valid (non‑padded) tokens inside every
    (ts_t × ts_h × ts_w) tile after padding ‑‑ flattened in the order
    (t‑tile, h‑tile, w‑tile) that `rearrange` uses.

    Returns
    -------
    torch.LongTensor  # shape: [∏ full_window_size]
    """
    # unpack
    t, h, w = dit_seq_shape
    ts_t, ts_h, ts_w = tile_size
    n_t, n_h, n_w = num_tiles

    def _sizes(dim_len: int, tile: int, n_tiles: int) -> torch.LongTensor:
        """Vector with the size of each tile along one dimension."""
        sizes = torch.full((n_tiles, ), tile, dtype=torch.int, device=device)
        # size of last (possibly partial) tile
        remainder = dim_len - (n_tiles - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(t, ts_t, n_t)  # [n_t]
    h_sizes = _sizes(h, ts_h, n_h)  # [n_h]
    w_sizes = _sizes(w, ts_w, n_w)  # [n_w]

    # broadcast‑multiply to get voxels per tile, then flatten
    block_sizes = (
        t_sizes[:, None, None]  # [n_t, 1,   1]
        * h_sizes[None, :, None]  # [1,   n_h, 1]
        * w_sizes[None, None, :]  # [1,   1,   n_w]
    ).reshape(-1)  # [n_t * n_h * n_w]

    return block_sizes


@functools.lru_cache(maxsize=10)
def get_non_pad_index(
    variable_block_sizes: torch.LongTensor,
    max_block_size: int,
):
    n_win = variable_block_sizes.shape[0]
    device = variable_block_sizes.device
    starts_pad = torch.arange(n_win, device=device) * max_block_size
    index_pad = starts_pad[:, None] + torch.arange(max_block_size, device=device)[None, :]
    index_mask = torch.arange(max_block_size, device=device)[None, :] < variable_block_sizes[:, None]
    return index_pad[index_mask]


class VideoSparseAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "VIDEO_SPARSE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["VideoSparseAttentionImpl"]:
        return VideoSparseAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["VideoSparseAttentionMetadata"]:
        return VideoSparseAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["VideoSparseAttentionMetadataBuilder"]:
        return VideoSparseAttentionMetadataBuilder


@dataclass
class VideoSparseAttentionMetadata(AttentionMetadata):
    current_timestep: int
    dit_seq_shape: list[int]
    num_tiles: list[int]
    total_seq_length: int
    tile_partition_indices: torch.LongTensor
    reverse_tile_partition_indices: torch.LongTensor
    variable_block_sizes: torch.LongTensor
    non_pad_index: torch.LongTensor
    # Precomputed fancy index that fuses ``x[:, non_pad_index][:, reverse_tile_partition_indices]``
    # in postprocess_output().  Avoids materializing the intermediate
    # ``[B, len(non_pad_index), H, D]`` tensor on every layer.
    untile_combined_index: torch.LongTensor
    # Per-step shared padded buffer used by tile().  Inference can reuse this
    # across VSA layers, but training disables it so activation checkpointing
    # can release the large tiled QKVG scratch tensor after each attention call.
    tile_buf: torch.Tensor | None = None
    cache_tile_buf: bool = True
    fused_tile_source_index: torch.Tensor | None = None
    fused_layout_active: bool = False


def compute_topk(sparsity: float, num_blocks: int) -> int:
    """Blocks to keep for a sparsity level, clamped to [1, num_blocks]."""
    return max(1, min(math.ceil((1 - sparsity) * num_blocks), num_blocks))


def _compute_cur_topk(attn_metadata: VideoSparseAttentionMetadata) -> int:
    return compute_topk(attn_metadata.VSA_sparsity, attn_metadata.variable_block_sizes.numel())


def scatter_into_tile_buf(
    x: torch.Tensor,
    target_shape: tuple[int, ...],
    dst_index: torch.Tensor,
    buf: torch.Tensor | None,
    src_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Zero-padded tile scatter shared by the VSA backends.

    Allocates (zeros) when ``buf`` is missing or mismatched; otherwise reuses
    it — pad slots are never written and every non-pad slot is fully
    overwritten per call, so a reused buffer stays valid. Callers own the
    buffer's lifetime (per-metadata for Wan VSA, per-builder for VSA-H3) and
    its aliasing contract: the result is only valid until the next call with
    the same buffer.

    Callers that pass ``buf`` also own its *graph* contract: ``buf[:, dst] = x``
    is an autograd-tracked in-place write, so a non-grad destination becomes a
    graph node as soon as ``x`` tracks gradients. A buffer that outlives the
    fragment it was written from — VSA-H3's is owned by the metadata builder,
    which serves the whole training run — then anchors that fragment, and the
    activations its backward saved, for the rest of training. A caller inside a
    grad-tracking forward should therefore pass ``buf=None`` unless the buffer
    is scoped to that forward: VSA-H3 does so unconditionally, and Wan VSA
    exposes the choice through ``cache_tile_buf`` (training defaults it off).
    """
    if (buf is None or buf.shape != target_shape or buf.dtype != x.dtype or buf.device != x.device):
        buf = torch.zeros(target_shape, device=x.device, dtype=x.dtype)
    buf[:, dst_index] = x if src_index is None else x[:, src_index]
    return buf


@functools.cache
def _get_tile_to_bhsd() -> Callable[..., Any] | None:
    try:
        from fastvideo_kernel.triton_kernels.vsa_tile_layout import tile_to_bhsd
    except ImportError:
        return None
    return tile_to_bhsd


def _can_fuse_vsa64_layout(x: torch.Tensor, metadata: VideoSparseAttentionMetadata) -> bool:
    """Restrict the fused layout to the validated SM100 BF16 inference route."""
    if envs.FASTVIDEO_DISABLE_VSA64_FUSED_LAYOUT.get():
        return False
    if _get_tile_to_bhsd() is None:
        return False
    return (math.prod(VSA_TILE_SIZE) == 64 and metadata.cache_tile_buf and not torch.is_grad_enabled()
            and not x.requires_grad and x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 4
            and x.shape[1] == metadata.total_seq_length and x.shape[-1] == 128
            and torch.cuda.get_device_capability(x.device) == (10, 0))


class VideoSparseAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(  # type: ignore
        self,
        current_timestep: int,
        raw_latent_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        VSA_sparsity: float,
        device: torch.device,
        cache_tile_buf: bool = True,
        **kwargs: dict[str, Any],
    ) -> VideoSparseAttentionMetadata:
        patch_size = patch_size
        dit_seq_shape = (raw_latent_shape[0] // patch_size[0], raw_latent_shape[1] // patch_size[1],
                         raw_latent_shape[2] // patch_size[2])

        num_tiles = (math.ceil(dit_seq_shape[0] / VSA_TILE_SIZE[0]), math.ceil(dit_seq_shape[1] / VSA_TILE_SIZE[1]),
                     math.ceil(dit_seq_shape[2] / VSA_TILE_SIZE[2]))
        total_seq_length = math.prod(dit_seq_shape)

        tile_partition_indices = get_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
        reverse_tile_partition_indices = get_reverse_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
        variable_block_sizes = construct_variable_block_sizes(dit_seq_shape, num_tiles, device)
        non_pad_index = get_non_pad_index(variable_block_sizes, math.prod(VSA_TILE_SIZE))
        untile_combined_index = non_pad_index[reverse_tile_partition_indices]

        return VideoSparseAttentionMetadata(
            current_timestep=current_timestep,
            dit_seq_shape=dit_seq_shape,  # type: ignore
            VSA_sparsity=VSA_sparsity,  # type: ignore
            num_tiles=num_tiles,  # type: ignore
            total_seq_length=total_seq_length,  # type: ignore
            tile_partition_indices=tile_partition_indices,  # type: ignore
            reverse_tile_partition_indices=reverse_tile_partition_indices,
            variable_block_sizes=variable_block_sizes,
            non_pad_index=non_pad_index,
            untile_combined_index=untile_combined_index,
            cache_tile_buf=cache_tile_buf)


class VideoSparseAttentionImpl(AttentionImpl):

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
        sp_group = get_sp_group()
        self.sp_size = sp_group.world_size

    def tile(self, x: torch.Tensor, attn_metadata: VideoSparseAttentionMetadata) -> torch.Tensor:
        """Tile ``x`` into ``attn_metadata.tile_buf`` and return it.

        The returned tensor aliases the per-metadata buffer and is only
        valid until the next ``tile()`` / ``preprocess_qkv`` call on the
        same ``attn_metadata``.  Callers must consume (or copy) the
        result before invoking another VSA layer with the same metadata.
        The default 64-token path materializes q/k/v copies in ``forward()``
        and consumes its transposed gate view synchronously in the coarse/sparse
        combine before returning. The SM100 inference path instead writes BHSD
        directly to this buffer. Both paths consume the result before the next
        layer reuses it; future callers must preserve this contract.
        """
        num_tiles = attn_metadata.num_tiles
        t_padded_size = num_tiles[0] * VSA_TILE_SIZE[0]
        h_padded_size = num_tiles[1] * VSA_TILE_SIZE[1]
        w_padded_size = num_tiles[2] * VSA_TILE_SIZE[2]
        target_shape = (x.shape[0], t_padded_size * h_padded_size * w_padded_size, x.shape[-2], x.shape[-1])

        if not attn_metadata.cache_tile_buf:
            return scatter_into_tile_buf(x, target_shape, attn_metadata.non_pad_index, None,
                                         attn_metadata.tile_partition_indices)

        # Buffer scoped to the per-step metadata (lazily allocated on the
        # first VSA layer's call within a denoising step), which keeps reuse
        # safe across concurrent requests.
        buf = scatter_into_tile_buf(x, target_shape, attn_metadata.non_pad_index, attn_metadata.tile_buf,
                                    attn_metadata.tile_partition_indices)
        attn_metadata.tile_buf = buf
        return buf

    def untile(self, x: torch.Tensor, untile_combined_index: torch.LongTensor) -> torch.Tensor:
        # Single fancy index using precomputed combined indices; avoids
        # the intermediate ``[B, len(non_pad_index), H, D]`` tensor that
        # the two-step ``x[:, non_pad_index][:, reverse_tile_partition_indices]``
        # would allocate on every layer.
        return x[:, untile_combined_index]

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        """Tile QKV; aliasing contract: see ``tile()``."""
        attn_metadata.fused_layout_active = _can_fuse_vsa64_layout(qkv, attn_metadata)
        if not attn_metadata.fused_layout_active:
            return self.tile(qkv, attn_metadata)

        tile_to_bhsd = _get_tile_to_bhsd()
        assert tile_to_bhsd is not None

        padded_sequence = attn_metadata.variable_block_sizes.numel() * math.prod(VSA_TILE_SIZE)
        if attn_metadata.fused_tile_source_index is None:
            source_index = torch.full((padded_sequence, ), -1, device=qkv.device, dtype=torch.int32)
            source_index[attn_metadata.non_pad_index] = attn_metadata.tile_partition_indices.to(torch.int32)
            attn_metadata.fused_tile_source_index = source_index
        target_shape = (qkv.shape[0], qkv.shape[2], padded_sequence, qkv.shape[3])
        buffer = attn_metadata.tile_buf
        if buffer is None or buffer.shape != target_shape or buffer.dtype != qkv.dtype or buffer.device != qkv.device:
            buffer = torch.empty(target_shape, device=qkv.device, dtype=qkv.dtype)
            attn_metadata.tile_buf = buffer
        tile_to_bhsd(
            qkv,
            attn_metadata.fused_tile_source_index,
            buffer,
            qkv.shape[1],
            padded_sequence,
            qkv.shape[2],
            qkv.shape[3],
        )
        return buffer

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        return self.untile(output, attn_metadata.untile_combined_index)

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        block_elements = math.prod(VSA_TILE_SIZE)
        cur_topk = _compute_cur_topk(attn_metadata)

        # 256-element tiles auto-route to the FA4 CuTe BSHD fastpath, which
        # consumes [B, S, H, D] directly -- skip the transpose round-trip.
        if block_elements == 256 and video_sparse_attn_bshd is not None:
            return video_sparse_attn_bshd(query,
                                          key,
                                          value,
                                          attn_metadata.variable_block_sizes,
                                          attn_metadata.variable_block_sizes,
                                          cur_topk,
                                          block_size=VSA_TILE_SIZE,
                                          compress_attn_weight=gate_compress)

        if video_sparse_attn is None:
            raise NotImplementedError("video_sparse_attn is not installed")
        # The SM100 inference preprocessing path already emits BHSD. Other
        # shapes, architectures, training, and the benchmark disable switch
        # retain the original BSHD scatter/transpose path.
        padded_sequence = attn_metadata.variable_block_sizes.numel() * block_elements
        if not (attn_metadata.fused_layout_active and query.shape[2] == padded_sequence):
            query = query.transpose(1, 2).contiguous()
            key = key.transpose(1, 2).contiguous()
            value = value.transpose(1, 2).contiguous()
            # The coarse/sparse combine only reads the gate elementwise at
            # block resolution, so its transposed view needs no contiguous copy.
            gate_compress = gate_compress.transpose(1, 2)
        return video_sparse_attn(query,
                                 key,
                                 value,
                                 attn_metadata.variable_block_sizes,
                                 attn_metadata.variable_block_sizes,
                                 cur_topk,
                                 block_size=VSA_TILE_SIZE,
                                 compress_attn_weight=gate_compress).transpose(1, 2)
