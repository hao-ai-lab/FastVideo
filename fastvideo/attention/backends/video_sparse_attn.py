# SPDX-License-Identifier: Apache-2.0
import functools
import math
from dataclasses import dataclass, field

import torch

try:
    from fastvideo_kernel import video_sparse_attn, video_sparse_attn_varlen
except ImportError:
    video_sparse_attn = None
    video_sparse_attn_varlen = None

from typing import Any

from fastvideo.attention.backends.abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                                                   AttentionMetadataBuilder)
from fastvideo.distributed import get_sp_group
from fastvideo.logger import init_logger

logger = init_logger(__name__)
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
) -> torch.LongTensor:
    """
    Compute the number of valid (non‑padded) tokens inside every
    (ts_t × ts_h × ts_w) tile after padding ‑‑ flattened in the order
    (t‑tile, h‑tile, w‑tile) that `rearrange` uses.

    Returns
    -------
    torch.LongTensor  # shape: [∏ full_window_size]
    """
    # unpack
    t, h, w = dit_seq_shape
    ts_t, ts_h, ts_w = VSA_TILE_SIZE
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
    VSA_sparsity: float
    num_tiles: list[int]
    total_seq_length: int
    tile_partition_indices: torch.LongTensor
    reverse_tile_partition_indices: torch.LongTensor
    variable_block_sizes: torch.LongTensor
    non_pad_index: torch.LongTensor
    # Optional varlen fields (None = standard non-varlen path)
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_kv: torch.Tensor | None = None
    variable_block_sizes_list: list | None = field(default=None, compare=False)
    q_variable_block_sizes_list: list | None = field(default=None, compare=False)


class VideoSparseAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(  # type: ignore
        self,
        current_timestep: int,
        raw_latent_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        VSA_sparsity: float,
        device: torch.device,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_kv: torch.Tensor | None = None,
        variable_block_sizes_list: list | None = None,
        q_variable_block_sizes_list: list | None = None,
        **kwargs: Any,
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
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            variable_block_sizes_list=variable_block_sizes_list,
            q_variable_block_sizes_list=q_variable_block_sizes_list,
        )

    def build_varlen(
        self,
        current_timestep: int,
        batch_latent_shapes: list[tuple[int, int, int]],
        patch_size: tuple[int, int, int],
        VSA_sparsity: float,
        device: torch.device,
        **kwargs: Any,
    ) -> VideoSparseAttentionMetadata:
        """Build metadata for a varlen batch where each sequence has a different latent shape."""
        variable_block_sizes_list = []
        q_variable_block_sizes_list = []
        q_lengths = []
        kv_lengths = []

        for raw_latent_shape in batch_latent_shapes:
            dit_seq_shape = (
                raw_latent_shape[0] // patch_size[0],
                raw_latent_shape[1] // patch_size[1],
                raw_latent_shape[2] // patch_size[2],
            )
            num_tiles = (
                math.ceil(dit_seq_shape[0] / VSA_TILE_SIZE[0]),
                math.ceil(dit_seq_shape[1] / VSA_TILE_SIZE[1]),
                math.ceil(dit_seq_shape[2] / VSA_TILE_SIZE[2]),
            )
            vbs = construct_variable_block_sizes(dit_seq_shape, num_tiles, device)
            variable_block_sizes_list.append(vbs)
            q_variable_block_sizes_list.append(vbs)
            padded_len = math.prod(num_tiles) * math.prod(VSA_TILE_SIZE)
            q_lengths.append(padded_len)
            kv_lengths.append(padded_len)

        cu_seqlens_q = torch.zeros(len(q_lengths) + 1, dtype=torch.int32, device=device)
        cu_seqlens_q[1:] = torch.tensor(q_lengths, dtype=torch.int32, device=device).cumsum(0)
        cu_seqlens_kv = torch.zeros(len(kv_lengths) + 1, dtype=torch.int32, device=device)
        cu_seqlens_kv[1:] = torch.tensor(kv_lengths, dtype=torch.int32, device=device).cumsum(0)

        return self.build(
            current_timestep=current_timestep,
            raw_latent_shape=batch_latent_shapes[0],
            patch_size=patch_size,
            VSA_sparsity=VSA_sparsity,
            device=device,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            variable_block_sizes_list=variable_block_sizes_list,
            q_variable_block_sizes_list=q_variable_block_sizes_list,
        )


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

    def tile(self, x: torch.Tensor, num_tiles: list[int], tile_partition_indices: torch.LongTensor,
             non_pad_index: torch.LongTensor) -> torch.Tensor:
        t_padded_size = num_tiles[0] * VSA_TILE_SIZE[0]
        h_padded_size = num_tiles[1] * VSA_TILE_SIZE[1]
        w_padded_size = num_tiles[2] * VSA_TILE_SIZE[2]

        x_padded = torch.zeros((x.shape[0], t_padded_size * h_padded_size * w_padded_size, x.shape[-2], x.shape[-1]),
                               device=x.device,
                               dtype=x.dtype)
        x_padded[:, non_pad_index] = x[:, tile_partition_indices]
        return x_padded

    def untile(self, x: torch.Tensor, reverse_tile_partition_indices: torch.LongTensor,
               non_pad_index: torch.LongTensor) -> torch.Tensor:
        x = x[:, non_pad_index][:, reverse_tile_partition_indices]
        return x

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        return self.tile(qkv, attn_metadata.num_tiles, attn_metadata.tile_partition_indices,
                         attn_metadata.non_pad_index)

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        return self.untile(output, attn_metadata.reverse_tile_partition_indices, attn_metadata.non_pad_index)

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        gate_compress = gate_compress.transpose(1, 2).contiguous()

        VSA_sparsity = attn_metadata.VSA_sparsity
        cur_topk = math.ceil((1 - VSA_sparsity) * (attn_metadata.total_seq_length / math.prod(VSA_TILE_SIZE)))

        if attn_metadata.cu_seqlens_q is not None:
            # varlen path: flat packed tensors [B*L, H, D]
            if video_sparse_attn_varlen is None:
                raise NotImplementedError("video_sparse_attn_varlen is not installed")
            q_flat = query.reshape(-1, query.shape[-2], query.shape[-1])
            k_flat = key.reshape(-1, key.shape[-2], key.shape[-1])
            v_flat = value.reshape(-1, value.shape[-2], value.shape[-1])
            caw_flat = gate_compress.reshape(-1, gate_compress.shape[-2], gate_compress.shape[-1])
            hidden_states = video_sparse_attn_varlen(
                q_flat, k_flat, v_flat,
                attn_metadata.cu_seqlens_q,
                attn_metadata.cu_seqlens_kv,
                attn_metadata.variable_block_sizes_list,
                attn_metadata.q_variable_block_sizes_list,
                cur_topk,
                block_size=VSA_TILE_SIZE,
                compress_attn_weight=caw_flat,
            )
            # [total_q, H, D] -> [B, L, H, D]
            return hidden_states.reshape(query.shape[0], -1, query.shape[-2], query.shape[-1])

        if video_sparse_attn is None:
            raise NotImplementedError("video_sparse_attn is not installed")
        hidden_states = video_sparse_attn(query,
                                          key,
                                          value,
                                          attn_metadata.variable_block_sizes,
                                          attn_metadata.variable_block_sizes,
                                          cur_topk,
                                          block_size=VSA_TILE_SIZE,
                                          compress_attn_weight=gate_compress).transpose(1, 2)

        return hidden_states
