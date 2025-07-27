# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange

try:
    from vsa import video_sparse_attn
except ImportError:
    video_sparse_attn = None

from fastvideo.attention.backends.abstract import (AttentionBackend,
                                                   AttentionImpl,
                                                   AttentionMetadata,
                                                   AttentionMetadataBuilder)
from fastvideo.distributed import get_sp_group
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

logger = init_logger(__name__)


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


class VideoSparseAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(
        self,
        current_timestep: int,
        forward_batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VideoSparseAttentionMetadata:
        if forward_batch.latents is None:
            raise ValueError("latents cannot be None")

        raw_latent_shape = forward_batch.raw_latent_shape
        if raw_latent_shape is None:
            raise ValueError("raw_latent_shape cannot be None")

        patch_size = fastvideo_args.pipeline_config.dit_config.patch_size
        dit_seq_shape = [
            raw_latent_shape[2] // patch_size[0],
            raw_latent_shape[3] // patch_size[1],
            raw_latent_shape[4] // patch_size[2]
        ]
        VSA_sparsity = forward_batch.VSA_sparsity

        return VideoSparseAttentionMetadata(current_timestep=current_timestep,
                                            dit_seq_shape=dit_seq_shape,
                                            VSA_sparsity=VSA_sparsity)


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
        self.VSA_base_tile_size = [4, 4, 4]
        self.dit_seq_shape: list[int]
        self.full_window_size: list[int]
        self.img_seq_length: int

    def tile(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: this by right should move inside sequence parallel operation
        x = rearrange(x,
                      "b (sp t h w) head d -> b (t sp) h w head d",
                      sp=self.sp_size,
                      t=self.dit_seq_shape[0] // self.sp_size,
                      h=self.dit_seq_shape[1],
                      w=self.dit_seq_shape[2])
        t_padded_size = self.full_window_size[0] * self.VSA_base_tile_size[0] 
        h_padded_size = self.full_window_size[1] * self.VSA_base_tile_size[1] 
        w_padded_size = self.full_window_size[2] * self.VSA_base_tile_size[2]
        
        x_padded = torch.zeros((x.shape[0], t_padded_size, h_padded_size, w_padded_size, x.shape[4], x.shape[5]), device=x.device, dtype=x.dtype)
        x_padded[:, :self.dit_seq_shape[0], :self.dit_seq_shape[1], :self.dit_seq_shape[2], :, :] = x
        
        return rearrange(
            x_padded,
            "b (n_t ts_t) (n_h ts_h) (n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
            n_t=self.full_window_size[0],
            n_h=self.full_window_size[1],
            n_w=self.full_window_size[2],
            ts_t=self.VSA_base_tile_size[0],
            ts_h=self.VSA_base_tile_size[1],
            ts_w=self.VSA_base_tile_size[2])

    def untile(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(
            x,
            "b (n_t n_h n_w ts_t ts_h ts_w) h d -> b (n_t ts_t) (n_h ts_h) (n_w ts_w) h d",
            n_t=self.full_window_size[0],
            n_h=self.full_window_size[1],
            n_w=self.full_window_size[2],
            ts_t=self.VSA_base_tile_size[0],
            ts_h=self.VSA_base_tile_size[1],
            ts_w=self.VSA_base_tile_size[2])
        x = x[:, :self.dit_seq_shape[0], :self.dit_seq_shape[1], :self.dit_seq_shape[2], :, :]
        # TODO: this by right should move inside sequence parallel operation
        return rearrange(x,
                         "b (t sp) h w head d -> b (sp t h w) head d",
                         sp=self.sp_size,
                         t=self.dit_seq_shape[0] // self.sp_size,
                         h=self.dit_seq_shape[1],
                         w=self.dit_seq_shape[2])

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        self.dit_seq_shape = attn_metadata.dit_seq_shape
        self.full_window_size = [
            math.ceil(self.dit_seq_shape[0] / self.VSA_base_tile_size[0]),
            math.ceil(self.dit_seq_shape[1] / self.VSA_base_tile_size[1]),
            math.ceil(self.dit_seq_shape[2] / self.VSA_base_tile_size[2])
        ]
        self.img_seq_length = math.prod(self.dit_seq_shape)
        return self.tile(qkv)

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        return self.untile(output)
    
    def construct_variable_block_sizes(
        self,
        dit_seq_shape: list[int],
        full_window_size: list[int],
        VSA_base_tile_size: list[int],
        device: torch.device,
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
        t,  h,  w  = dit_seq_shape
        ts_t, ts_h, ts_w = VSA_base_tile_size
        n_t, n_h, n_w    = full_window_size

        def _sizes(dim_len: int, tile: int, n_tiles: int) -> torch.LongTensor:
            """Vector with the size of each tile along one dimension."""
            sizes = torch.full((n_tiles,), tile, dtype=torch.int, device=device)
            # size of last (possibly partial) tile
            remainder = dim_len - (n_tiles - 1) * tile
            sizes[-1] = remainder if remainder > 0 else tile
            return sizes

        t_sizes = _sizes(t,  ts_t, n_t)          # [n_t]
        h_sizes = _sizes(h,  ts_h, n_h)          # [n_h]
        w_sizes = _sizes(w,  ts_w, n_w)          # [n_w]

        # broadcast‑multiply to get voxels per tile, then flatten
        block_sizes = (
            t_sizes[:, None, None]            # [n_t, 1,   1]
            * h_sizes[None, :, None]          # [1,   n_h, 1]
            * w_sizes[None, None, :]          # [1,   1,   n_w]
        ).reshape(-1)                         # [n_t * n_h * n_w]

        return block_sizes

        

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

        cur_topk = math.ceil(
            (1 - VSA_sparsity) *
            (self.img_seq_length / math.prod(self.VSA_base_tile_size)))

        if video_sparse_attn is None:
            raise NotImplementedError("video_sparse_attn is not installed")
        variable_block_sizes = self.construct_variable_block_sizes(self.dit_seq_shape, self.full_window_size, self.VSA_base_tile_size, device=query.device)
        hidden_states = video_sparse_attn(
            query,
            key,
            value,
            variable_block_sizes=variable_block_sizes,
            topk=cur_topk,
            block_size=self.VSA_base_tile_size,
            compress_attn_weight=gate_compress).transpose(1, 2)

        return hidden_states
