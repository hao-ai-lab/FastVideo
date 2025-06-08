import json
from dataclasses import dataclass
from typing import List, Optional, Type

import torch
from einops import rearrange

import fastvideo.v1.envs as envs
from fastvideo.v1.attention.backends.abstract import (AttentionBackend,
                                                      AttentionImpl,
                                                      AttentionMetadata,
                                                      AttentionMetadataBuilder)
from fastvideo.v1.distributed import get_sp_group
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.attention.backends.video_sparse_attn_patterns.sparse_attns import sparse_attn_c_s_p

logger = init_logger(__name__)



class VideoSparseAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        # TODO(will-refactor): check this
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "VIDEO_SPARSE_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["VideoSparseAttentionImpl"]:
        return VideoSparseAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["VideoSparseAttentionMetadata"]:
        return VideoSparseAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["VideoSparseAttentionMetadataBuilder"]:
        return VideoSparseAttentionMetadataBuilder


@dataclass
class VideoSparseAttentionMetadata(AttentionMetadata):
    current_timestep: int


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

        return VideoSparseAttentionMetadata(current_timestep=current_timestep, )


class VideoSparseAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: Optional[int] = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        # TODO(will-refactor): for now this is the mask strategy, but maybe we should
        # have a more general config for STA?
        self.prefix = prefix
        sp_group = get_sp_group()
        self.sp_size = sp_group.world_size
        # STA config
        self.STA_base_tile_size = [4, 4, 4]
        self.img_latent_shape_mapping = {
            76800: '20x48x80',
        }
        self.full_window_mapping = {
            '20x48x80': [5, 12, 20]
        }

    def tile(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x,
                      "b (sp t h w) head d -> b (t sp h w) head d",
                      sp=self.sp_size,
                      t=self.img_latent_shape_int[0] // self.sp_size,
                      h=self.img_latent_shape_int[1],
                      w=self.img_latent_shape_int[2])
        return rearrange(
            x,
            "b (n_t ts_t n_h ts_h n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
            n_t=self.full_window_size[0],
            n_h=self.full_window_size[1],
            n_w=self.full_window_size[2],
            ts_t=self.STA_base_tile_size[0],
            ts_h=self.STA_base_tile_size[1],
            ts_w=self.STA_base_tile_size[2])

    def untile(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(
            x,
            "b (n_t n_h n_w ts_t ts_h ts_w) h d -> b (n_t ts_t n_h ts_h n_w ts_w) h d",
            n_t=self.full_window_size[0],
            n_h=self.full_window_size[1],
            n_w=self.full_window_size[2],
            ts_t=self.STA_base_tile_size[0],
            ts_h=self.STA_base_tile_size[1],
            ts_w=self.STA_base_tile_size[2])
        return rearrange(x,
                         "b (t sp h w) head d -> b (sp t h w) head d",
                         sp=self.sp_size,
                         t=self.img_latent_shape_int[0] // self.sp_size,
                         h=self.img_latent_shape_int[1],
                         w=self.img_latent_shape_int[2])

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        self.img_latent_shape_int = [attn_metadata.img_latent_shape[0],
                                     attn_metadata.img_latent_shape[1] // 2,
                                     attn_metadata.img_latent_shape[2] // 2]
        self.full_window_size = [self.img_latent_shape_int[0] // self.STA_base_tile_size[0],
                                 self.img_latent_shape_int[1] // self.STA_base_tile_size[1],
                                 self.img_latent_shape_int[2] // self.STA_base_tile_size[2]]
        self.img_seq_length = self.img_latent_shape_int[
            0] * self.img_latent_shape_int[1] * self.img_latent_shape_int[2]
        return self.tile(qkv)

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        return self.untile(output)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate_compress: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:


        timestep = attn_metadata.current_timestep
        # pattern:'.double_blocks.0.attn.impl' or '.single_blocks.0.attn.impl'
        layer_idx = int(self.prefix.split('.')[-3])

        # TODO: remove hardcode

        text_length = q.shape[1] - self.img_seq_length
        has_text = text_length > 0

        query = q.transpose(1, 2).contiguous()
        key = k.transpose(1, 2).contiguous()
        value = v.transpose(1, 2).contiguous()
        gate_compress = gate_compress.transpose(1, 2).contiguous()

        cur_topk = 32
        hidden_states = sparse_attn_c_s_p(
            query, key, value,  topk=cur_topk, block_size=(4, 4, 4), compress_attn_weight=gate_compress
        ).transpose(1, 2)

        return hidden_states
