import json
from dataclasses import dataclass
from typing import List, Optional, Type

import torch
from einops import rearrange
from st_attn import sliding_tile_attention

import fastvideo.v1.envs as envs
from fastvideo.v1.attention.backends.abstract import (AttentionBackend,
                                                      AttentionImpl,
                                                      AttentionMetadata,
                                                      AttentionMetadataBuilder)
from fastvideo.v1.distributed import get_sp_group
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch

logger = init_logger(__name__)


# TODO(will-refactor): move this to a utils file
def dict_to_3d_list(mask_strategy) -> List[List[List[Optional[torch.Tensor]]]]:
    indices = [tuple(map(int, key.split('_'))) for key in mask_strategy]

    max_timesteps_idx = max(
        timesteps_idx for timesteps_idx, layer_idx, head_idx in indices) + 1
    max_layer_idx = max(layer_idx
                        for timesteps_idx, layer_idx, head_idx in indices) + 1
    max_head_idx = max(head_idx
                       for timesteps_idx, layer_idx, head_idx in indices) + 1

    result = [[[None for _ in range(max_head_idx)]
               for _ in range(max_layer_idx)] for _ in range(max_timesteps_idx)]

    for key, value in mask_strategy.items():
        timesteps_idx, layer_idx, head_idx = map(int, key.split('_'))
        result[timesteps_idx][layer_idx][head_idx] = value

    return result


class RangeDict(dict):

    def __getitem__(self, item):
        for key in self.keys():
            if isinstance(key, tuple):
                low, high = key
                if low <= item <= high:
                    return super().__getitem__(key)
            elif key == item:
                return super().__getitem__(key)
        raise KeyError(f"seq_len {item} not supported for STA")


class SlidingTileAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        # TODO(will-refactor): check this
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "SLIDING_TILE_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["SlidingTileAttentionImpl"]:
        return SlidingTileAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["SlidingTileAttentionMetadata"]:
        return SlidingTileAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["SlidingTileAttentionMetadataBuilder"]:
        return SlidingTileAttentionMetadataBuilder


@dataclass
class SlidingTileAttentionMetadata(AttentionMetadata):
    current_timestep: int


class SlidingTileAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(
        self,
        current_timestep: int,
        forward_batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> SlidingTileAttentionMetadata:

        return SlidingTileAttentionMetadata(current_timestep=current_timestep, )


class SlidingTileAttentionImpl(AttentionImpl):

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
        config_file = envs.FASTVIDEO_ATTENTION_CONFIG
        if config_file is None:
            raise ValueError("FASTVIDEO_ATTENTION_CONFIG is not set")

        with open(config_file) as f:
            mask_strategy = json.load(f)
        mask_strategy = dict_to_3d_list(mask_strategy)

        self.prefix = prefix
        self.mask_strategy = mask_strategy
        sp_group = get_sp_group()
        self.sp_size = sp_group.world_size
        # STA config
        self.STA_base_tile_size = [6, 8, 8]
        self.img_latent_shape_mapping = RangeDict({
            (115200, 115456): '30x48x80',
            82944: '36x48x48',
            69120: '18x48x80',
        })
        self.full_window_mapping = {
            '30x48x80': [5, 6, 10],
            '36x48x48': [6, 6, 6],
            '18x48x80': [3, 6, 10]
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
        img_sequence_length = qkv.shape[1]
        self.img_latent_shape_str = self.img_latent_shape_mapping[
            img_sequence_length]
        self.full_window_size = self.full_window_mapping[
            self.img_latent_shape_str]
        self.img_latent_shape_int = list(
            map(int, self.img_latent_shape_str.split('x')))
        self.img_seq_length = self.img_latent_shape_int[
            0] * self.img_latent_shape_int[1] * self.img_latent_shape_int[2]
        return self.tile(qkv)

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: SlidingTileAttentionMetadata,
    ) -> torch.Tensor:
        return self.untile(output)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: SlidingTileAttentionMetadata,
    ) -> torch.Tensor:

        assert self.mask_strategy is not None, "mask_strategy cannot be None for SlidingTileAttention"
        assert self.mask_strategy[
            0] is not None, "mask_strategy[0] cannot be None for SlidingTileAttention"

        timestep = attn_metadata.current_timestep
        # pattern:'.double_blocks.0.attn.impl' or '.single_blocks.0.attn.impl'
        layer_idx = int(self.prefix.split('.')[-3])

        # TODO: remove hardcode

        text_length = q.shape[1] - self.img_seq_length
        has_text = text_length > 0

        query = q.transpose(1, 2).contiguous()
        key = k.transpose(1, 2).contiguous()
        value = v.transpose(1, 2).contiguous()

        head_num = query.size(1)
        sp_group = get_sp_group()
        current_rank = sp_group.rank_in_group
        start_head = current_rank * head_num
        windows = [
            self.mask_strategy[timestep][layer_idx][head_idx + start_head]
            for head_idx in range(head_num)
        ]
        # if has_text is False:
        #     from IPython import embed
        #     embed()
        hidden_states = sliding_tile_attention(
            query, key, value, windows, text_length, has_text,
            self.img_latent_shape_str).transpose(1, 2)

        return hidden_states
