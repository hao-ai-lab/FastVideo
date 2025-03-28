# SPDX-License-Identifier: Apache-2.0

import json

import torch
from einops import rearrange

import fastvideo.v1.envs as envs
from fastvideo.v1.distributed import get_sp_group

from .abstract import AttentionImpl, AttentionLayer


class SlidingTileAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
    ) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads)
        config_file = envs.FASTVIDEO_ATTENTION_CONFIG
        if config_file is None:
            raise ValueError("FASTVIDEO_ATTENTION_CONFIG is not set")

        try:
            with open(config_file) as f:
                config = json.load(f)
        except FileNotFoundError:
            raise ValueError(
                f"FASTVIDEO_ATTENTION_CONFIG file not found: {config_file}")
        except json.JSONDecodeError:
            raise ValueError(
                f"FASTVIDEO_ATTENTION_CONFIG file is not a valid JSON file: {config_file}"
            )

        self.config = config
        sp_group = get_sp_group()
        self.sp_size = sp_group.sp_size

    def tile(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = rearrange(x,
                      "b (sp t h w) head d -> b (t sp h w) head d",
                      sp=sp_size,
                      t=30 // sp_size,
                      h=48,
                      w=80)
        return rearrange(
            x,
            "b (n_t ts_t n_h ts_h n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
            n_t=5,
            n_h=6,
            n_w=10,
            ts_t=6,
            ts_h=8,
            ts_w=8)

    def untile(
        self,
        tiled_output: torch.Tensor,
    ) -> torch.Tensor:
        x = rearrange(
            x,
            "b (n_t n_h n_w ts_t ts_h ts_w) h d -> b (n_t ts_t n_h ts_h n_w ts_w) h d",
            n_t=5,
            n_h=6,
            n_w=10,
            ts_t=6,
            ts_h=8,
            ts_w=8)
        return rearrange(x,
                         "b (t sp h w) head d -> b (sp t h w) head d",
                         sp=sp_size,
                         t=30 // sp_size,
                         h=48,
                         w=80)

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query, encoder_query = q
        key, encoder_key = k
        value, encoder_value = v
        text_length = text_mask.sum()

        if self.sp_size > 1:
            # batch_size, seq_len, attn_heads, head_dim
            query = all_to_all_4D(query, scatter_dim=2, gather_dim=1)
            key = all_to_all_4D(key, scatter_dim=2, gather_dim=1)
            value = all_to_all_4D(value, scatter_dim=2, gather_dim=1)

            def shrink_head(encoder_state, dim):
                local_heads = encoder_state.shape[dim] // nccl_info.sp_size
                return encoder_state.narrow(
                    dim, nccl_info.rank_within_group * local_heads, local_heads)

            encoder_query = shrink_head(encoder_query, dim=2)
            encoder_key = shrink_head(encoder_key, dim=2)
            encoder_value = shrink_head(encoder_value, dim=2)
            # [b, s, h, d]

        sequence_length = query.size(1)
        encoder_sequence_length = encoder_query.size(1)

        if mask_strategy[0] is not None:
            query = torch.cat(
                [self.tile(query, nccl_info.sp_size), encoder_query],
                dim=1).transpose(1, 2)
            key = torch.cat([self.tile(key, nccl_info.sp_size), encoder_key],
                            dim=1).transpose(1, 2)
            value = torch.cat(
                [self.tile(value, nccl_info.sp_size), encoder_value],
                dim=1).transpose(1, 2)

            head_num = query.size(1)
            current_rank = nccl_info.rank_within_group
            start_head = current_rank * head_num
            windows = [
                mask_strategy[head_idx + start_head]
                for head_idx in range(head_num)
            ]

            hidden_states = sliding_tile_attention(query, key, value, windows,
                                                   text_length).transpose(1, 2)
        else:
            raise NotImplementedError(
                "mask_strategy cannot be None for STAttention")
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            (sequence_length, encoder_sequence_length), dim=1)
        if self.sp_size > 1:
            hidden_states = all_to_all_4D(hidden_states,
                                          scatter_dim=1,
                                          gather_dim=2)
            encoder_hidden_states = all_gather(encoder_hidden_states,
                                               dim=2).contiguous()

        hidden_states = hidden_states.to(query.dtype)
        encoder_hidden_states = encoder_hidden_states.to(query.dtype)
