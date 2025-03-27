# SPDX-License-Identifier: Apache-2.0
# Adapted from transformers: https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/models/t5/modeling_t5.py

# Derived from T5 implementation posted on HuggingFace; license below:
#
# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch T5 model."""

import math
import re
from typing import Iterable, List, Optional, Set, Tuple
import copy
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5Config
# TODO best way to handle xformers imports?

# TODO func should be in backend interface
# from vllm.attention.backends.xformers import (XFormersMetadata, _get_attn_bias,
#                                               _set_attn_bias)
# from vllm.attention.layer import Attention, AttentionMetadata
# from vllm.config import CacheConfig, VllmConfig
from fastvideo.v1.distributed import get_tensor_model_parallel_world_size
from fastvideo.v1.layers.activation import get_act_fn
from fastvideo.v1.layers.layernorm import RMSNorm
from fastvideo.v1.layers.linear import (ColumnParallelLinear,
                                        QKVParallelLinear,
                                        RowParallelLinear)
# from vllm.model_executor.layers.logits_processor import LogitsProcessor
# from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from fastvideo.v1.layers.vocab_parallel_embedding import VocabParallelEmbedding
from fastvideo.v1.models.loader.weight_utils import (
    default_weight_loader)

class QuantizationConfig:
    pass

class AttentionType:
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """
    # Decoder attention between previous layer Q/K/V
    DECODER = "decoder"
    # Encoder attention between previous layer Q/K/V for encoder-decoder
    ENCODER = "encoder"
    # Encoder attention between previous layer Q/K/V
    ENCODER_ONLY = "encoder_only"
    # Attention between dec. Q and enc. K/V for encoder-decoder
    ENCODER_DECODER = "encoder_decoder"

@dataclass
class AttentionMetadata:
    attn_bias: torch.Tensor

class T5DenseActDense(nn.Module):

    def __init__(self,
                 config: T5Config,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.wi = ColumnParallelLinear(config.d_model, config.d_ff, bias=False)
        self.wo = RowParallelLinear(config.d_ff,
                                    config.d_model,
                                    bias=False,
                                    quant_config=quant_config)
        self.act = get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states, _ = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        # if (
        #     isinstance(self.wo.weight, torch.Tensor)
        #     and hidden_states.dtype != self.wo.weight.dtype
        #     and self.wo.weight.dtype != torch.int8
        # ):
        #     hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states, _ = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Module):

    def __init__(self,
                 config: T5Config,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.wi_0 = ColumnParallelLinear(config.d_model,
                                         config.d_ff,
                                         bias=False,
                                         quant_config=quant_config)
        self.wi_1 = ColumnParallelLinear(config.d_model,
                                         config.d_ff,
                                         bias=False,
                                         quant_config=quant_config)
        # Should not run in fp16 unless mixed-precision is used,
        # see https://github.com/huggingface/transformers/issues/20287.
        self.wo = RowParallelLinear(config.d_ff,
                                    config.d_model,
                                    bias=False,
                                    quant_config=quant_config)
        self.act = get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_gelu = self.act(self.wi_0(hidden_states)[0])
        hidden_linear, _ = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states, _ = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):

    def __init__(self,
                 config: T5Config,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(
                config, quant_config=quant_config)
        else:
            self.DenseReluDense = T5DenseActDense(config,
                                                  quant_config=quant_config)

        self.layer_norm = RMSNorm(config.d_model,
                                      eps=config.layer_norm_epsilon)

    def forward(self, hidden_states) -> torch.Tensor:
        forwarded_states = self.layer_norm.forward_native(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + forwarded_states
        return hidden_states

# T5 has attn_bias and does not use softmax scaling
class T5MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, attn_bias=None):
        b, _, n, c = q.shape
        attn = torch.einsum('binc,bjnc->bnij', q, k)
        if attn_bias is not None:
            attn += attn_bias
            
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum('bnij,bjnc->binc', attn, v)
        x = x.reshape(b, -1, n * c)
        return x

class T5Attention(nn.Module):

    def __init__(self,
                 config: T5Config,
                 attn_type: AttentionType,
                 has_relative_attention_bias=False,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.attn_type = attn_type
        # Cross-attention has no relative pos encoding anyway
        self.is_decoder = attn_type == AttentionType.DECODER
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = \
            config.relative_attention_num_buckets
        self.relative_attention_max_distance = \
            config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv

        # Partition heads across multiple tensor parallel GPUs.
        tp_world_size = get_tensor_model_parallel_world_size()
        assert config.num_heads % tp_world_size == 0
        self.n_heads = config.num_heads // tp_world_size

        self.inner_dim = self.n_heads * self.key_value_proj_dim
        # No GQA in t5.
        self.n_kv_heads = self.n_heads

        self.qkv_proj = QKVParallelLinear(self.d_model,
                                          self.d_model // self.n_heads,
                                          self.n_heads,
                                          self.n_kv_heads,
                                          bias=False,
                                          quant_config=quant_config)

        # NOTE (NickLucche) T5 employs a scaled weight initialization scheme
        # instead of scaling attention scores directly.
        # self.attn = Attention(self.n_heads,
        #                       config.d_kv,
        #                       1.0,
        #                       quant_config=quant_config,
        #                       prefix=f"{prefix}.attn",
        #                       attn_type=self.attn_type)
        self.attn = T5MultiHeadAttention()

        # Only the first SelfAttention block in encoder decoder has this
        # embedding layer, the others reuse its output.
        if self.has_relative_attention_bias:
            self.relative_attention_bias = \
                VocabParallelEmbedding(self.relative_attention_num_buckets,
                                       self.n_heads,
                                       org_num_embeddings=self.relative_attention_num_buckets,
                                       padding_size=self.relative_attention_num_buckets,
                                       quant_config=quant_config)
        self.o = RowParallelLinear(
            self.inner_dim,
            self.d_model,
            bias=False,
            quant_config=quant_config,
        )

    @staticmethod
    def _relative_position_bucket(relative_position,
                                  bidirectional=True,
                                  num_buckets=32,
                                  max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. 
        The relative position is defined as memory_position - query_position, 
        i.e. the distance in tokens from the attending position to the 
        attended-to position. If bidirectional=False, then positive relative 
        positions are invalid. We use smaller buckets for small absolute 
        relative_position and larger buckets for larger absolute 
        relative_positions. All relative positions >=max_distance map to the
        same bucket. All relative positions <=-max_distance map to the same 
        bucket. This should allow for more graceful generalization to longer
        sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """# noqa: E501
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(
                torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position,
                                           torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins
        # in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) /
            math.log(max_distance / max_exact) *
            (num_buckets - max_exact)).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1))

        relative_buckets += torch.where(is_small, relative_position,
                                        relative_position_if_large)
        return relative_buckets

    def compute_bias(self,
                     query_length,
                     key_length,
                     device=None) -> torch.Tensor:
        """Compute binned relative position bias"""
        # TODO possible tp issue?
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length,
                                        dtype=torch.long,
                                        device=device)[:, None]
        memory_position = torch.arange(key_length,
                                       dtype=torch.long,
                                       device=device)[None, :]
        # max_seq_len, nh
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        x = values.permute([2, 0, 1]).unsqueeze(
            0)  # shape (1, num_heads, query_length, key_length)
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,  # (num_tokens, d_model)
        attention_mask: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        bs, seq_len, _ = hidden_states.shape        
        num_seqs = bs
        n, c = self.n_heads, self.d_model // self.n_heads
        qkv, _ = self.qkv_proj(hidden_states)
        # Projection of 'own' hidden state (self-attention). No GQA here.
        q, k, v = qkv.split(self.inner_dim, dim=-1)
        q = q.view(bs, -1, n, c)
        k = k.view(bs, -1, n, c)
        v = v.view(bs, -1, n, c)

        assert attn_metadata is not None
        attn_bias = attn_metadata.attn_bias
        # Not compatible with CP here (as all encoder-decoder models),
        # as it assumes homogeneous batch (prefills or decodes).
        if self.has_relative_attention_bias:
            # Self-attention. Compute T5 relative positional encoding.
            # The bias term is computed on longest sequence in batch. Biases
            # for shorter sequences are slices of the longest.

            # bias expected shape: (num_seqs, NH, L, L_pad) for prefill,
            #                      (num_seqs, NH, 1, L_pad) for decodes.
            assert self.attn_type == AttentionType.ENCODER
            # TODO (NickLucche) avoid extra copy on repeat,
            # provide multiple slices of same memory
            attn_bias = self.compute_bias(seq_len, seq_len).repeat(num_seqs, 1, 1, 1)
            attn_metadata.attn_bias = attn_bias
        else:
            # Encoder/Decoder Self-Attention Layer, attn bias already cached.
            assert attn_bias is not None

        if attention_mask is not None:
            attention_mask = attention_mask.view(bs, 1, 1,
                             -1) if attention_mask.ndim == 2 else attention_mask.unsqueeze(1)
            attn_bias.masked_fill_(attention_mask == 0, torch.finfo(q.dtype).min)
        attn_output = self.attn(q, k, v, attn_bias)
        output, _ = self.o(attn_output)
        return output


class T5LayerSelfAttention(nn.Module):

    def __init__(
        self,
        config,
        has_relative_attention_bias=False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.SelfAttention = T5Attention(
            config,
            AttentionType.DECODER
            if "decoder" in prefix else AttentionType.ENCODER,
            has_relative_attention_bias=has_relative_attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.SelfAttention")
        self.layer_norm = RMSNorm(config.d_model,
                                      eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        normed_hidden_states = self.layer_norm.forward_native(hidden_states)
        attention_output = self.SelfAttention(
            hidden_states=normed_hidden_states,
            attention_mask=attention_mask,
            attn_metadata=attn_metadata,
        )
        hidden_states = hidden_states + attention_output
        return hidden_states


class T5LayerCrossAttention(nn.Module):

    def __init__(self,
                 config,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.EncDecAttention = T5Attention(config,
                                           AttentionType.ENCODER_DECODER,
                                           has_relative_attention_bias=False,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.EncDecAttention")
        self.layer_norm = RMSNorm(config.d_model,
                                      eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        normed_hidden_states = self.layer_norm.forward_native(hidden_states)
        attention_output = self.EncDecAttention(
            hidden_states=normed_hidden_states,
            attn_metadata=attn_metadata,
        )
        hidden_states = hidden_states + attention_output
        return hidden_states


class T5Block(nn.Module):

    def __init__(self,
                 config: T5Config,
                 is_decoder: bool,
                 has_relative_attention_bias=False,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.is_decoder = is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn"))

        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.cross_attn"))

        self.layer.append(T5LayerFF(config, quant_config=quant_config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:

        hidden_states = self.layer[0](
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attn_metadata=attn_metadata
        )
        if self.is_decoder:
            hidden_states = self.layer[1](
                hidden_states=hidden_states,
                attn_metadata=attn_metadata
            )

            # Apply Feed Forward layer
            hidden_states = self.layer[2](hidden_states)
        else:
            hidden_states = self.layer[1](hidden_states)
        return hidden_states


class T5Stack(nn.Module):

    def __init__(self,
                 config: T5Config,
                 is_decoder: bool,
                 n_layers: int,
                 embed_tokens=None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 is_umt5: bool = False):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.is_umt5 = is_umt5
        if is_umt5:
            self.block = nn.ModuleList([
                T5Block(config,
                        is_decoder=is_decoder,
                        has_relative_attention_bias=True,
                        quant_config=quant_config,
                        prefix=f"{prefix}.blocks.{i}") for i in range(n_layers)
            ])
        else:
            # Only the first block has relative positional encoding.
            self.block = nn.ModuleList([
                T5Block(config,
                        is_decoder=is_decoder,
                        has_relative_attention_bias=i == 0,
                        quant_config=quant_config,
                        prefix=f"{prefix}.blocks.{i}") for i in range(n_layers)
            ])
        self.final_layer_norm = RMSNorm(config.d_model,
                                            eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for idx, block in enumerate(self.block):
            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                attn_metadata=attn_metadata,
            )
        hidden_states = self.final_layer_norm.forward_native(hidden_states)
        return hidden_states


# class T5Model(nn.Module):
#     _tied_weights_keys = [
#         "encoder.embed_tokens.weight", "decoder.embed_tokens.weight"
#     ]

#     def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
#         super().__init__()
#         config: T5Config = vllm_config.model_config.hf_config
#         cache_config = vllm_config.cache_config
#         quant_config = vllm_config.quant_config
#         lora_config = vllm_config.lora_config

#         lora_vocab = (lora_config.lora_extra_vocab_size *
#                       (lora_config.max_loras or 1)) if lora_config else 0
#         self.vocab_size = config.vocab_size + lora_vocab
#         self.padding_idx = config.pad_token_id
#         self.shared = VocabParallelEmbedding(
#             config.vocab_size,
#             config.d_model,
#             org_num_embeddings=config.vocab_size)

#         self.encoder = T5Stack(config,
#                                False,
#                                config.num_layers,
#                                self.shared,
#                                cache_config=cache_config,
#                                quant_config=quant_config,
#                                prefix=f"{prefix}.encoder")
#         self.decoder = T5Stack(config,
#                                True,
#                                config.num_decoder_layers,
#                                self.shared,
#                                cache_config=cache_config,
#                                quant_config=quant_config,
#                                prefix=f"{prefix}.decoder")

#     def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
#         return self.shared(input_ids)

#     def forward(self, input_ids: torch.Tensor, encoder_input_ids: torch.Tensor,
#                 kv_caches: List[torch.Tensor],
#                 attn_metadata: AttentionMetadata) -> torch.Tensor:
#         encoder_hidden_states = None

#         if encoder_input_ids.numel() > 0:
#             # Run encoder attention if a non-zero number of encoder tokens
#             # are provided as input: on a regular generate call, the encoder
#             # runs once, on the prompt. Subsequent decoder calls reuse output
#             # `encoder_hidden_states`.
#             encoder_hidden_states = self.encoder(input_ids=encoder_input_ids,
#                                                  kv_caches=kv_caches,
#                                                  attn_metadata=attn_metadata)
#             # Clear attention bias state.
#             attn_metadata.attn_bias = None
#             attn_metadata.encoder_attn_bias = None
#             attn_metadata.cross_attn_bias = None

#         decoder_outputs = self.decoder(
#             input_ids=input_ids,
#             encoder_hidden_states=encoder_hidden_states,
#             kv_caches=kv_caches,
#             attn_metadata=attn_metadata)

#         # When capturing CUDA Graph
#         attn_metadata.attn_bias = None
#         attn_metadata.encoder_attn_bias = None
#         attn_metadata.cross_attn_bias = None
#         return decoder_outputs

class T5EncoderModel(nn.Module):
    _tied_weights_keys = ["encoder.embed_tokens.weight"]
    _keys_to_ignore_on_load_unexpected = [r"decoder"]

    def __init__(self, config: T5Config, prefix: str = ""):
        super().__init__()

        quant_config = None

        self.shared = VocabParallelEmbedding(
            config.vocab_size,
            config.d_model,
            org_num_embeddings=config.vocab_size)

        self.encoder = T5Stack(config,
            False,
            config.num_layers,
            self.shared,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder")

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs

class UMT5EncoderModel(nn.Module):
    def __init__(self, config: T5Config, prefix: str = ""):
        super().__init__()

        quant_config = None

        self.shared = VocabParallelEmbedding(
            config.vocab_size,
            config.d_model,
            org_num_embeddings=config.vocab_size)

        self.encoder = T5Stack(config,
            False,
            config.num_layers,
            self.shared,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder",
            is_umt5=True)

    def get_input_embeddings(self):
        return self.shared

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        attn_metadata = AttentionMetadata(None)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attn_metadata=attn_metadata,
        )

        return encoder_outputs

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q", "q"),
            (".qkv_proj", ".k", "k"),
            (".qkv_proj", ".v", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            loaded = False
            if "decoder" in name or "lm_head" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue
                
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded = True
                break
            if not loaded:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

# class T5ForConditionalGeneration(nn.Module):
#     _keys_to_ignore_on_load_unexpected = [
#         "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
#     ]
#     _tied_weights_keys = [
#         "encoder.embed_tokens.weight", "decoder.embed_tokens.weight",
#         "lm_head.weight"
#     ]

#     def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
#         super().__init__()
#         config: T5Config = vllm_config.model_config.hf_config
#         self.model_dim = config.d_model
#         self.config = config
#         self.unpadded_vocab_size = config.vocab_size
#         if lora_config := vllm_config.lora_config:
#             self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

#         self.model = T5Model(vllm_config=vllm_config,
#                              prefix=maybe_prefix(prefix, "model"))
#         # Although not in config, this is the default for hf models.
#         if self.config.tie_word_embeddings:
#             self.lm_head = self.model.shared
#             # in transformers this is smt more explicit, as in (after load)
#             # self.lm_head.weight = self.model.shared.weight
#         else:
#             self.lm_head = ParallelLMHead(self.unpadded_vocab_size,
#                                           config.d_model,
#                                           org_num_embeddings=config.vocab_size)

#         self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
#                                                 config.vocab_size)
#         self.sampler = get_sampler()

#     def compute_logits(
#         self,
#         hidden_states: torch.Tensor,
#         sampling_metadata,
#     ) -> Optional[torch.Tensor]:
#         if self.config.tie_word_embeddings:
#             # Rescale output before projecting on vocab
#             # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586 # noqa: E501
#             hidden_states = hidden_states * (self.model_dim**-0.5)
#         logits = self.logits_processor(self.lm_head, hidden_states,
#                                        sampling_metadata)
#         return logits

#     def sample(
#         self,
#         logits: Optional[torch.Tensor],
#         sampling_metadata,
#     ) -> Optional[SamplerOutput]:
#         next_tokens = self.sampler(logits, sampling_metadata)
#         return next_tokens

#     def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
#         return self.model.shared(input_ids)

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#         kv_caches: List[torch.Tensor],
#         attn_metadata: AttentionMetadata,
#         intermediate_tensors: Optional[torch.Tensor] = None,
#         *,
#         encoder_input_ids: torch.Tensor,
#         encoder_positions: torch.Tensor,
#         **kwargs,
#     ) -> torch.Tensor:
#         return self.model(input_ids, encoder_input_ids, kv_caches,
#                           attn_metadata)

#     def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
#         model_params_dict = dict(self.named_parameters(remove_duplicate=False))
#         loaded_params: Set[str] = set()
#         renamed_reg = [
#             (re.compile(r'block\.(\d+)\.layer\.0'), r'blocks.\1.self_attn'),
#             (re.compile(r'decoder.block\.(\d+)\.layer\.1'),
#              r'decoder.blocks.\1.cross_attn'),
#             (re.compile(r'decoder.block\.(\d+)\.layer\.2'),
#              r'decoder.blocks.\1.ffn'),
#             # encoder has no cross-attn, but rather self-attention+ffn.
#             (re.compile(r'encoder.block\.(\d+)\.layer\.1'),
#              r'encoder.blocks.\1.ffn'),
#             (re.compile(r'\.o\.'), r'.out_proj.'),
#         ]
#         stacked_params_mapping = [
#             # (param_name, shard_name, shard_id)
#             (".qkv_proj.", ".q.", "q"),
#             (".qkv_proj.", ".k.", "k"),
#             (".qkv_proj.", ".v.", "v")
#         ]

#         for name, loaded_weight in weights:
#             # No relative position attn bias on cross attention.
#             if name in self._keys_to_ignore_on_load_unexpected:
#                 continue

#             # Handle some renaming
#             for reg in renamed_reg:
#                 name = re.sub(*reg, name)

#             top_module, _ = name.split('.', 1)
#             if top_module != 'lm_head':
#                 name = f"model.{name}"

#             # Split q/k/v layers to unified QKVParallelLinear
#             for (param_name, weight_name, shard_id) in stacked_params_mapping:
#                 if weight_name not in name:
#                     continue
#                 name = name.replace(weight_name, param_name)
#                 param = model_params_dict[name]
#                 weight_loader = param.weight_loader
#                 weight_loader(param, loaded_weight, shard_id)
#                 break
#             else:
#                 # Not a q/k/v layer.
#                 param = model_params_dict[name]
#                 weight_loader = getattr(param, "weight_loader",
#                                         default_weight_loader)
#                 weight_loader(param, loaded_weight)
#             loaded_params.add(name)
#         return loaded_params