# SPDX-License-Identifier: Apache-2.0

from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn
from transformers import BertConfig

# from vllm.attention import Attention, AttentionType
from fastvideo.v1.attention import LocalAttention
# from vllm.compilation.decorators import support_torch_compile
# from vllm.config import CacheConfig, PoolerConfig, VllmConfig
# CacheConfig is used for attention, could be replaced with LocalAttention
# VllmConfig is replaced with BertConfig (needs to implement)
from fastvideo.v1.configs.models.encoders.bert import BertConfig
# from vllm.distributed import get_tensor_model_parallel_world_size
from fastvideo.v1.distributed import (divide,
                                      get_tensor_model_parallel_world_size)
# from vllm.forward_context import get_forward_context
from fastvideo.v1.forward_context import get_forward_context
# from vllm.model_executor.layers.activation import get_act_fn
from fastvideo.v1.layers.activation import get_act_fn
# from vllm.model_executor.layers.linear import (ColumnParallelLinear,
#                                                QKVParallelLinear,
#                                                RowParallelLinear)
from fastvideo.v1.layers.linear import (ColumnParallelLinear, QKVParallelLinear,
                                        RowParallelLinear)
# from vllm.model_executor.layers.pooler import (CrossEncodingPooler, Pooler,
#                                                PoolingType)
# from vllm.model_executor.layers.quantization import QuantizationConfig
from fastvideo.v1.layers.quantization import QuantizationConfig
# from vllm.model_executor.layers.vocab_parallel_embedding import (
#     VocabParallelEmbedding)
from fastvideo.v1.layers.vocab_parallel_embedding import VocabParallelEmbedding
# from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from fastvideo.v1.models.loader.weight_utils import default_weight_loader
# from vllm.model_executor.pooling_metadata import PoolingMetadata
# from vllm.sequence import PoolerOutput
# from vllm.transformers_utils.config import (
#     get_cross_encoder_activation_function)
from fastvideo.v1.models.encoders.base import TextEncoder
# from .interfaces import SupportsCrossEncoding, SupportsQuant, SupportsV0Only
# from .utils import WeightsMapper, maybe_prefix

from fastvideo.v1.platforms import _Backend
class BertEmbedding(nn.Module):

    def __init__(self, config: BertConfig):

        super().__init__()
        self.size = config.hidden_size
        self.word_embeddings = VocabParallelEmbedding(config.vocab_size,
                                                      config.hidden_size)
        self.position_embeddings = VocabParallelEmbedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = VocabParallelEmbedding(
            config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        # self.position_ids = nn.Parameter(
        #     torch.empty((1, config.max_position_embeddings)), )

        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type != "absolute":
            raise ValueError("Only 'absolute' position_embedding_type" +
                             " is supported")

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_shape = input_ids.size()

        # Input embeddings.
        inputs_embeds = self.word_embeddings(input_ids)

        # Position embeddings.
        position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape,
                                         dtype=torch.long,
                                         device=inputs_embeds.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class BertPooler(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[0, :]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# @support_torch_compile
class BertEncoder(nn.Module):

    def __init__(self, config: BertConfig, quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__()
        # config = vllm_config.model_config.hf_config
        # cache_config = vllm_config.cache_config
        # quant_config = vllm_config.quant_config
        self.layer = nn.ModuleList([
            BertLayer(config=config,
                    #   cache_config=cache_config,
                      quant_config=quant_config,
                      prefix=f"{prefix}.layer.{layer_idx}")
            for layer_idx in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        print("hidden_states shape", hidden_states.shape)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self,
                 config: BertConfig,
                #  cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()

        self.attention = BertAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            layer_norm_eps=config.layer_norm_eps,
            # cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attention")

        self.intermediate = BertIntermediate(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.intermediate")

        self.output = BertOutput(hidden_size=config.hidden_size,
                                 intermediate_size=config.intermediate_size,
                                 layer_norm_eps=config.layer_norm_eps,
                                 quant_config=quant_config,
                                 prefix=f"{prefix}.output")

    def forward(self, hidden_states: torch.Tensor):
        attn_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attn_output)
        output = self.output(intermediate_output, attn_output)
        return output


class BertAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        layer_norm_eps: float,
        # cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.self = BertSelfAttention(hidden_size=hidden_size,
                                      num_attention_heads=num_attention_heads,
                                    #   cache_config=cache_config,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.output")

        self.output = BertSelfOutput(hidden_size=hidden_size,
                                     layer_norm_eps=layer_norm_eps,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.output")

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        self_output = self.self(hidden_states)
        return self.output(self_output, hidden_states)


class BertSelfAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        # cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_attention_heads
        assert self.total_num_heads % tp_size == 0

        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = self.hidden_size // self.total_num_heads
        assert self.head_dim * self.total_num_heads == self.hidden_size

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj")

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)
        
        self.attn = LocalAttention(num_heads=self.num_heads,
                              head_size=self.head_dim,
                              num_kv_heads=self.num_kv_heads,
                            #   softmax_scale=self.scaling,
                            #   cache_config=cache_config,
                            #   quant_config=quant_config,
                            #   prefix=f"{prefix}.attn",
                            #   attn_type=AttentionType.ENCODER_ONLY
                            causal=False,
                            supported_attention_backends=(_Backend.FLASH_ATTN,
                                                           _Backend.TORCH_SDPA)
                            ) # River TODO, fix hardcoded backend

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        B, L, _ = q.shape
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_heads, self.head_dim)
        v = v.reshape(B, L, self.num_heads, self.head_dim)
        output = self.attn(q, k, v)
        output = output.reshape(B, L, self.hidden_size)
        return output
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    # ) -> torch.Tensor:
    #     qkv, _ = self.qkv_proj(hidden_states)
    #     q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    #     output = self.attn(q, k, v)
    #     return output

class BertSelfOutput(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 layer_norm_eps: float,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.dense = RowParallelLinear(input_size=hidden_size,
                                       output_size=hidden_size,
                                       bias=True,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.dense")
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 hidden_act: str,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.dense = ColumnParallelLinear(input_size=hidden_size,
                                          output_size=intermediate_size,
                                          bias=True,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.dense")
        self.intermediate_act_fn = get_act_fn(hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 layer_norm_eps: float,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()

        self.dense = RowParallelLinear(input_size=intermediate_size,
                                       output_size=hidden_size,
                                       bias=True,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.dense")

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertModel(TextEncoder):
    packed_modules_mapping = {"qkv_proj": ["query", "key", "value"]}

    def __init__(
        self,
        config: BertConfig,
    ) -> None:
        super().__init__(config)
        self.embeddings = BertEmbedding(config=config)
        print("prefix here ",config.prefix)
        self.encoder = BertEncoder(config=config,
                                prefix=f"{config.prefix}.encoder")

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids)
        return self.encoder(hidden_states)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "query", "q"),
            ("qkv_proj", "key", "k"),
            ("qkv_proj", "value", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            name = name[len("bert."):] if name.startswith("bert.") else name
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
