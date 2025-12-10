# SPDX-License-Identifier: Apache-2.0
# Adapted from transformers: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py

import math
from typing import Any, Optional, Tuple, Union, List, Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.configs.models.encoders import BaseEncoderOutput, Qwen2_5_VLConfig
from fastvideo.distributed import get_tp_rank, get_tp_world_size
from fastvideo.layers.activation import get_act_fn, SiluAndMul
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from fastvideo.layers.quantization import QuantizationConfig
from fastvideo.layers.vocab_parallel_embedding import VocabParallelEmbedding
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.models.loader.weight_utils import default_weight_loader
from fastvideo.models.mask_utils import sdpa_mask
from fastvideo.logger import init_logger

logger = init_logger(__name__)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    # If attention_mask is not None, convert it to boolean type
    if attention_mask is not None and attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.bool()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = [s * 2 for s in mrope_section]
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2_5_VLRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        
        self.rope_type = config.rope_scaling.get("rope_type", "default")
        self.base = config.rope_theta
        
        # Simplified initialization
        head_dim = config.hidden_size // config.num_attention_heads
        dim = head_dim
        self.attention_scaling = 1.0

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    def forward(self, x, position_ids):
        # In contrast to other models, Qwen2_5_VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, quant_config: QuantizationConfig | None = None, prefix: str = ""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class Qwen2_5_VLAttention(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int, quant_config: QuantizationConfig | None = None, prefix: str = ""):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        tp_size = get_tp_world_size()
        self.total_num_heads = self.num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        
        self.total_num_kv_heads = self.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
             assert self.total_num_kv_heads % tp_size == 0
             self.num_kv_heads = self.total_num_kv_heads // tp_size
        else:
             assert tp_size % self.total_num_kv_heads == 0
             self.num_kv_heads = 1 

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
            prefix=f"{prefix}.qkv_proj",
        )
        
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        
        self.layer_type = config.layer_types[layer_idx] if config.layer_types else "full_attention"
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        qkv, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.config.rope_scaling["mrope_section"]
        )

        attn_output = sdpa_attention_forward(self, query_states, key_states, value_states, attention_mask, dropout=self.config.attention_dropout, scaling=self.scaling, is_causal=False)[0].reshape(bsz, q_len, -1)
        attn_output, _ = self.o_proj(attn_output)
        
        return attn_output

class Qwen2_5_VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int, quant_config: QuantizationConfig | None = None, prefix: str = ""):
        super().__init__()
        self.self_attn = Qwen2_5_VLAttention(config, layer_idx, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        self.mlp = Qwen2_5_VLMLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
            
        return outputs

class Qwen2_5_VLTextModel(TextEncoder):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)

        quant_config = None

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size
        )
        
        self.layers = nn.ModuleList([
            Qwen2_5_VLDecoderLayer(config, layer_idx, quant_config=quant_config, prefix=f"{config.prefix}.layers.{layer_idx}")
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)
            
        hidden_states = inputs_embeds
        
        if position_ids is None:
            seq_length = hidden_states.shape[1]
            cache_position = torch.arange(seq_length, device=hidden_states.device)
            position_ids = cache_position.view(1, 1, -1).expand(3, hidden_states.shape[0], -1)

        mask_kwargs = {
            "batch_size": hidden_states.shape[0],
            "cache_position": cache_position,
            "kv_length": attention_mask.shape[-1],
            "kv_offset": 0,
            "attention_mask": attention_mask
        }

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        all_hidden_states = () if output_hidden_states else None
        
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=sdpa_mask(**mask_kwargs),
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]
            
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        return BaseEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in self.config.stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                # if name.endswith(".bias") and name not in params_dict:
                #     continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                # if name.endswith(".bias") and name not in params_dict:
                #     continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
