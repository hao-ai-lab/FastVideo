# SPDX-License-Identifier: Apache-2.0
"""Native LingBot-Video Qwen3-VL language model for text-only conditioning."""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.vocab_parallel_embedding import VocabParallelEmbedding
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.models.encoders.qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
)


class LingBotVideoQwen3VLAttention(Qwen3Attention):
    """Qwen3-VL attention with the official masked repeat-K/V SDPA path."""

    def _apply_qwen3_vl_rope(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply NeoX RoPE with Qwen3-VL's input-dtype multiply-add ordering."""
        flat_positions = positions.flatten()
        cos_sin = self.rotary_emb.cos_sin_cache.index_select(0, flat_positions)
        cos_half, sin_half = cos_sin.chunk(2, dim=-1)
        cos = torch.cat((cos_half, cos_half), dim=-1).to(query.dtype)
        sin = torch.cat((sin_half, sin_half), dim=-1).to(query.dtype)
        if flat_positions.numel() == query.shape[1]:
            cos = cos.view(1, query.shape[1], 1, self.head_dim)
            sin = sin.view(1, query.shape[1], 1, self.head_dim)
        else:
            cos = cos.view(*query.shape[:2], 1, self.head_dim)
            sin = sin.view(*query.shape[:2], 1, self.head_dim)

        def rotate(tensor: torch.Tensor) -> torch.Tensor:
            first, second = tensor.chunk(2, dim=-1)
            rotated = torch.cat((-second, first), dim=-1)
            return tensor * cos + rotated * sin

        return rotate(query), rotate(key)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply fused projections, QK norm, RoPE, and causal grouped attention."""
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        batch_size, sequence_length = query.shape[:2]
        query = query.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, sequence_length, self.num_kv_heads, self.head_dim)
        value = value.reshape(batch_size, sequence_length, self.num_kv_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = self._apply_qwen3_vl_rope(positions, query, key)
        no_padding = attention_mask is None
        if no_padding:
            attention_output = torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                dropout_p=0.0,
                is_causal=sequence_length > 1,
                scale=self.scaling,
                enable_gqa=self.num_heads != self.num_kv_heads,
            ).transpose(1, 2)
        else:
            groups = self.num_heads // self.num_kv_heads
            key = (key[:, :, :, None, :].expand(-1, -1, -1, groups, -1).reshape(batch_size, sequence_length,
                                                                                self.num_heads, self.head_dim))
            value = (value[:, :, :, None, :].expand(-1, -1, -1, groups, -1).reshape(batch_size, sequence_length,
                                                                                    self.num_heads, self.head_dim))
            causal_mask = torch.ones(sequence_length, sequence_length, device=query.device, dtype=torch.bool).tril()
            key_mask = attention_mask.to(device=query.device, dtype=torch.bool)
            sdpa_mask = causal_mask[None, None, :, :] & key_mask[:, None, None, :]
            attention_output = torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=sdpa_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            ).transpose(1, 2)
        output, _ = self.o_proj(attention_output.reshape(batch_size, sequence_length, -1))
        return output


class LingBotVideoQwen3VLDecoderLayer(Qwen3DecoderLayer):
    """Qwen3-VL decoder layer with explicit official residual rounding order."""

    def __init__(self, config: Any, prefix: str) -> None:
        """Build the final Qwen3-VL attention once to avoid orphan parameters."""
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        quant_config = getattr(config, "quant_config", None)
        self.self_attn = LingBotVideoQwen3VLAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            bias=config.attention_bias,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run attention and MLP with each residual sum rounded before normalization."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class LingBotVideoQwen3VLTextModel(Qwen3ForCausalLM):
    """Load the Qwen3-VL language-model subset without its vision tower or LM head."""

    supports_hf_from_pretrained = False

    def __init__(self, config) -> None:
        """Construct the exact Qwen3-VL module graph without replacing base layers."""
        TextEncoder.__init__(self, config)
        self.quant_config = getattr(config, "quant_config", None)
        if getattr(config, "lora_config", None) is not None:
            max_loras = getattr(config.lora_config, "max_loras", 1)
            lora_vocab_size = getattr(config.lora_config, "lora_extra_vocab_size", 1)
            lora_vocab = lora_vocab_size * max_loras
        else:
            lora_vocab = 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=self.quant_config,
        )
        self.layers = nn.ModuleList(
            LingBotVideoQwen3VLDecoderLayer(config, prefix=f"{config.prefix}.layers.{index}")
            for index in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> BaseEncoderOutput:
        """Run explicit Qwen3-VL layers and return the requested hidden-state tuple."""
        del kwargs
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            hidden_states = self.get_input_embeddings(input_ids)
        else:
            hidden_states = inputs_embeds
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        if attention_mask is not None and bool(attention_mask.to(torch.bool).all()):
            attention_mask = None
        all_hidden_states: tuple[torch.Tensor, ...] | None = () if output_hidden_states else None
        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states, )
            hidden_states = layer(position_ids, hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states += (hidden_states, )
        return BaseEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Accept either official compound keys or converted native keys."""
        prefix = "model.language_model."
        language_weights = ((name[len(prefix):] if name.startswith(prefix) else name, tensor)
                            for name, tensor in weights
                            if name.startswith(prefix) or not name.startswith(("model.", "lm_head.")))
        return super().load_weights(language_weights)


EntryClass = LingBotVideoQwen3VLTextModel
