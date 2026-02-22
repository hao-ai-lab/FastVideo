# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 Text Encoder — Approach B (FastVideo-native implementation)

Architecture (Qwen3-2B as used in Ovis-Image-7B):
  - Standard transformer decoder (no MLLM vision, pure text)
  - GQA attention (16 Q heads, 8 KV heads)
  - QK-Norm: RMSNorm applied to Q and K before attention (Qwen3 specific)
  - SwiGLU MLP via MergedColumnParallelLinear + RowParallelLinear
  - RoPE position embeddings (standard, not multi-modal)
  - Tensor Parallelism via QKVParallelLinear / RowParallelLinear
  - Quantization support via quant_config
  - Proper weight loading with stacked_params_mapping

Adapted from fastvideo/models/encoders/qwen2_5.py with the following deltas:
  1. QK-Norm after Q/K split (Qwen3 adds q_norm, k_norm per attention layer)
  2. Standard RoPE (not multi-modal / mrope)
  3. attention_bias=False by default
"""

import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.configs.models.encoders import BaseEncoderOutput, Qwen3Config
from fastvideo.distributed import get_tp_rank, get_tp_world_size
from fastvideo.layers.activation import SiluAndMul
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.linear import (MergedColumnParallelLinear,
                                     QKVParallelLinear, RowParallelLinear)
from fastvideo.layers.quantization import QuantizationConfig
from fastvideo.layers.vocab_parallel_embedding import VocabParallelEmbedding
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.models.loader.weight_utils import default_weight_loader
from fastvideo.models.mask_utils import sdpa_mask
from fastvideo.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


class Qwen3RotaryEmbedding(nn.Module):
    """Standard RoPE for Qwen3 (non-multimodal)."""

    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        arch = config.arch_config
        self.head_dim = arch.head_dim
        self.base = arch.rope_theta
        self.attention_scaling = 1.0

        half = self.head_dim // 2
        inv_freq = 1.0 / (
            self.base ** (
                torch.arange(0, half, dtype=torch.int64).float().to(device) /
                half))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, seq, head_dim] (used only for device/dtype)
            position_ids: [B, seq]
        Returns:
            (cos, sin) each [B, seq, head_dim]
        """
        inv_freq_expanded = (self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1))
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @
                     position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings. q/k: [B, heads, seq, head_dim]."""
    cos = cos.unsqueeze(1)  # [B, 1, seq, head_dim]
    sin = sin.unsqueeze(1)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class Qwen3MLP(nn.Module):
    """SwiGLU MLP with Tensor-Parallel linear layers."""

    def __init__(self, config: Qwen3Config,
                 quant_config: QuantizationConfig | None = None,
                 prefix: str = ""):
        super().__init__()
        arch = config.arch_config
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=arch.hidden_size,
            output_sizes=[arch.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=arch.intermediate_size,
            output_size=arch.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Qwen3Attention(nn.Module):
    """
    GQA attention with QK-Norm and Tensor Parallelism.

    QK-Norm is the main architectural difference from Qwen2: after splitting
    QKV, RMSNorm is applied to Q and K before computing attention.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int,
                 quant_config: QuantizationConfig | None = None,
                 prefix: str = ""):
        super().__init__()
        arch = config.arch_config
        self.hidden_size = arch.hidden_size
        self.head_dim = arch.head_dim
        self.num_heads = arch.num_attention_heads
        self.num_kv_heads = arch.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        tp_size = get_tp_world_size()
        assert self.num_heads % tp_size == 0
        self.num_heads_per_rank = self.num_heads // tp_size
        # KV heads: if fewer than tp_size, replicate across ranks
        self.num_kv_heads_per_rank = max(1, self.num_kv_heads // tp_size)

        self.q_size = self.num_heads_per_rank * self.head_dim
        self.kv_size = self.num_kv_heads_per_rank * self.head_dim
        self.num_kv_groups = self.num_heads_per_rank // self.num_kv_heads_per_rank

        # Qwen3 uses attention_bias=False for QKV, but True for o_proj
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=arch.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # QK-Norm: Qwen3-specific
        self.q_norm = RMSNorm(self.head_dim, eps=arch.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=arch.rms_norm_eps)

        # Sliding window (layer-dependent in Qwen3)
        layer_types = arch.layer_types or []
        layer_type = (layer_types[layer_idx]
                      if layer_idx < len(layer_types) else "full_attention")
        self.sliding_window = (arch.sliding_window
                               if layer_type == "sliding_attention" else None)

    @staticmethod
    def _repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return hidden
        B, heads, seq, dim = hidden.shape
        return hidden[:, :, None, :, :].expand(B, heads, n_rep, seq,
                                               dim).reshape(B, heads * n_rep,
                                                            seq, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # [B, seq, heads, head_dim]
        q = q.view(bsz, q_len, self.num_heads_per_rank, self.head_dim)
        k = k.view(bsz, q_len, self.num_kv_heads_per_rank, self.head_dim)
        v = v.view(bsz, q_len, self.num_kv_heads_per_rank, self.head_dim)

        # QK-Norm (Qwen3-specific)
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)

        # [B, heads, seq, head_dim] for RoPE + attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rope(q, k, cos, sin)

        # GQA: expand KV to match Q head count
        k = self._repeat_kv(k, self.num_kv_groups)
        v = self._repeat_kv(v, self.num_kv_groups)

        # SDPA
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scaling,
            is_causal=False,
        )

        # [B, seq, hidden]
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_out, _ = self.o_proj(attn_out)
        return attn_out


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class Qwen3DecoderLayer(nn.Module):

    def __init__(self, config: Qwen3Config, layer_idx: int,
                 quant_config: QuantizationConfig | None = None,
                 prefix: str = ""):
        super().__init__()
        arch = config.arch_config
        self.self_attn = Qwen3Attention(config,
                                        layer_idx,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.self_attn")
        self.mlp = Qwen3MLP(config,
                            quant_config=quant_config,
                            prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(arch.hidden_size, eps=arch.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(arch.hidden_size,
                                               eps=arch.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class Qwen3Model(TextEncoder):
    """
    Native FastVideo Qwen3 text encoder.

    Supports:
      - Tensor Parallelism (QKVParallelLinear + RowParallelLinear)
      - INT8/FP8 quantization (via quant_config)
      - FSDP sharding (conditions defined in Qwen3ArchConfig)
      - Proper HuggingFace weight loading (stacked_params_mapping)
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        arch = config.arch_config
        quant_config = getattr(config, "quant_config", None)

        self.embed_tokens = VocabParallelEmbedding(
            arch.vocab_size,
            arch.hidden_size,
            org_num_embeddings=arch.vocab_size,
        )
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                config,
                layer_idx,
                quant_config=quant_config,
                prefix=f"model.layers.{layer_idx}",
            ) for layer_idx in range(arch.num_hidden_layers)
        ])
        self.norm = RMSNorm(arch.hidden_size, eps=arch.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        arch = self.config.arch_config
        output_hidden_states = (output_hidden_states if output_hidden_states
                                is not None else
                                getattr(arch, "output_hidden_states", False))

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)

        hidden_states = inputs_embeds
        seq_length = hidden_states.shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length,
                                        device=hidden_states.device).unsqueeze(0)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Build SDPA attention mask
        cache_position = torch.arange(seq_length, device=hidden_states.device)
        sdpa_attn_mask = None
        if attention_mask is not None:
            sdpa_attn_mask = sdpa_mask(
                batch_size=hidden_states.shape[0],
                cache_position=cache_position,
                kv_length=attention_mask.shape[-1],
                kv_offset=0,
                attention_mask=attention_mask,
            )

        all_hidden_states: tuple | None = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=sdpa_attn_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

    def load_weights(
            self,
            weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights from HuggingFace checkpoint using stacked_params_mapping
        to fuse q_proj/k_proj/v_proj -> qkv_proj and gate_proj/up_proj -> gate_up_proj.
        """
        arch = self.config.arch_config
        stacked_params_mapping = getattr(arch, "stacked_params_mapping", [])

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Try stacked param mapping first
            matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    continue
                param = params_dict[name_mapped]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_mapped)
                matched = True
                break

            if matched:
                continue

            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
