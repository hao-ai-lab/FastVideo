# SPDX-License-Identifier: Apache-2.0
"""Qwen3 text encoders for FastVideo diffusion models.

Two ports coexist in this module, dispatched by HF architecture name in
``fastvideo/models/registry.py``:

- ``Qwen3Model`` (+ ``OvisQwen3*`` helpers): the Qwen3-2B / Ovis2.5-2B encoder
  used by Ovis-Image-7B. Custom RoPE + SDPA, chat-template path.
- ``Qwen3ForCausalLM`` (+ ``Qwen3*`` helpers): ported from SGLang, used by
  Flux2 Klein. Uses ``get_rope`` + ``LocalAttention`` and ``output_hidden_states``.

The two sets of internal helper classes are intentionally namespaced apart
(``OvisQwen3MLP`` vs ``Qwen3MLP``, etc.) so both encoders can live side by side.
"""
from collections.abc import Iterable
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.attention import LocalAttention
from fastvideo.configs.models.encoders import BaseEncoderOutput, Qwen3Config
from fastvideo.configs.models.encoders.qwen3 import Qwen3TextConfig
from fastvideo.distributed import get_tp_world_size
from fastvideo.layers.activation import SiluAndMul
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.linear import (MergedColumnParallelLinear,
                                     QKVParallelLinear, RowParallelLinear)
from fastvideo.layers.quantization import QuantizationConfig
from fastvideo.layers.rotary_embedding import get_rope
from fastvideo.layers.vocab_parallel_embedding import VocabParallelEmbedding
from fastvideo.logger import init_logger
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.models.loader.weight_utils import (default_weight_loader,
                                                  maybe_remap_kv_scale_name)
from fastvideo.models.mask_utils import sdpa_mask

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Ovis-Image Qwen3 encoder (Qwen3-2B / Ovis2.5-2B). Native FastVideo port.
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


class OvisQwen3MLP(nn.Module):
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


class OvisQwen3Attention(nn.Module):
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


class OvisQwen3DecoderLayer(nn.Module):

    def __init__(self, config: Qwen3Config, layer_idx: int,
                 quant_config: QuantizationConfig | None = None,
                 prefix: str = ""):
        super().__init__()
        arch = config.arch_config
        self.self_attn = OvisQwen3Attention(config,
                                            layer_idx,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.self_attn")
        self.mlp = OvisQwen3MLP(config,
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


class Qwen3Model(TextEncoder):
    """
    Native FastVideo Qwen3 text encoder (Ovis-Image).

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
            OvisQwen3DecoderLayer(
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


# ---------------------------------------------------------------------------
# Flux2 Klein Qwen3 encoder. Ported from SGLang:
# python/sglang/multimodal_gen/runtime/models/encoders/qwen3.py
# ---------------------------------------------------------------------------
class Qwen3MLP(nn.Module):
    """Qwen3 MLP with SwiGLU activation and tensor parallelism."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class Qwen3Attention(nn.Module):
    """Qwen3 attention with QK-Norm and tensor parallelism.

    Key difference from LLaMA: RMSNorm is applied to Q and K before attention.
    """

    def __init__(
        self,
        config: Qwen3TextConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 1000000.0,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 40960,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tp_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.rotary_dim = self.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )

        self.attn = LocalAttention(
            self.num_heads,
            self.head_dim,
            self.num_kv_heads,
            softmax_scale=self.scaling,
            causal=True,
            supported_attention_backends=config._supported_attention_backends,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        batch_size, seq_len = q.shape[0], q.shape[1]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.reshape(batch_size, seq_len, -1)
        k = k.reshape(batch_size, seq_len, -1)

        q, k = self.rotary_emb(positions, q, k)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if attention_mask is None:
            attn_output = self.attn(q, k, v)
        else:
            q_sdpa = q.transpose(1, 2)
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)
            causal_mask = torch.ones(
                seq_len,
                seq_len,
                device=q.device,
                dtype=torch.bool,
            ).tril()
            key_mask = attention_mask.to(device=q.device, dtype=torch.bool)
            attn_mask = causal_mask[None, None, :, :] & key_mask[:, None, None, :]
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
                enable_gqa=self.num_heads != self.num_kv_heads,
            ).transpose(1, 2)

        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 transformer decoder layer."""

    def __init__(
        self,
        config: Qwen3TextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 40960)
        attention_bias = getattr(config, "attention_bias", False)

        self.self_attn = Qwen3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
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
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3ForCausalLM(TextEncoder):
    """Qwen3 causal language model for text encoding in diffusion models (e.g. Flux2 Klein).

    Features:
    - Tensor parallelism support
    - FlashAttention/SDPA support via LocalAttention
    - QK-Norm for better training stability
    - output_hidden_states for Klein (layers 9, 18, 27)
    """

    supports_hf_from_pretrained = True

    def __init__(self, config: Qwen3TextConfig) -> None:
        super().__init__(config)

        self.config = config
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
            [
                Qwen3DecoderLayer(
                    config=config,
                    quant_config=self.quant_config,
                    prefix=f"{config.prefix}.layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @classmethod
    def from_pretrained_local(
        cls,
        model_path: str,
        model_config: Qwen3TextConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> nn.Module:
        from transformers import AutoModelForCausalLM

        if device.type == "cpu" and torch.cuda.is_available():
            from fastvideo.distributed import get_local_torch_device

            device = get_local_torch_device()

        return AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).eval().to(device)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> BaseEncoderOutput:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            assert input_ids is not None
            hidden_states = self.get_input_embeddings(input_ids)

        residual = None

        if position_ids is None:
            position_ids = torch.arange(
                0, hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)

        all_hidden_states: tuple[Any, ...] | None = (
            () if output_hidden_states else None
        )

        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states += (
                    (hidden_states,)
                    if residual is None
                    else (hidden_states + residual,)
                )
            hidden_states, residual = layer(
                position_ids,
                hidden_states,
                residual,
                attention_mask=attention_mask,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        return BaseEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name.startswith("model."):
                name = name[6:]

            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            if "scale" in name:
                kv_scale_name: str | None = maybe_remap_kv_scale_name(
                    name, params_dict
                )
                if kv_scale_name is None:
                    continue
                name = kv_scale_name

            for (
                param_name,
                weight_name,
                shard_id,
            ) in self.config.arch_config.stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)

        return loaded_params


EntryClass = Qwen3ForCausalLM
