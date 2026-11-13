# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 text encoder: hybrid Gated DeltaNet linear attention + gated full attention.

The encoder runs replicated (no tensor or sequence parallelism) and
inference-only, so the linear-attention layers use a pure-PyTorch chunked gated
delta rule instead of the fused ``fla`` / ``causal_conv1d`` kernels. Full
attention calls ``scaled_dot_product_attention`` directly with head-major
``[B, H, L, D]`` inputs: that is the layout the reference text model feeds SDPA,
and keeping it makes bf16 hidden states reproducible against that reference.
"""
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.models.encoders.qwen3_5 import Qwen3_5TextArchConfig, Qwen3_5TextConfig
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.quantization import QuantizationConfig
from fastvideo.layers.vocab_parallel_embedding import VocabParallelEmbedding
from fastvideo.logger import init_logger
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.models.loader.weight_utils import default_weight_loader

logger = init_logger(__name__)

_GATED_DELTA_CHUNK_SIZE = 64
# Checkpoints saved from the multimodal wrapper nest the text tower.
_CHECKPOINT_PREFIXES = ("model.language_model.", "language_model.", "model.")
_SKIPPED_CHECKPOINT_PREFIXES = ("lm_head.", "mtp.", "visual.", "model.visual.")


class Qwen3_5RMSNorm(nn.Module):
    """Zero-centered RMSNorm: the stored weight is an offset from one."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        # The scale is applied in float32 before the downcast.
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Qwen3_5RMSNormGated(nn.Module):
    """Per-head RMSNorm followed by a SiLU output gate (plain, not zero-centered, weight)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = _GATED_DELTA_CHUNK_SIZE,
) -> torch.Tensor:
    """Chunked gated delta rule over a full prefix with a zero initial state.

    Inputs are ``[batch, seq, heads, dim]`` (``g`` / ``beta`` are
    ``[batch, seq, heads]``); q/k must already be L2-normalized. The recurrence
    runs in float32 and the output is returned in the query dtype.
    """
    output_dtype = query.dtype
    query, key, value, beta, g = (x.transpose(1, 2).contiguous().to(torch.float32)
                                  for x in (query, key, value, beta, g))

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1]**0.5))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = (x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                                         for x in (query, key, value, k_beta, v_beta))
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    upper_with_diagonal = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
                                     diagonal=0)

    # Intra-chunk decay and the (I + strictly-lower)^-1 WY transform, solved by
    # forward substitution row by row.
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(upper_with_diagonal, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    recurrent_state = torch.zeros(batch_size,
                                  num_heads,
                                  k_head_dim,
                                  v_head_dim,
                                  dtype=value.dtype,
                                  device=value.device)
    core_attn_out = torch.zeros_like(value)
    for i in range(total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        # decay_mask is already zero above the diagonal, which keeps this causal.
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        recurrent_state = (recurrent_state * g[:, :, i, -1, None, None].exp() +
                           (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new)

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    return core_attn_out.transpose(1, 2).contiguous().to(output_dtype)


class Qwen3_5GatedDeltaNet(nn.Module):
    """Gated DeltaNet token mixer used by the ``linear_attention`` layers."""

    def __init__(
        self,
        config: Qwen3_5TextArchConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        if self.num_v_heads % self.num_k_heads:
            raise ValueError(f"Qwen3.5 linear value heads ({self.num_v_heads}) must be a multiple of key heads "
                             f"({self.num_k_heads})")
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported Qwen3.5 activation: {config.hidden_act}")

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = ReplicatedLinear(self.value_dim,
                                         self.hidden_size,
                                         bias=False,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.out_proj")
        self.in_proj_qkv = ReplicatedLinear(self.hidden_size,
                                            self.conv_dim,
                                            bias=False,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.in_proj_qkv")
        self.in_proj_z = ReplicatedLinear(self.hidden_size,
                                          self.value_dim,
                                          bias=False,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.in_proj_z")
        self.in_proj_b = ReplicatedLinear(self.hidden_size,
                                          self.num_v_heads,
                                          bias=False,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.in_proj_b")
        self.in_proj_a = ReplicatedLinear(self.hidden_size,
                                          self.num_v_heads,
                                          bias=False,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.in_proj_a")

    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if padding_mask is not None:
            # Zero padded tokens so they cannot leak into the recurrent state.
            hidden_states = (hidden_states * padding_mask[:, :, None]).to(hidden_states.dtype)
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv, _ = self.in_proj_qkv(hidden_states)
        z, _ = self.in_proj_z(hidden_states)
        b, _ = self.in_proj_b(hidden_states)
        a, _ = self.in_proj_a(hidden_states)

        # Causal depthwise conv: left padding of kernel-1, keep the first seq_len outputs.
        mixed_qkv = mixed_qkv.transpose(1, 2)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv.to(self.conv1d.weight.dtype))[:, :, :seq_len]).to(mixed_qkv.dtype)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # float32 keeps exp(A_log) finite for half-precision weights.
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads > self.num_k_heads:
            repeats = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)

        core_attn_out = chunk_gated_delta_rule(_l2norm(query), _l2norm(key), value, g=g, beta=beta)

        core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        output, _ = self.out_proj(core_attn_out)
        return output


class Qwen3_5RotaryEmbedding(nn.Module):
    """Text-only partial RoPE.

    Qwen3.5's interleaved M-RoPE collapses to standard RoPE when the temporal,
    height and width positions are identical, which holds for text tokens.
    """

    def __init__(self, config: Qwen3_5TextArchConfig) -> None:
        super().__init__()
        rope = config.resolved_rope_parameters()
        if rope["rope_type"] != "default":
            raise NotImplementedError(f"Qwen3.5 rope_type {rope['rope_type']!r} is not supported")
        rotary_dim = int(config.head_dim * rope["partial_rotary_factor"])
        inv_freq = 1.0 / (rope["rope_theta"]**(
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cpu") / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=hidden_states.device, dtype=torch.float32)
        device_type = hidden_states.device.type if hidden_states.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            inv_freq = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
            freqs = (inv_freq @ position_ids[:, None, :].float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch_size, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None].expand(batch_size, num_kv_heads, groups, seq_len, head_dim)
    return x.reshape(batch_size, num_kv_heads * groups, seq_len, head_dim)


def _apply_partial_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the leading ``cos.shape[-1]`` channels of ``[B, H, L, D]`` and pass the rest through."""
    rotary_dim = cos.shape[-1]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rot = x_rot * cos + _rotate_half(x_rot) * sin
    return torch.cat([x_rot, x_pass], dim=-1)


class Qwen3_5Attention(nn.Module):
    """Gated full attention with per-head QK-norm, partial RoPE and GQA."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        arch = config.arch_config
        self.num_heads = arch.num_attention_heads
        self.num_kv_heads = arch.num_key_value_heads
        if self.num_heads % self.num_kv_heads:
            raise ValueError(f"Qwen3.5 attention heads ({self.num_heads}) must be a multiple of KV heads "
                             f"({self.num_kv_heads})")
        self.head_dim = arch.head_dim
        self.scaling = self.head_dim**-0.5
        # q_proj packs, per head, the query followed by its output gate.
        self.q_proj = ReplicatedLinear(arch.hidden_size,
                                       self.num_heads * self.head_dim * 2,
                                       bias=arch.attention_bias,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.q_proj")
        self.k_proj = ReplicatedLinear(arch.hidden_size,
                                       self.num_kv_heads * self.head_dim,
                                       bias=arch.attention_bias,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.k_proj")
        self.v_proj = ReplicatedLinear(arch.hidden_size,
                                       self.num_kv_heads * self.head_dim,
                                       bias=arch.attention_bias,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.v_proj")
        self.o_proj = ReplicatedLinear(self.num_heads * self.head_dim,
                                       arch.hidden_size,
                                       bias=arch.attention_bias,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.o_proj")
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=arch.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=arch.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        packed, _ = self.q_proj(hidden_states)
        query, gate = torch.chunk(packed.view(batch_size, seq_len, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, -1)
        key, _ = self.k_proj(hidden_states)
        value, _ = self.v_proj(hidden_states)

        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key.view(batch_size, seq_len, -1, self.head_dim)).transpose(1, 2)
        value = value.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        query = _apply_partial_rotary(query, cos, sin)
        key = _apply_partial_rotary(key, cos, sin)

        groups = self.num_heads // self.num_kv_heads
        # Native GQA is only used where SDPA keeps a fused kernel (no mask, head_dim <= 256).
        native_gqa = padding_mask is None and self.head_dim <= 256
        if groups > 1 and not native_gqa:
            key = _repeat_kv(key, groups)
            value = _repeat_kv(value, groups)
        if padding_mask is None:
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=seq_len > 1,
                scale=self.scaling,
                enable_gqa=groups > 1 and native_gqa,
            )
        else:
            causal = torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool).tril()
            mask = causal[None, None] & padding_mask.to(device=query.device, dtype=torch.bool)[:, None, None, :]
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3_5MLP(nn.Module):

    def __init__(
        self,
        config: Qwen3_5TextArchConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported Qwen3.5 activation: {config.hidden_act}")
        self.gate_proj = ReplicatedLinear(config.hidden_size,
                                          config.intermediate_size,
                                          bias=False,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.gate_proj")
        self.up_proj = ReplicatedLinear(config.hidden_size,
                                        config.intermediate_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.up_proj")
        self.down_proj = ReplicatedLinear(config.intermediate_size,
                                          config.hidden_size,
                                          bias=False,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.down_proj")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        output, _ = self.down_proj(F.silu(gate) * up)
        return output


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_type: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        arch = config.arch_config
        self.layer_type = layer_type
        if layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(arch, quant_config=quant_config, prefix=f"{prefix}.linear_attn")
        else:
            self.self_attn = Qwen3_5Attention(config, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        self.mlp = Qwen3_5MLP(arch, quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = Qwen3_5RMSNorm(arch.hidden_size, eps=arch.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(arch.hidden_size, eps=arch.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, padding_mask)
        else:
            hidden_states = self.self_attn(hidden_states, position_embeddings, padding_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3_5TextModel(TextEncoder):
    """Qwen3.5 text tower used as a diffusion text encoder (e.g. PAN2).

    ``hidden_states`` follows the Transformers contract: entry 0 is the token
    embeddings, entry ``i`` is the output of layer ``i - 1``, and the last entry
    is the final-normed output (it replaces the raw last-layer output).
    """

    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__(config)
        self.config = config
        arch = config.arch_config
        self.quant_config = getattr(config, "quant_config", None)
        self.layer_types = arch.resolved_layer_types()

        self.embed_tokens = VocabParallelEmbedding(
            arch.vocab_size,
            arch.hidden_size,
            org_num_embeddings=arch.vocab_size,
            quant_config=self.quant_config,
        )
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(
                config,
                layer_type,
                quant_config=self.quant_config,
                prefix=f"{config.prefix}.layers.{index}",
            ) for index, layer_type in enumerate(self.layer_types)
        ])
        self.norm = Qwen3_5RMSNorm(arch.hidden_size, eps=arch.rms_norm_eps)
        self.rotary_emb = Qwen3_5RotaryEmbedding(arch)

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
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.arch_config.output_hidden_states)
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            assert input_ids is not None
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            inputs_embeds = self.get_input_embeddings(input_ids)
        hidden_states = inputs_embeds
        batch_size, seq_len = hidden_states.shape[:2]

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        elif position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        elif position_ids.ndim != 2:
            raise ValueError("Qwen3.5 text encoder expects text position_ids shaped [batch, seq]")

        # An all-ones mask is equivalent to no mask; only real padding takes the masked path.
        padding_mask = None
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError("Qwen3.5 text encoder expects a 2D [batch, seq] padding mask")
            if not bool(attention_mask.to(torch.bool).all()):
                padding_mask = attention_mask.to(hidden_states.device)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states: tuple[torch.Tensor, ...] | None = () if output_hidden_states else None
        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states, )
            hidden_states = layer(hidden_states, position_embeddings, padding_mask)

        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states += (hidden_states, )

        return BaseEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attention_mask=attention_mask,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name.startswith(_SKIPPED_CHECKPOINT_PREFIXES) or "rotary_emb.inv_freq" in name:
                continue
            for checkpoint_prefix in _CHECKPOINT_PREFIXES:
                if name.startswith(checkpoint_prefix):
                    name = name[len(checkpoint_prefix):]
                    break
            if name not in params_dict:
                logger.debug("Skipping unexpected Qwen3.5 checkpoint tensor %s", name)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


EntryClass = Qwen3_5TextModel
