# SPDX-License-Identifier: Apache-2.0
"""FastVideo-native MiniMax H3 joint audio-video diffusion transformer."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo import envs
from fastvideo.attention import DistributedAttention
from fastvideo.attention.layer import DistributedAttention_VSA
from fastvideo.attention.selector import get_attn_backend
from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3Config
from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather_with_unpad,
    sequence_model_parallel_shard,
)
from fastvideo.distributed.parallel_state import get_sp_world_size, model_parallel_is_initialized
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.layers.quantization import QuantizationConfig
from fastvideo.layers.visual_embedding import Timesteps
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.minimax_h3_fusions import (
    HAVE_TRITON,
    fused_qknorm_rope,
    fused_residual_gate_rmsnorm_modulate,
    fused_rmsnorm_modulate,
    minimax_h3_swiglu,
)
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.profiler import nvtx_range
from fastvideo.utils import get_compute_dtype

logger = init_logger(__name__)

MINIMAX_H3_MODALITY_NUM = 3
_CFG = MiniMaxH3Config()
_MINIMAX_H3_FUSION_NAMES = frozenset({"modulate", "qknorm_rope", "swiglu"})


def _enabled_minimax_h3_fusions(value: str | None = None) -> frozenset[str]:
    """Parse the independently switchable inference fusion set."""
    raw = envs.FASTVIDEO_MINIMAX_H3_FUSIONS if value is None else value
    normalized = raw.strip().lower()
    if normalized in {"", "0", "none"}:
        return frozenset()
    if normalized in {"1", "all"}:
        return _MINIMAX_H3_FUSION_NAMES
    enabled = frozenset(item.strip() for item in normalized.split(",") if item.strip())
    unknown = enabled - _MINIMAX_H3_FUSION_NAMES
    if unknown:
        supported = ",".join(sorted(_MINIMAX_H3_FUSION_NAMES))
        raise ValueError(f"Unknown MiniMax H3 fusion(s) {sorted(unknown)}; expected a subset of {supported}.")
    return enabled


def _can_run_minimax_h3_fusion(tensor: torch.Tensor) -> bool:
    """Triton kernels are inference-only and stay outside Dynamo capture.

    The ``HAVE_TRITON`` check makes the eager fallback exact: on a CUDA build
    whose Triton failed to import, an enabled fusion falls back instead of
    hitting the strict wrappers' hard RuntimeError mid-forward.
    """
    return (HAVE_TRITON and tensor.is_cuda and not torch.is_grad_enabled() and not torch.compiler.is_compiling())


class MiniMaxH3RotaryPosEmbed(nn.Module):
    """Three-axis rotary frequencies over packed `(t, h, w)` coordinates."""

    def __init__(self, rope_freq_dim: int, rope_theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (rope_theta**(torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) /
                                       (2 * rope_freq_dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build rotary tensors on the device that owns the packed positions."""
        position_ids = position_ids.to(torch.float32)
        # Analytic rotary positional embedding (RoPE) state is non-persistent,
        # so runtime coordinates own the device after loading or state offload.
        inv_freq = self.inv_freq.to(position_ids.device)
        freqs = position_ids.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        freqs_t, freqs_h, freqs_w = freqs.unbind(dim=1)
        freqs = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos(), freqs.sin()


class MiniMaxH3FeedForward(nn.Module):
    """Bias-free H3 SwiGLU with value-first packed halves."""

    def __init__(
        self,
        hidden_size: int,
        ffn_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        fuse_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.fc_in = ReplicatedLinear(
            hidden_size,
            2 * ffn_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_in",
        )
        self.fc_out = ReplicatedLinear(
            ffn_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_out",
        )
        self.fuse_swiglu = fuse_swiglu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc_in(hidden_states)
        if self.fuse_swiglu and _can_run_minimax_h3_fusion(hidden_states):
            hidden_states = minimax_h3_swiglu(hidden_states)
        else:
            hidden_states, gate = hidden_states.chunk(2, dim=-1)
            hidden_states = hidden_states * F.silu(gate)
        hidden_states, _ = self.fc_out(hidden_states)
        return hidden_states


class MiniMaxH3Attention(nn.Module):
    """Full self-attention over one sequence-parallel packed document."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm_eps: float,
        supported_attention_backends: tuple[AttentionBackendEnum, ...],
        quant_config: QuantizationConfig | None,
        prefix: str,
        fuse_qknorm_rope: bool = False,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.to_q = ReplicatedLinear(
            hidden_size,
            inner_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_q",
        )
        self.to_k = ReplicatedLinear(
            hidden_size,
            inner_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_k",
        )
        self.to_v = ReplicatedLinear(
            hidden_size,
            inner_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_v",
        )
        self.norm_q = nn.RMSNorm(attention_head_dim, eps=qk_norm_eps)
        self.norm_k = nn.RMSNorm(attention_head_dim, eps=qk_norm_eps)
        self.to_out = ReplicatedLinear(
            inner_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_out",
        )
        self.fuse_qknorm_rope = fuse_qknorm_rope
        # VSA carries a learned gate on its pooled-compression branch. The H3
        # checkpoint has no such weight, so the loader zero-initializes it
        # (ALLOWED_NEW_PARAM_PATTERNS) and the branch is exactly disabled
        # until finetuned. Built only when VSA-H3 actually resolves, keeping
        # the FLASH/SDPA paths and their state_dict untouched.
        resolved_backend = get_attn_backend(attention_head_dim,
                                            get_compute_dtype(),
                                            supported_attention_backends=supported_attention_backends)
        use_vsa = resolved_backend.get_name() == "VIDEO_SPARSE_ATTN_H3"
        attention_cls = DistributedAttention_VSA if use_vsa else DistributedAttention
        self.distributed_attention = attention_cls(
            num_heads=num_attention_heads,
            head_size=attention_head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=prefix,
        )
        self.to_gate_compress: ReplicatedLinear | None = None
        # None = unchecked; the first forward tests the loaded weight once and
        # skips the gate branch entirely while it is structurally zero.
        self._gate_compress_active: bool | None = None
        if use_vsa:
            self.to_gate_compress = ReplicatedLinear(
                hidden_size,
                inner_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.to_gate_compress",
            )

    def _gate_active(self) -> bool:
        """True when the compression gate can contribute.

        The zero-initialized (not-yet-finetuned) gate makes the whole branch
        a guaranteed zero — a full GEMM plus a third more all-to-all/tile
        traffic per layer for nothing — so test the loaded weight once and
        cache the answer. Under grad the branch always runs (gradients must
        reach a zero gate for it to ever train).
        """
        if torch.is_grad_enabled():
            return True
        if self._gate_compress_active is None:
            weight = self.to_gate_compress.weight
            # bool() on a DTensor reduction resolves collectively, so every
            # rank caches the same answer.
            self._gate_compress_active = bool((weight != 0).any())
        return self._gate_compress_active

    @staticmethod
    def _apply_rotary_emb(
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Rotate the RoPE prefix while preserving the rest of each attention head."""
        cos, sin = rotary_emb
        rotary_dim = cos.shape[-1]
        hidden_states_rotary = hidden_states[..., :rotary_dim]
        hidden_states_pass = hidden_states[..., rotary_dim:]
        cos = cos.to(hidden_states.dtype)[None, :, None, :]
        sin = sin.to(hidden_states.dtype)[None, :, None, :]
        first_half, second_half = hidden_states_rotary.chunk(2, dim=-1)
        hidden_states_rotated = torch.cat((-second_half, first_half), dim=-1)
        hidden_states_rotary = hidden_states_rotary * cos + hidden_states_rotated * sin
        return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        original_seq_len: int,
    ) -> torch.Tensor:
        query, _ = self.to_q(hidden_states)
        key, _ = self.to_k(hidden_states)
        value, _ = self.to_v(hidden_states)
        query = query.unflatten(-1, (self.num_attention_heads, self.attention_head_dim))
        key = key.unflatten(-1, (self.num_attention_heads, self.attention_head_dim))
        value = value.unflatten(-1, (self.num_attention_heads, self.attention_head_dim))
        if (self.fuse_qknorm_rope and rotary_emb is not None and _can_run_minimax_h3_fusion(query)):
            cos, sin = rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            query = fused_qknorm_rope(query, self.norm_q.weight, cos, sin, self.norm_q.eps)
            key = fused_qknorm_rope(key, self.norm_k.weight, cos, sin, self.norm_k.eps)
        else:
            query = self.norm_q(query)
            key = self.norm_k(key)
            if rotary_emb is not None:
                query = self._apply_rotary_emb(query, rotary_emb)
                key = self._apply_rotary_emb(key, rotary_emb)

        # H3 rotates only 96/128 channels, which the generic `freqs_cis`
        # branch cannot express. Apply it above, then pass no RoPE here.
        extra_attention_kwargs = {}
        if self.to_gate_compress is not None and self._gate_active():
            gate_compress, _ = self.to_gate_compress(hidden_states)
            extra_attention_kwargs["gate_compress"] = gate_compress.unflatten(
                -1, (self.num_attention_heads, self.attention_head_dim))
        hidden_states, _ = self.distributed_attention(
            query,
            key,
            value,
            original_seq_len=original_seq_len,
            freqs_cis=None,
            **extra_attention_kwargs,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states, _ = self.to_out(hidden_states)
        return hidden_states


class MiniMaxH3TokenRefinerBlock(nn.Module):
    """Plain pre-norm Transformer block for the projected text stream."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        supported_attention_backends: tuple[AttentionBackendEnum, ...],
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size,
            num_attention_heads,
            attention_head_dim,
            qk_norm_eps,
            supported_attention_backends,
            quant_config,
            prefix=f"{prefix}.attn",
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = MiniMaxH3FeedForward(
            hidden_size,
            ffn_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.ff",
        )

    def forward(self, hidden_states: torch.Tensor, original_seq_len: int) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), None, original_seq_len)
        return hidden_states + self.ff(self.norm2(hidden_states))


class MiniMaxH3TokenRefiner(nn.Module):
    """Two-block text refiner used before packing the modalities."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        num_layers: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
        supported_attention_backends: tuple[AttentionBackendEnum, ...],
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.refiner_blocks = nn.ModuleList([
            MiniMaxH3TokenRefinerBlock(
                hidden_size,
                num_attention_heads,
                attention_head_dim,
                ffn_dim,
                norm_eps,
                qk_norm_eps,
                supported_attention_backends,
                quant_config,
                prefix=f"{prefix}.refiner_blocks.{index}",
            ) for index in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(hidden_size, eps=final_norm_eps)

    def forward(self, hidden_states: torch.Tensor, original_seq_len: int) -> torch.Tensor:
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, original_seq_len)
        return self.final_norm(hidden_states)


class MiniMaxH3AdaLayerNormModulation(nn.Module):
    """Produce six modulation tables for every `(timestep, modality)` pair."""

    def __init__(
        self,
        time_embed_dim: int,
        hidden_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        apply_silu: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.apply_silu = apply_silu
        self.linear = ReplicatedLinear(
            time_embed_dim,
            6 * hidden_size * MINIMAX_H3_MODALITY_NUM,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.apply_silu:
            temb = F.silu(temb)
        temb, _ = self.linear(temb.to(self.linear.weight.dtype))
        return temb.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


class MiniMaxH3AdaLayerNormOut(nn.Module):
    """Final RMSNorm with per-timestep row modulation."""

    def __init__(
        self,
        hidden_size: int,
        time_embed_dim: int,
        eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        apply_silu: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.apply_silu = apply_silu
        self.linear = ReplicatedLinear(
            time_embed_dim,
            2 * hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.apply_silu:
            temb = F.silu(temb)
        shift_scale, _ = self.linear(temb.to(self.linear.weight.dtype))
        shift, scale = shift_scale.chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)


class MiniMaxH3TransformerBlock(nn.Module):
    """Packed self-attention and feed-forward branches with row-indexed AdaLN."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        supported_attention_backends: tuple[AttentionBackendEnum, ...],
        quant_config: QuantizationConfig | None,
        prefix: str,
        adaln_apply_silu: bool = True,
        fuse_modulate: bool = False,
        fuse_qknorm_rope: bool = False,
        fuse_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size,
            num_attention_heads,
            attention_head_dim,
            qk_norm_eps,
            supported_attention_backends,
            quant_config,
            prefix=f"{prefix}.attn",
            fuse_qknorm_rope=fuse_qknorm_rope,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = MiniMaxH3FeedForward(
            hidden_size,
            ffn_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.ff",
            fuse_swiglu=fuse_swiglu,
        )
        self.adaln_proj = MiniMaxH3AdaLayerNormModulation(
            time_embed_dim,
            hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.adaln_proj",
            apply_silu=adaln_apply_silu,
        )
        self.fuse_modulate = fuse_modulate

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            t.to(hidden_states.dtype) for t in self.adaln_proj(temb))

        use_modulate_fusion = self.fuse_modulate and _can_run_minimax_h3_fusion(hidden_states)
        if use_modulate_fusion:
            norm_hidden_states = fused_rmsnorm_modulate(
                hidden_states,
                self.norm1.weight,
                scale_msa,
                shift_msa,
                adaln_indices,
                self.norm1.eps,
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (
                1.0 + scale_msa.index_select(0, adaln_indices)) + shift_msa.index_select(0, adaln_indices)
        attention_output = self.attn(norm_hidden_states, rotary_emb, original_seq_len)
        if use_modulate_fusion:
            hidden_states, norm_hidden_states = fused_residual_gate_rmsnorm_modulate(
                hidden_states,
                attention_output,
                gate_msa,
                self.norm2.weight,
                scale_mlp,
                shift_mlp,
                adaln_indices,
                self.norm2.eps,
            )
        else:
            hidden_states = hidden_states + gate_msa.index_select(0, adaln_indices) * attention_output
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (
                1.0 + scale_mlp.index_select(0, adaln_indices)) + shift_mlp.index_select(0, adaln_indices)
        feed_forward_output = self.ff(norm_hidden_states)
        return hidden_states + gate_mlp.index_select(0, adaln_indices) * feed_forward_output


class MiniMaxH3Transformer3DModel(BaseDiT):
    """Joint H3 Transformer over one padless text/audio/video document.

    The layout builder validates semantic rows before denoising. Sequence-
    parallel padding is transport-only and `DistributedAttention` trims it
    before attention.
    """

    _fsdp_shard_conditions = _CFG._fsdp_shard_conditions
    _compile_conditions = _CFG._compile_conditions
    _supported_attention_backends = _CFG._supported_attention_backends
    param_names_mapping = _CFG.param_names_mapping
    reverse_param_names_mapping = _CFG.reverse_param_names_mapping
    lora_param_names_mapping = _CFG.lora_param_names_mapping
    _keep_in_fp32_modules = frozenset({
        "proj_in",
        "audio_proj_in",
        "time_embedder",
        "proj_out",
        "audio_proj_out",
        "rope",
    })

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        """Keep the released input, timestep, and output projections in FP32.

        Factorized AdaLN uses FP16; BF16 is ~1.7x worse there.
        """
        # Precedence: the factorized-AdaLN FP16 pin wins over
        # uniform_parameter_dtype on purpose. Under FSDP's one-dtype rule the
        # resulting mix hard-fails at load time, which beats silently training
        # AdaLN in BF16. Rank-reduced checkpoints are inference artifacts --
        # train from the full-rank release.
        if getattr(self, "adaln_rank", None) is not None and (
                ".adaln_proj." in name or name.startswith(("norm_out.linear.", "adaln_basis."))):
            return torch.float16
        if self.config.uniform_parameter_dtype:
            return default_dtype
        return torch.float32 if name.split(".", 1)[0] in self._keep_in_fp32_modules else default_dtype

    def __init__(self, config: MiniMaxH3Config, hf_config: dict[str, Any]) -> None:
        super().__init__(config, hf_config)
        arch = config.arch_config
        self.enabled_fusions = _enabled_minimax_h3_fusions()
        if self.enabled_fusions:
            if HAVE_TRITON:
                logger.info(
                    "MiniMax H3 inference fusions enabled: %s (CUDA inference-only; grad-enabled and "
                    "torch.compile-captured forwards fall back to eager).",
                    ",".join(sorted(self.enabled_fusions)))
            else:
                logger.warning(
                    "FASTVIDEO_MINIMAX_H3_FUSIONS requested %s but Triton is unavailable; "
                    "every forward stays on the eager path.", ",".join(sorted(self.enabled_fusions)))
        sp_world_size = get_sp_world_size() if model_parallel_is_initialized() else 1
        if arch.num_attention_heads % sp_world_size:
            raise ValueError(f"MiniMax H3 attention heads ({arch.num_attention_heads}) must be divisible by "
                             f"sequence parallel size ({sp_world_size}).")

        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.in_channels
        self.patch_size = tuple(int(value) for value in arch.patch_size)
        video_patch_dim = arch.in_channels * math.prod(arch.patch_size)

        self.proj_in = ReplicatedLinear(
            video_patch_dim,
            arch.hidden_size,
            bias=True,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.proj_in",
        )
        self.audio_proj_in = ReplicatedLinear(
            arch.audio_in_channels,
            arch.hidden_size,
            bias=True,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.audio_proj_in",
        )
        self.context_embedder = ReplicatedLinear(
            arch.text_dim,
            arch.hidden_size,
            bias=True,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.context_embedder",
        )
        self.time_proj = Timesteps(
            num_channels=arch.freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedder = MLP(
            arch.freq_dim,
            arch.time_embed_hidden_dim,
            arch.time_embed_dim,
            act_type="silu",
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.time_embedder",
        )
        self.adaln_rank: int | None = arch.adaln_rank
        if self.adaln_rank is not None and config.uniform_parameter_dtype:
            raise ValueError(
                "Rank-reduced AdaLN checkpoints (adaln_rank set) cannot be trained: "
                "uniform_parameter_dtype needs one dtype for every trainable "
                "parameter, but factorized AdaLN weights are pinned to FP16 "
                "(BF16 reconstructs them ~1.7x worse). Fine-tune the full-rank "
                "checkpoint instead, then re-fit the basis with "
                "scripts/checkpoint_conversion/convert_minimax_h3_adaln_rank.py.")
        adaln_dim = self.adaln_rank or arch.time_embed_dim
        self.adaln_basis = ReplicatedLinear(
            arch.time_embed_dim,
            self.adaln_rank,
            bias=False,
            quant_config=None,
            prefix=f"{config.prefix}.adaln_basis",
        ) if self.adaln_rank else None

        self.rope = MiniMaxH3RotaryPosEmbed(arch.rope_freq_dim, arch.rope_theta)
        # per-generation caches for loop-invariant work (see _rotary_for /
        # _refined_text); plain attrs, never in state_dict
        self._rope_cache: tuple | None = None
        self._text_cache: tuple | None = None
        self.token_refiner = MiniMaxH3TokenRefiner(
            arch.hidden_size,
            arch.num_attention_heads,
            arch.attention_head_dim,
            arch.ffn_dim,
            arch.num_refiner_layers,
            arch.norm_eps,
            arch.qk_norm_eps,
            arch.final_norm_eps,
            # The refiner attends over the text stream only; the packed-sequence
            # VSA backend must never be selected for it.
            tuple(backend for backend in self.supported_attention_backends
                  if backend != AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3),
            config.quant_config,
            prefix=f"{config.prefix}.token_refiner",
        )
        self.transformer_blocks = nn.ModuleList([
            MiniMaxH3TransformerBlock(
                arch.hidden_size,
                arch.num_attention_heads,
                arch.attention_head_dim,
                arch.ffn_dim,
                adaln_dim,
                arch.norm_eps,
                arch.qk_norm_eps,
                self.supported_attention_backends,
                config.quant_config,
                prefix=f"{config.prefix}.transformer_blocks.{index}",
                adaln_apply_silu=self.adaln_rank is None,
                fuse_modulate="modulate" in self.enabled_fusions,
                fuse_qknorm_rope="qknorm_rope" in self.enabled_fusions,
                fuse_swiglu="swiglu" in self.enabled_fusions,
            ) for index in range(arch.num_layers)
        ])
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            arch.hidden_size,
            adaln_dim,
            arch.final_norm_eps,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.norm_out",
            apply_silu=self.adaln_rank is None,
        )
        self.proj_out = ReplicatedLinear(
            arch.hidden_size,
            video_patch_dim,
            bias=True,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.proj_out",
        )
        self.audio_proj_out = ReplicatedLinear(
            arch.hidden_size,
            arch.audio_in_channels,
            bias=True,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.audio_proj_out",
        )
        self.__post_init__()

    def prepare_for_compile(self) -> None:
        """Pipeline hook, called once right before torch.compile wraps the blocks.

        Dynamo capture traces the eager branch of every fusion guard, so an
        enabled ``FASTVIDEO_MINIMAX_H3_FUSIONS`` set is silently inert inside
        compiled block forwards (H3 compiles per-block by default). Say so
        once instead of leaving the flag looking active.
        """
        if self.enabled_fusions:
            logger.warning(
                "torch.compile is enabled for MiniMax H3, so the requested inference fusions (%s) are "
                "inert inside compiled block forwards; the compiled eager path runs instead.",
                ",".join(sorted(self.enabled_fusions)))

    def materialize_non_persistent_buffers(
        self,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Rebuild analytic RoPE state on the checkpoint loader device.

        RoPE frequencies are absent from the checkpoint, so meta-device model
        construction and device moves must derive the buffer from architecture
        fields before the first forward pass.
        """
        del dtype
        if self.rope.inv_freq.is_meta or self.rope.inv_freq.device != device:
            arch = self.config.arch_config
            inv_freq = 1.0 / (arch.rope_theta
                              **(torch.arange(0, 2 * arch.rope_freq_dim, 2, device=device, dtype=torch.float32) /
                                 (2 * arch.rope_freq_dim)))
            self.rope._buffers["inv_freq"] = inv_freq

    def _rotary_for(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin depend only on positions; the denoising stage holds one
        position_ids tensor for the whole loop, so cache per layout and
        pre-cast once instead of rebuilding every step and re-casting in
        every block. Holding the source tensor keeps the identity check safe
        against id reuse."""
        if torch.compiler.is_compiling():
            cos, sin = self.rope(position_ids)
            return cos.to(dtype), sin.to(dtype)
        cached = self._rope_cache
        if (cached is not None and cached[0] is position_ids and cached[1] == dtype
                and cached[2][0].device == position_ids.device):
            return cached[2]
        cos, sin = self.rope(position_ids)
        value = (cos.to(dtype), sin.to(dtype))
        self._rope_cache = (position_ids, dtype, value)
        return value

    def _refined_text(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """The prompt embedding is constant across the denoising loop and the
        refiner blocks are timestep-free, so refine once per generation
        (saves two transformer blocks and four collectives per step). Skipped
        under grad (training needs the graph) and under compile."""
        cacheable = not torch.is_grad_enabled() and not torch.compiler.is_compiling()
        if (cacheable and self._text_cache is not None and self._text_cache[0] is encoder_hidden_states
                and self._text_cache[1].device == encoder_hidden_states.device):
            return self._text_cache[1]
        text_embeds, _ = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
        text_original_seq_len = text_embeds.shape[1]
        sp_world_size = get_sp_world_size() if model_parallel_is_initialized() else 1
        if sp_world_size > 1:
            text_embeds, _ = sequence_model_parallel_shard(text_embeds, dim=1)
        text_embeds = self.token_refiner(text_embeds, text_original_seq_len)
        if sp_world_size > 1:
            text_embeds = sequence_model_parallel_all_gather_with_unpad(
                text_embeds,
                text_original_seq_len,
                dim=1,
            )
        if cacheable:
            self._text_cache = (encoder_hidden_states, text_embeds)
        return text_embeds

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict video and audio velocities from one caller-defined packed layout."""
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"position_ids must have shape (seq_len, 3), got {tuple(position_ids.shape)}.")
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length, ) or timestep_indices.shape != (sequence_length, ):
            raise ValueError("token_tags and timestep_indices must both match the packed sequence length.")
        if hidden_states.shape[1] != video_indices.numel():
            raise ValueError("hidden_states row count must match video_indices.")
        if audio_hidden_states.shape[1] != audio_indices.numel():
            raise ValueError("audio_hidden_states row count must match audio_indices.")
        if encoder_hidden_states.shape[1] != text_indices.numel():
            raise ValueError("encoder_hidden_states row count must match text_indices.")

        video_embeds, _ = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        audio_embeds, _ = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.weight.dtype))
        text_embeds = self._refined_text(encoder_hidden_states)
        rotary_emb = self._rotary_for(position_ids, text_embeds.dtype)
        sp_world_size = get_sp_world_size() if model_parallel_is_initialized() else 1

        # text/video/audio indices partition [0, sequence_length), so the
        # uninitialized buffer is fully overwritten; in-place index_copy_ avoids
        # the three full-buffer clones out-of-place index_copy would make.
        packed_hidden_states = text_embeds.new_empty((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        packed_hidden_states.index_copy_(1, text_indices, text_embeds)
        packed_hidden_states.index_copy_(1, video_indices, video_embeds.to(text_embeds.dtype))
        packed_hidden_states.index_copy_(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.fc_in.weight.dtype))
        if self.adaln_basis is not None:
            temb, _ = self.adaln_basis(F.silu(temb).to(self.adaln_basis.weight.dtype))
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
        local_timestep_indices = timestep_indices
        original_seq_len = sequence_length

        if sp_world_size > 1:
            packed_hidden_states, _ = sequence_model_parallel_shard(packed_hidden_states, dim=1)
            rotary_cos, _ = sequence_model_parallel_shard(rotary_emb[0], dim=0)
            rotary_sin, _ = sequence_model_parallel_shard(rotary_emb[1], dim=0)
            adaln_indices, _ = sequence_model_parallel_shard(adaln_indices, dim=0)
            local_timestep_indices, _ = sequence_model_parallel_shard(local_timestep_indices, dim=0)
            rotary_emb = (rotary_cos, rotary_sin)

        # The eager driver owns profiling markers while each block's compiled
        # forward owns the graph that the marker surrounds.
        for block_index, block in enumerate(self.transformer_blocks):
            with nvtx_range(f"minimax_h3.transformer_block.{block_index}"):
                packed_hidden_states = block(
                    packed_hidden_states,
                    temb,
                    adaln_indices,
                    rotary_emb,
                    original_seq_len,
                )

        packed_hidden_states = self.norm_out(
            packed_hidden_states,
            temb,
            local_timestep_indices,
        ).to(self.proj_out.weight.dtype)
        video_output, _ = self.proj_out(packed_hidden_states)
        audio_output, _ = self.audio_proj_out(packed_hidden_states)
        if sp_world_size > 1:
            video_output = sequence_model_parallel_all_gather_with_unpad(video_output, original_seq_len, dim=1)
            audio_output = sequence_model_parallel_all_gather_with_unpad(audio_output, original_seq_len, dim=1)
        video_output = video_output.index_select(1, video_indices)
        audio_output = audio_output.index_select(1, audio_indices)

        return video_output, audio_output


EntryClass = MiniMaxH3Transformer3DModel
