# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 video super-resolution (SR) DiT.

A text-free Kandinsky6 DiT: no text tower and no cross-attention; a learned ``pooled_bias`` stands in for the pooled
empty-caption embedding the model was trained with.  The blocks reuse the Kandinsky6 building blocks
(``models/dits/kandinsky6.py``).  Input ``[B, T, H, W, 2C + 1]`` (noised latent | anchor | anchor mask), output
``[B, T, H, W, out_visual_dim]`` (``n_grid * C`` for the pi-Flow checkpoint).

The residual stream stays in the model's ``compute_dtype`` between blocks, as in the Diffusers model. Modulation,
norm and time-embedding maths happen in fp32 internally and are cast back to ``compute_dtype`` right after each gated
residual add. Sequence parallelism is not supported (dense ``LocalAttention``), and NABLA sparse attention is not wired.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from fastvideo.configs.models.dits.kandinsky6_sr import Kandinsky6SRArchConfig, Kandinsky6SRConfig
from fastvideo.layers.layernorm import LayerNormScaleShift
from fastvideo.layers.quantization import QuantizationConfig
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.kandinsky6 import (
    Kandinsky6Attention,
    Kandinsky6FeedForward,
    Kandinsky6Modulation,
    Kandinsky6OutLayer,
    Kandinsky6RoPE3D,
    Kandinsky6TimeEmbeddings,
    Kandinsky6VisualEmbeddings,
    _build_rotary_freqs,
)
from fastvideo.platforms import AttentionBackendEnum

logger = init_logger(__name__)

_ARCH_CONFIG_DEFAULTS = Kandinsky6SRConfig().arch_config

# Parameters used in fp32 maths (modulation, time embedding, QK norms) stay fp32 when the DiT is loaded in bf16, so a
# fp32 checkpoint is not rounded there.
_FP32_PARAM_MARKERS = ("pooled_bias", "time_embeddings.", ".visual_modulation.", "out_layer.modulation.",
                       "query_norm.", "key_norm.")


class Kandinsky6SRDecoderBlock(nn.Module):
    """Text-free decoder block: modulated self-attention + modulated feed-forward (6 modulation params).

    ``x = (x + gate * attention(LN(x) * (1 + scale) + shift)).to(compute_dtype)`` per sub-layer: the gated
    residual add happens in fp32 (matching the norm/modulation maths) and is cast back to ``compute_dtype``
    immediately after, so the residual stream tracks the model's working dtype between blocks instead of staying
    fp32 forever after the first block.
    """

    def __init__(self,
                 model_dim: int,
                 time_dim: int,
                 ff_dim: int,
                 head_dim: int,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
                 prefix: str = "",
                 quant_config: QuantizationConfig | None = None):
        super().__init__()
        self.visual_modulation = Kandinsky6Modulation(time_dim, model_dim, 6)
        self.self_attention_norm = LayerNormScaleShift(model_dim,
                                                       norm_type="layer",
                                                       eps=1e-5,
                                                       elementwise_affine=False,
                                                       dtype=torch.float32,
                                                       compute_dtype=torch.float32)
        self.self_attention = Kandinsky6Attention(model_dim,
                                                  head_dim,
                                                  supported_attention_backends=supported_attention_backends,
                                                  prefix=f"{prefix}.self_attention",
                                                  quant_config=quant_config)
        self.feed_forward_norm = LayerNormScaleShift(model_dim,
                                                     norm_type="layer",
                                                     eps=1e-5,
                                                     elementwise_affine=False,
                                                     dtype=torch.float32,
                                                     compute_dtype=torch.float32)
        self.feed_forward = Kandinsky6FeedForward(model_dim, ff_dim, prefix=f"{prefix}.feed_forward",
                                                  quant_config=quant_config)

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor, rope: torch.Tensor,
                compute_dtype: torch.dtype) -> torch.Tensor:
        # `compute_dtype` comes from the model's never-quantized embedding layer (see
        # Kandinsky6SRTransformer3DModel.compute_dtype), not a local linear here: under FP8,
        # `convert_model_to_fp8` pops `.weight` off quantized ``ReplicatedLinear`` layers like
        # ``self_attention.to_query``, so reading its dtype directly would raise AttributeError.
        sa_params, ff_params = torch.chunk(self.visual_modulation(time_embed).unsqueeze(dim=1), 2, dim=-1)

        shift, scale, gate = torch.chunk(sa_params, 3, dim=-1)
        h = self.self_attention_norm(x.float(), shift=shift, scale=scale, convert_modulation_dtype=True)
        h = self.self_attention(h.to(compute_dtype), rotary_emb=rope)
        x = (x.float() + gate.float() * h.float()).to(compute_dtype)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        h = self.feed_forward_norm(x.float(), shift=shift, scale=scale, convert_modulation_dtype=True)
        h = self.feed_forward(h.to(compute_dtype))
        return (x.float() + gate.float() * h.float()).to(compute_dtype)


class Kandinsky6SROutLayer(Kandinsky6OutLayer):
    """``Kandinsky6OutLayer`` for an fp32 residual stream: modulate in fp32, cast to the parameter dtype, project."""

    def forward(self, visual_embed: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        shift, scale = torch.chunk(self.modulation(time_embed).unsqueeze(dim=1), 2, dim=-1)
        h = (self.norm(visual_embed.float()) * (scale.float()[:, None, None] + 1.0) + shift.float()[:, None, None])
        x, _ = self.out_layer(h.to(self.out_layer.weight.dtype))

        batch_size, duration, height, width, _ = x.shape
        x = (x.view(
            batch_size,
            duration,
            height,
            width,
            -1,
            self.patch_size[0],
            self.patch_size[1],
            self.patch_size[2],
        ).permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(1, 2).flatten(2, 3).flatten(3, 4))
        return x


class Kandinsky6SRTransformer3DModel(BaseDiT):
    """``forward(hidden_states [B, T, H, W, 2C + 1], timestep [B], visual_rope_pos, scale_factor)``
    -> ``[B, T, H, W, out_visual_dim]``."""

    _fsdp_shard_conditions = _ARCH_CONFIG_DEFAULTS._fsdp_shard_conditions
    _compile_conditions = _ARCH_CONFIG_DEFAULTS._compile_conditions
    param_names_mapping = _ARCH_CONFIG_DEFAULTS.param_names_mapping
    reverse_param_names_mapping = _ARCH_CONFIG_DEFAULTS.reverse_param_names_mapping
    lora_param_names_mapping = _ARCH_CONFIG_DEFAULTS.lora_param_names_mapping
    _supported_attention_backends = _ARCH_CONFIG_DEFAULTS._supported_attention_backends

    def __init__(self, config: Kandinsky6SRConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch = config.arch_config
        assert isinstance(arch, Kandinsky6SRArchConfig)
        self._reject_unknown_config_keys(arch, hf_config)
        quant_config = config.quant_config
        self.quant_config = quant_config

        head_dim = sum(arch.axes_dims)
        self.in_visual_dim = arch.in_visual_dim
        self.model_dim = arch.model_dim
        self.patch_size = arch.patch_size
        self.input_channels = arch.input_channels

        self.time_embeddings = Kandinsky6TimeEmbeddings(arch.model_dim, arch.time_dim)
        self.pooled_bias = nn.Parameter(torch.zeros(arch.time_dim))

        self.visual_embeddings = Kandinsky6VisualEmbeddings(self.input_channels, arch.model_dim, arch.patch_size)
        self.visual_rope_embeddings = Kandinsky6RoPE3D(arch.axes_dims)
        self.visual_transformer_blocks = nn.ModuleList([
            Kandinsky6SRDecoderBlock(arch.model_dim,
                                     arch.time_dim,
                                     arch.ff_dim,
                                     head_dim,
                                     self._supported_attention_backends,
                                     prefix=f"{config.prefix}.visual_transformer_blocks.{i}",
                                     quant_config=quant_config) for i in range(arch.num_visual_blocks)
        ])
        self.out_layer = Kandinsky6SROutLayer(arch.model_dim, arch.time_dim, arch.out_visual_dim, arch.patch_size)
        if arch.attention_params is not None:
            self._warn_if_sparse_attention_requested(arch.attention_params)
        if arch.nabla_threshold is not None:
            logger.warning(
                "Kandinsky6SR checkpoint requests NABLA sparse attention via nabla_threshold=%s; the SR DiT runs "
                "dense attention (sparse NABLA is not wired), so results differ slightly from the reference.",
                arch.nabla_threshold)

        self.gradient_checkpointing = False
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.__post_init__()

    @staticmethod
    def _reject_unknown_config_keys(arch: Kandinsky6SRArchConfig, hf_config: dict[str, Any]) -> None:
        """``update_model_arch`` drops keys the arch config does not declare; a real checkpoint flag we do not know
        would then silently run a different model, so refuse it."""
        unknown = sorted(set(hf_config) - arch.known_config_keys())
        if unknown:
            raise ValueError(f"Kandinsky6SR transformer/config.json has unsupported keys {unknown}; refusing to "
                             "silently ignore them.")

    @staticmethod
    def _warn_if_sparse_attention_requested(attention_params: dict) -> None:
        for size, params in attention_params.items():
            kind = params.get("type") if isinstance(params, dict) else getattr(params, "type", None)
            if kind not in (None, "flash", "sdpa", "dense"):
                logger.warning(
                    "Kandinsky6SR checkpoint requests attention type %r for visual size %s; the SR DiT runs dense "
                    "attention (sparse NABLA is not wired), so results differ slightly from the reference.", kind,
                    size)
                return

    @property
    def compute_dtype(self) -> torch.dtype:
        """Parameter dtype of the linear stacks (the dtype the DiT input is cast to)."""
        return self.visual_embeddings.in_layer.weight.dtype

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        if any(marker in name for marker in _FP32_PARAM_MARKERS):
            return torch.float32
        return default_dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        visual_rope_pos: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor],
        scale_factor: tuple[float, ...] = (1.0, 1.0, 1.0),
        return_dict: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the DiT.

        Args:
            hidden_states: ``[B, T, H, W, 2C + 1]`` noised latent | anchor | anchor mask.
            timestep: ``[B]`` scheduler timestep (sigma * 1000).
            visual_rope_pos: ``(arange(T), arange(H // ph), arange(W // pw))`` position indices (shared by the batch).
            scale_factor: RoPE frequency scale per axis (t, h, w).
        """
        if kwargs:
            raise TypeError(f"Kandinsky6SRTransformer3DModel.forward got unsupported arguments {sorted(kwargs)}; the "
                            "SR DiT has no text / sparse-attention / LQ inputs.")
        if hidden_states.ndim != 5:
            raise ValueError(f"hidden_states must be [B,T,H,W,C], got {tuple(hidden_states.shape)}")
        if hidden_states.shape[-1] != self.input_channels:
            raise ValueError(
                f"hidden_states has {hidden_states.shape[-1]} channels but the trained input layer expects "
                f"{self.input_channels} (noised latent | anchor | anchor mask).")

        time_embed = self.time_embeddings(timestep)
        time_embed = time_embed + self.pooled_bias.to(time_embed.dtype)

        visual_embed = self.visual_embeddings(hidden_states.to(self.compute_dtype).contiguous())
        batch_size, duration, height, width, dim = visual_embed.shape
        visual_rope = self.visual_rope_embeddings((batch_size, duration, height, width), visual_rope_pos, scale_factor)
        visual_embed = visual_embed.flatten(1, 3)
        visual_rope = visual_rope.flatten(1, 3)

        compute_dtype = self.compute_dtype
        for block in self.visual_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                visual_embed = torch.utils.checkpoint.checkpoint(block, visual_embed, time_embed, visual_rope,
                                                                 compute_dtype, use_reentrant=False)
            else:
                visual_embed = block(visual_embed, time_embed, visual_rope, compute_dtype)

        visual_embed = visual_embed.reshape(batch_size, duration, height, width, dim)
        return self.out_layer(visual_embed, time_embed)

    def materialize_non_persistent_buffers(self, device: torch.device, dtype: torch.dtype | None = None) -> None:
        rope = self.visual_rope_embeddings
        for i, (axes_dim, ax_max_pos) in enumerate(zip(rope.axes_dims, rope.max_pos, strict=True)):
            name = f"args_{i}"
            buf = getattr(rope, name, None)
            if isinstance(buf, torch.Tensor) and buf.is_meta:
                freq = _build_rotary_freqs(axes_dim // 2, rope.max_period).to(device=device)
                pos = torch.arange(ax_max_pos, dtype=freq.dtype, device=device)
                rope._buffers[name] = torch.outer(pos, freq)

        time_module = self.time_embeddings
        if isinstance(time_module.freqs, torch.Tensor) and time_module.freqs.is_meta:
            time_module.freqs = _build_rotary_freqs(time_module.model_dim // 2, time_module.max_period).to(device=device)


EntryClass = Kandinsky6SRTransformer3DModel
