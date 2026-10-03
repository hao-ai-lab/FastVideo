# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 VFM Transformer (Phase 2b.1 skeleton).

This module lands the parameter-named module tree of
``Cosmos3VFMTransformer`` together with two standalone math utilities used
by both the FastVideo port and the Phase 2a parity tests:

* ``compute_mrope_position_ids_text`` / ``compute_mrope_position_ids_vision`` —
  the unified-3D mRoPE position ID generators ported verbatim from
  ``vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py`` lines
  113-177 (HEAD ``8536f5b1421f``).

* ``Cosmos3VFMTransformer.patchify`` /
  ``Cosmos3VFMTransformer.unpatchify`` — the
  ``[B,C,T,H,W] <-> [B, T*hp*wp, p*p*C]`` patch tokenizer, ported from
  the same reference lines 1009-1036.

Every per-layer ``forward()`` raises ``NotImplementedError("Phase 2b.2")``
because the layer math (RoPE application, QK-norm, GQA SDPA, UND/GEN
attention plumbing) is intentionally deferred to Phase 2b.2. Only the
module composition + tensor shape contract is exercised here, enough to
make 4 Tier-A parity tests pass:

* ``test_cosmos3_mrope_parity.py::test_compute_mrope_position_ids_text_and_vision``
* ``test_cosmos3_patchify_unpatchify_parity.py::test_patchify_unpatchify_roundtrip``
* ``test_cosmos3_patchify_unpatchify_parity.py::test_patchify_default_patch_size``
* ``test_cosmos3_state_dict_keys.py::test_fastvideo_cosmos3_dit_module_tree_param_names``

The constructor signature mirrors the upstream
``Cosmos3VFMTransformer(od_config, *, temporal_compression_factor=None)``
contract so the Phase 2a tests (which were authored against the upstream
shape) can drive it directly. Phase 2b.2+ will adapt this to FastVideo's
``BaseDiT(config, hf_config)`` loader path via the ``TransformerLoader``.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.configs.models.dits.cosmos3 import Cosmos3ArchConfig, Cosmos3VideoConfig
from fastvideo.models.dits.base import BaseDiT


EntryClass = ["Cosmos3VFMTransformer"]


def _tf_config_get(config: Any, key: str, default: Any) -> Any:
    """Read a value from a dict, dataclass, or simple namespace.

    Mirrors ``transformer_cosmos3._tf_config_get`` at reference line 77.
    """
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def compute_mrope_position_ids_text(
    num_tokens: int,
    temporal_offset: int,
) -> tuple[torch.Tensor, int]:
    """Generate 3D mRoPE position IDs for text tokens.

    Text tokens broadcast a single monotonically-increasing position-ID
    sequence across all three (t, h, w) axes. Verbatim port of
    ``transformer_cosmos3.compute_mrope_position_ids_text`` (reference
    lines 113-124).
    """
    ids = torch.arange(num_tokens, dtype=torch.long) + temporal_offset
    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()
    return mrope_ids, temporal_offset + num_tokens


def compute_mrope_position_ids_vision(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    base_temporal_compression_factor: int | None = None,
    enable_fps_modulation: bool = True,
    start_frame_offset: int = 0,
) -> tuple[torch.Tensor, int]:
    """Generate 3D mRoPE position IDs for vision tokens.

    Builds a ``(t, h, w)`` position grid (Qwen3-VL style, spatial indices
    reset per temporal segment) flattened in t-major order. Optionally
    modulates the temporal axis by ``base_fps / tcf * (1 / (fps / tcf))``
    so two clips at different FPS retain wall-clock-aligned temporal
    positions. Verbatim port of
    ``transformer_cosmos3.compute_mrope_position_ids_vision`` (reference
    lines 127-177).
    """
    fps_modulation = enable_fps_modulation and fps is not None

    if fps_modulation:
        assert fps is not None
        tps = fps / temporal_compression_factor
        effective_base_tcf = (base_temporal_compression_factor
                              if base_temporal_compression_factor is not None
                              else temporal_compression_factor)
        base_tps = base_fps / effective_base_tcf
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        t_index = (((frame_indices + start_frame_offset) / tps * base_tps + temporal_offset)
                   .view(-1, 1).expand(-1, grid_h * grid_w).flatten())
    else:
        t_index = (torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
                   + int(temporal_offset) + start_frame_offset)

    h_index = (torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten())
    w_index = (torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten())

    if fps_modulation:
        mrope_ids = torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)

    next_offset = math.floor(mrope_ids.max().item()) + 1
    return mrope_ids, next_offset


class _Qwen3VLTextRMSNorm(nn.Module):
    """Qwen3-VL / T5-style RMSNorm with an ``eps`` and a learnable ``weight``.

    Mirrors ``Qwen3VLTextRMSNorm`` at reference line 89-107 so checkpoint
    keys match the canonical ``*.norm*.weight`` / ``*.{q,k}_norm.weight``
    parameter naming.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Phase 2b.2")


class _TimestepEmbedder(nn.Module):
    """Sinusoidal-to-vector timestep embedder used by the GEN pathway.

    Skeleton of ``TimestepEmbedder`` at reference line 257-283. The
    ``linear_1``/``linear_2`` parameter names match the upstream
    checkpoint so the Phase 5 remap can copy weights without renaming.
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size

        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=target_dtype) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Phase 2b.2")


class _Cosmos3GatedMLP(nn.Module):
    """Gated-MLP block with ``gate_proj`` / ``up_proj`` / ``down_proj`` names.

    Skeleton of ``Cosmos3GatedMLP`` at reference line 288-326. The three
    linear-layer parameter names are load-bearing for the checkpoint
    remap; Phase 2b.2 will swap them for FastVideo-native
    ``ReplicatedLinear`` shards.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Phase 2b.2")


class _Cosmos3SkeletonAttention(nn.Module):
    """Shared skeleton for UND causal- and GEN cross-attention.

    Exposes the q/k/v/o projections at the upstream parameter-name
    locations (``*.{q,k,v,o}_proj`` and ``*.{q,k}_norm``) with the
    GQA-correct in/out dimensions
    (``q_proj`` writes ``num_attention_heads * head_dim`` channels;
    ``k_proj``/``v_proj`` write ``num_key_value_heads * head_dim``
    channels). Phase 2b.2 will replace the bare ``nn.Linear`` shells with
    ``ColumnParallelLinear``-equivalent FastVideo layers and implement
    the RoPE + SDPA forward.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.q_norm = _Qwen3VLTextRMSNorm(head_dim, eps=rms_norm_eps, dtype=dtype)
        self.k_norm = _Qwen3VLTextRMSNorm(head_dim, eps=rms_norm_eps, dtype=dtype)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("Phase 2b.2")


class _Cosmos3UndDecoderLayer(nn.Module):
    """UND (understanding) decoder layer skeleton.

    The submodule names ``self_attn``, ``input_layernorm``,
    ``post_attention_layernorm``, ``mlp`` mirror the upstream layer
    composition at reference line 624-678.
    """

    def __init__(self, arch: Cosmos3ArchConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.self_attn = _Cosmos3SkeletonAttention(
            hidden_size=arch.hidden_size,
            num_attention_heads=arch.num_attention_heads,
            num_key_value_heads=arch.num_key_value_heads,
            head_dim=arch.head_dim,
            rms_norm_eps=arch.rms_norm_eps,
            dtype=dtype,
        )
        self.input_layernorm = _Qwen3VLTextRMSNorm(arch.hidden_size, eps=arch.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = _Qwen3VLTextRMSNorm(arch.hidden_size, eps=arch.rms_norm_eps, dtype=dtype)
        self.mlp = _Cosmos3GatedMLP(arch.hidden_size, arch.intermediate_size)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("Phase 2b.2")


class _Cosmos3GenDecoderLayer(nn.Module):
    """GEN (generation) decoder layer skeleton.

    Substitutes ``cross_attention`` for ``self_attn`` (mirrors upstream
    reference line 681-751); otherwise identical to the UND layer in
    naming. The ``layer_idx`` is captured for the eventual cached-KV
    path so Phase 2b.2 can route per-layer UND keys/values correctly.
    """

    def __init__(self, layer_idx: int, arch: Cosmos3ArchConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.cross_attention = _Cosmos3SkeletonAttention(
            hidden_size=arch.hidden_size,
            num_attention_heads=arch.num_attention_heads,
            num_key_value_heads=arch.num_key_value_heads,
            head_dim=arch.head_dim,
            rms_norm_eps=arch.rms_norm_eps,
            dtype=dtype,
        )
        self.input_layernorm = _Qwen3VLTextRMSNorm(arch.hidden_size, eps=arch.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = _Qwen3VLTextRMSNorm(arch.hidden_size, eps=arch.rms_norm_eps, dtype=dtype)
        self.mlp = _Cosmos3GatedMLP(arch.hidden_size, arch.intermediate_size)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("Phase 2b.2")


class _Cosmos3LanguageModel(nn.Module):
    """Understanding-pathway container.

    Holds ``embed_tokens``, ``layers``, and a final ``norm``. The
    submodule names mirror the upstream reference at line 757-831 so the
    checkpoint remap target keys ``language_model.embed_tokens.*``,
    ``language_model.layers.{i}.*``, and ``language_model.norm.*`` are
    populated as-is by the FastVideo module tree.
    """

    def __init__(self, arch: Cosmos3ArchConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(arch.vocab_size, arch.hidden_size)
        self.layers = nn.ModuleList(
            [_Cosmos3UndDecoderLayer(arch, dtype) for _ in range(arch.num_hidden_layers)])
        self.norm = _Qwen3VLTextRMSNorm(arch.hidden_size, eps=arch.rms_norm_eps, dtype=dtype)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("Phase 2b.2")


class Cosmos3VFMTransformer(BaseDiT):
    """Cosmos3 VFM Transformer — UND language model + GEN denoising layers.

    Phase 2b.1 ships only the module-tree skeleton and the two pure math
    utilities (``patchify``/``unpatchify``). All per-layer ``forward``
    methods raise ``NotImplementedError("Phase 2b.2")``; the DiT-level
    ``forward`` likewise. The constructor signature
    ``(od_config, *, temporal_compression_factor=None)`` matches the
    upstream reference at ``transformer_cosmos3.py`` line 899 so the
    Phase 2a parity tests can drive it directly.

    Phase 2b.2 will:
      * Replace the per-layer ``nn.Linear``s with FastVideo-native
        ``ReplicatedLinear`` / ``DistributedAttention`` plumbing.
      * Implement ``forward`` (UND once, GEN per-step with cached K/V).
      * Adapt the constructor to the FastVideo
        ``BaseDiT(config, hf_config)`` loader contract via
        ``TransformerLoader``.
    """

    _fsdp_shard_conditions = Cosmos3VideoConfig().arch_config._fsdp_shard_conditions
    _compile_conditions = Cosmos3VideoConfig().arch_config._compile_conditions
    param_names_mapping = Cosmos3VideoConfig().arch_config.param_names_mapping
    reverse_param_names_mapping: dict[str, str] = {}

    def __init__(
        self,
        od_config: object | None = None,
        *,
        temporal_compression_factor: int | None = None,
    ) -> None:
        nn.Module.__init__(self)

        model_config = getattr(od_config, "tf_model_config", None) if od_config is not None else None
        rope_scaling = _tf_config_get(model_config, "rope_scaling", {}) or {}

        self.hidden_size = int(_tf_config_get(model_config, "hidden_size", 4096))
        self.num_hidden_layers = int(_tf_config_get(model_config, "num_hidden_layers", 36))
        self.num_attention_heads = int(_tf_config_get(model_config, "num_attention_heads", 32))
        self.num_key_value_heads = int(_tf_config_get(model_config, "num_key_value_heads", 8))
        self.head_dim = int(_tf_config_get(model_config, "head_dim", 128))
        self.intermediate_size = int(_tf_config_get(model_config, "intermediate_size", 12288))
        self.vocab_size = int(_tf_config_get(model_config, "vocab_size", 151936))
        self.rms_norm_eps = float(_tf_config_get(model_config, "rms_norm_eps", 1e-6))
        self.rope_theta = float(_tf_config_get(model_config, "rope_theta", 5_000_000))
        self.mrope_section = list(rope_scaling.get("mrope_section", [24, 20, 20]))
        self.latent_patch_size = int(_tf_config_get(model_config, "latent_patch_size", 2))
        self.latent_channel_size = int(_tf_config_get(model_config, "latent_channel", 48))
        self.timestep_scale = float(_tf_config_get(model_config, "timestep_scale", 0.001))
        self.base_fps = float(_tf_config_get(model_config, "base_fps", 24.0))
        if temporal_compression_factor is None:
            resolved_tcf = int(_tf_config_get(model_config, "temporal_compression_factor", 4))
        else:
            resolved_tcf = int(temporal_compression_factor)
        self.temporal_compression_factor = resolved_tcf
        self.enable_fps_modulation = bool(_tf_config_get(model_config, "enable_fps_modulation", True))
        self.temporal_modality_margin = int(
            _tf_config_get(model_config, "unified_3d_mrope_temporal_modality_margin", 15000))

        self.patch_latent_dim = (self.latent_patch_size**2) * self.latent_channel_size
        self.num_channels_latents = self.latent_channel_size

        dtype = getattr(od_config, "dtype", torch.bfloat16) if od_config is not None else torch.bfloat16

        arch = Cosmos3ArchConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            mrope_section=self.mrope_section,
            latent_patch_size=self.latent_patch_size,
            latent_channel=self.latent_channel_size,
            timestep_scale=self.timestep_scale,
            base_fps=self.base_fps,
            temporal_compression_factor=self.temporal_compression_factor,
            enable_fps_modulation=self.enable_fps_modulation,
            temporal_modality_margin=self.temporal_modality_margin,
            in_channels=self.latent_channel_size,
            out_channels=self.latent_channel_size,
        )

        self.language_model = _Cosmos3LanguageModel(arch, dtype)
        self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
        self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
        self.time_embedder = _TimestepEmbedder(self.hidden_size, target_dtype=dtype)
        self.gen_layers = nn.ModuleList(
            [_Cosmos3GenDecoderLayer(i, arch, dtype) for i in range(self.num_hidden_layers)])
        self.norm_moe_gen = _Qwen3VLTextRMSNorm(self.hidden_size, eps=self.rms_norm_eps, dtype=dtype)

    def _pad_to_patch_size(self, h: int, w: int) -> tuple[int, int, int, int]:
        """Return ``(hp, wp, H_padded, W_padded)`` for ``latent_patch_size`` padding.

        Mirrors ``Cosmos3VFMTransformer._pad_to_patch_size`` at reference
        lines 1002-1007.
        """
        p = self.latent_patch_size
        h_padded = ((h + p - 1) // p) * p
        w_padded = ((w + p - 1) // p) * p
        return h_padded // p, w_padded // p, h_padded, w_padded

    def patchify(self, latents: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        """``[B, C, t, h, w] -> [B, t*hp*wp, p*p*C]``.

        Pads ``h``/``w`` up to a multiple of ``latent_patch_size`` before
        reshaping. Verbatim port of
        ``Cosmos3VFMTransformer.patchify`` (reference lines 1009-1021).
        """
        batch_size = latents.shape[0]
        p = self.latent_patch_size
        c = self.latent_channel_size
        hp, wp, h_padded, w_padded = self._pad_to_patch_size(h, w)

        if h_padded != h or w_padded != w:
            latents = F.pad(latents, (0, w_padded - w, 0, h_padded - h))

        x = latents.reshape(batch_size, c, t, hp, p, wp, p)
        x = x.permute(0, 2, 3, 5, 4, 6, 1)
        return x.reshape(batch_size, t * hp * wp, p * p * c)

    def unpatchify(self, tokens: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        """``[B, t*hp*wp, p*p*C] -> [B, C, t, h, w]``, cropping ``h``/``w`` padding.

        Verbatim port of ``Cosmos3VFMTransformer.unpatchify`` (reference
        lines 1023-1036).
        """
        batch_size = tokens.shape[0]
        p = self.latent_patch_size
        c = self.latent_channel_size
        hp, wp, h_padded, w_padded = self._pad_to_patch_size(h, w)

        x = tokens.reshape(batch_size, t, hp, wp, p, p, c)
        x = x.permute(0, 6, 1, 2, 4, 3, 5)
        x = x.reshape(batch_size, c, t, h_padded, w_padded)

        if h_padded != h or w_padded != w:
            x = x[:, :, :, :h, :w]
        return x

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("Phase 2b.2")
